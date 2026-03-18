"""
models/diffusion_model.py
==========================
SGLDv2 Diffusion Model

Key difference from v1:
  v1 → fine-tuned ControlNet (350M trainable params, destroys SD prior)
  v2 → SketchAdapter injection (22M trainable params, SD stays frozen)

SD components: VAE + UNet + CLIP → ALL FROZEN
Only trained: SketchAdapter (token encoder + injected k/v projections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional, Union

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from models.sketch_adapter import SketchAdapter


# ─────────────────────────────────────────────────────────────────────────────
# Default model IDs
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"


# ─────────────────────────────────────────────────────────────────────────────
# SGLDv2 Model  (training)
# ─────────────────────────────────────────────────────────────────────────────

class SGLDv2Model(nn.Module):
    """
    Full SGLDv2 model for training.

    Architecture:
      1. CLIP text encoder    → text embeddings          [FROZEN]
      2. VAE encoder          → photo latents            [FROZEN]
      3. DDPM noise scheduler → noisy latents            [no weights]
      4. SketchAdapter        → sketch tokens            [TRAINED ~22M params]
      5. UNet (with injected  → denoised latents         [FROZEN + adapter hooks]
         sketch attention)
      6. VAE decoder          → final image              [FROZEN]
    """

    def __init__(
        self,
        base_model:    str   = DEFAULT_BASE_MODEL,
        img_size:      int   = 256,
        num_tokens:    int   = 16,
        attn_scale:    float = 1.0,
        device:        str   = "cuda",
        dtype:         torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device = device
        self.dtype  = dtype

        # ── Load SD components ────────────────────────────────────────────
        print("  [SGLDv2] Loading CLIP tokenizer + encoder …")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            base_model, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model, subfolder="text_encoder"
        )

        print("  [SGLDv2] Loading VAE …")
        self.vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae"
        )

        print("  [SGLDv2] Loading UNet …")
        self.unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet"
        )

        print("  [SGLDv2] Loading noise scheduler …")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            base_model, subfolder="scheduler"
        )

        # ── Freeze ALL SD components ───────────────────────────────────────
        for model in [self.text_encoder, self.vae, self.unet]:
            model.requires_grad_(False)
            model.eval()

        # ── Build and inject SketchAdapter ─────────────────────────────────
        print("  [SGLDv2] Building SketchAdapter …")
        self.sketch_adapter = SketchAdapter(
            in_channels    = 5,
            token_dim      = 768,
            num_tokens     = num_tokens,
            num_enc_layers = 4,
            img_size       = img_size,
            attn_scale     = attn_scale,
        )
        # Inject decoupled cross-attention into every UNet attn2 layer
        self.sketch_adapter.inject_into_unet(self.unet)

        # Move all to device
        self.to(device)
        print("  [SGLDv2] Ready. SD is fully frozen.")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def encode_text(self, captions: List[str]) -> torch.Tensor:
        """CLIP text encoding → (B, 77, 768)."""
        tokens = self.tokenizer(
            captions,
            padding    = "max_length",
            max_length = self.tokenizer.model_max_length,
            truncation = True,
            return_tensors = "pt",
        ).input_ids.to(self.device)
        with torch.no_grad():
            return self.text_encoder(tokens)[0]

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        VAE encode → latents.
        images : (B, 3, H, W) float [-1, 1]
        """
        with torch.no_grad():
            latents = self.vae.encode(
                images.to(dtype=self.vae.dtype)
            ).latent_dist.sample()
            return latents * self.vae.config.scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        VAE decode → images.
        returns: (B, 3, H, W) float [-1, 1]
        """
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            images = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample
        return images

    def add_noise(self, latents, noise, timesteps):
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    def trainable_parameters(self):
        """Only SketchAdapter params are trained."""
        return self.sketch_adapter.trainable_parameters()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_latents:   torch.Tensor,
        timesteps:       torch.Tensor,
        encoder_hidden:  torch.Tensor,
        structure_maps:  torch.Tensor,
    ) -> torch.Tensor:
        """
        SGLDv2 training forward pass.

        Parameters
        ----------
        noisy_latents  : (B, 4, H/8, W/8)  – latents + noise
        timesteps      : (B,)
        encoder_hidden : (B, 77, 768)       – CLIP text embeddings
        structure_maps : (B, 5, H, W)       – edge+depth+seg tensor

        Returns
        -------
        noise_pred : (B, 4, H/8, W/8)
        """
        # 1. Encode structure maps → sketch tokens
        sketch_tokens = self.sketch_adapter.encode(
            structure_maps.to(dtype=self.dtype)
        )                                              # (B, num_tokens, 768)

        # 2. Store tokens in injected attention layers
        self.sketch_adapter.set_sketch_tokens(self.unet, sketch_tokens)

        # 3. UNet forward (injected layers use stored sketch tokens)
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states = encoder_hidden,
        ).sample

        return noise_pred

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save only the SketchAdapter (SD doesn't need saving — it's frozen)."""
        self.sketch_adapter.save(path)

    def load_adapter(self, path: str) -> None:
        """Load a previously trained SketchAdapter."""
        self.sketch_adapter.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# SGLDv2 Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SGLDv2InferencePipeline:
    """
    Inference pipeline for SGLDv2.

    Loads the trained SketchAdapter into a standard SD pipeline.
    Uses DDIM/UniPC scheduler for fast, high-quality generation.
    """

    def __init__(
        self,
        adapter_path:    str,
        base_model:      str  = DEFAULT_BASE_MODEL,
        device:          str  = "cuda",
        num_tokens:      int  = 16,
        img_size:        int  = 256,
        attn_scale:      float = 1.0,
    ):
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        print("  [InferencePipeline] Loading SGLDv2 model …")
        self.model = SGLDv2Model(
            base_model  = base_model,
            img_size    = img_size,
            num_tokens  = num_tokens,
            attn_scale  = attn_scale,
            device      = device,
            dtype       = dtype,
        )
        self.model.load_adapter(adapter_path)
        self.model.eval()

        # Fast scheduler for inference
        self.model.noise_scheduler = UniPCMultistepScheduler.from_config(
            self.model.noise_scheduler.config
        )
        print("  [InferencePipeline] Ready.")

    @torch.inference_mode()
    def generate(
        self,
        structure_maps:   torch.Tensor,
        prompt:           str   = "a realistic photo, highly detailed",
        negative_prompt:  str   = "low quality, blurry, sketch, cartoon",
        guidance_scale:   float = 7.5,
        num_steps:        int   = 30,
        seed:             int   = 42,
        width:            int   = 256,
        height:           int   = 256,
    ) -> Image.Image:
        """
        Generate an image conditioned on structure maps + text prompt.

        structure_maps : (1, 5, H, W) or (5, H, W) float tensor
        """
        if structure_maps.dim() == 3:
            structure_maps = structure_maps.unsqueeze(0)

        structure_maps = structure_maps.to(self.device)
        B = structure_maps.shape[0]

        # Encode sketch tokens
        sketch_tokens = self.model.sketch_adapter.encode(
            structure_maps.to(dtype=self.model.dtype)
        )
        self.model.sketch_adapter.set_sketch_tokens(
            self.model.unet, sketch_tokens
        )

        # Text embeddings (with CFG: conditional + unconditional)
        text_emb  = self.model.encode_text([prompt]          * B)
        uncond    = self.model.encode_text([negative_prompt] * B)
        text_emb  = torch.cat([uncond, text_emb])

        # Initial noise
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents   = torch.randn(
            (B, 4, height // 8, width // 8),
            generator = generator,
            device    = self.device,
            dtype     = self.model.dtype,
        )
        latents  *= self.model.noise_scheduler.init_noise_sigma

        # Denoising loop
        self.model.noise_scheduler.set_timesteps(num_steps)
        for t in self.model.noise_scheduler.timesteps:
            # Duplicate latents for CFG
            latent_in = torch.cat([latents] * 2)
            latent_in = self.model.noise_scheduler.scale_model_input(
                latent_in, t
            )

            # Duplicate sketch tokens for CFG
            tokens_cfg = torch.cat([sketch_tokens] * 2)
            self.model.sketch_adapter.set_sketch_tokens(
                self.model.unet, tokens_cfg
            )

            noise_pred = self.model.unet(
                latent_in, t,
                encoder_hidden_states = text_emb,
            ).sample

            # CFG guidance
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (
                noise_cond - noise_uncond
            )

            latents = self.model.noise_scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # Decode
        images = self.model.decode_latents(latents)
        image  = (images[0].float() * 0.5 + 0.5).clamp(0, 1)
        return Image.fromarray(
            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
        )
