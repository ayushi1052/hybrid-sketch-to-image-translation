"""
models/diffusion_model.py
==========================
SGLDv2 Diffusion Model — with exception handling

Gradient-safe design:
  sketch_tokens → cat([text, sketch]) → frozen UNet → loss → grad → adapter ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from models.sketch_adapter import SketchAdapter

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"


class SGLDv2Model(nn.Module):
    """
    SGLDv2: Frozen SD + Trained SketchAdapter.
    Gradient flow: loss ← UNet ← cat([text, sketch_tokens]) ← adapter ✓
    """

    def __init__(
        self,
        base_model: str  = DEFAULT_BASE_MODEL,
        img_size:   int  = 256,
        num_tokens: int  = 16,
        device:     str  = "cuda",
        dtype:      torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device = device
        self.dtype  = dtype

        # ── Load SD components ─────────────────────────────────────────────
        print("  [SGLDv2] Loading CLIP tokenizer + encoder …")
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                base_model, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                base_model, subfolder="text_encoder"
            )
        except OSError as e:
            raise OSError(
                f"Could not load CLIP from '{base_model}': {e}\n"
                "  Check internet connection or HuggingFace token."
            ) from e

        print("  [SGLDv2] Loading VAE …")
        try:
            self.vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        except OSError as e:
            raise OSError(f"Could not load VAE: {e}") from e

        print("  [SGLDv2] Loading UNet …")
        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                base_model, subfolder="unet"
            )
        except OSError as e:
            raise OSError(f"Could not load UNet: {e}") from e
        except torch.cuda.OutOfMemoryError:
            raise torch.cuda.OutOfMemoryError(
                "GPU out of memory loading UNet.\n"
                "  Try: --mixed_precision fp16  or free up GPU memory."
            )

        print("  [SGLDv2] Loading noise scheduler …")
        try:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                base_model, subfolder="scheduler"
            )
        except Exception as e:
            raise RuntimeError(f"Could not load scheduler: {e}") from e

        # ── Freeze SD ──────────────────────────────────────────────────────
        for m in [self.text_encoder, self.vae, self.unet]:
            m.requires_grad_(False)
            m.eval()

        # ── SketchAdapter ──────────────────────────────────────────────────
        print("  [SGLDv2] Building SketchAdapter …")
        try:
            self.sketch_adapter = SketchAdapter(
                in_channels    = 5,
                token_dim      = 768,
                num_tokens     = num_tokens,
                num_enc_layers = 4,
                img_size       = img_size,
            )
        except Exception as e:
            raise RuntimeError(f"SketchAdapter init failed: {e}") from e

        self.to(device)
        print("  [SGLDv2] Ready. SD is fully frozen.")

    def encode_text(self, captions: List[str]) -> torch.Tensor:
        """CLIP text → (B, 77, 768). Raises on failure."""
        try:
            tokens = self.tokenizer(
                captions,
                padding    = "max_length",
                max_length = self.tokenizer.model_max_length,
                truncation = True,
                return_tensors = "pt",
            ).input_ids.to(self.device)
            with torch.no_grad():
                return self.text_encoder(tokens)[0]
        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {e}") from e

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """VAE encode → latents. images: (B,3,H,W) in [-1,1]."""
        try:
            with torch.no_grad():
                lat = self.vae.encode(
                    images.to(dtype=self.vae.dtype)
                ).latent_dist.sample()
                return lat * self.vae.config.scaling_factor
        except torch.cuda.OutOfMemoryError:
            raise torch.cuda.OutOfMemoryError(
                "OOM during VAE encoding. Try smaller batch_size."
            )
        except Exception as e:
            raise RuntimeError(f"VAE encoding failed: {e}") from e

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE decode → images in [-1,1]."""
        try:
            latents = latents / self.vae.config.scaling_factor
            with torch.no_grad():
                return self.vae.decode(
                    latents.to(dtype=self.vae.dtype)
                ).sample
        except Exception as e:
            raise RuntimeError(f"VAE decoding failed: {e}") from e

    def add_noise(self, latents, noise, timesteps):
        try:
            return self.noise_scheduler.add_noise(latents, noise, timesteps)
        except Exception as e:
            raise RuntimeError(f"add_noise failed: {e}") from e

    def trainable_parameters(self):
        return list(self.sketch_adapter.parameters())

    def forward(
        self,
        noisy_latents:  torch.Tensor,
        timesteps:      torch.Tensor,
        encoder_hidden: torch.Tensor,
        structure_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient-safe forward:
          structure_maps → adapter → sketch_tokens
          cat([text, sketch_tokens]) → frozen UNet → noise_pred
          loss ← noise_pred ← UNet ← sketch_tokens ← adapter ✓
        """
        # Encode structure → sketch tokens  [TRAINED — grad flows here]
        try:
            sketch_tokens = self.sketch_adapter(structure_maps)  # (B, 16, 768)
        except Exception as e:
            raise RuntimeError(f"SketchAdapter forward failed: {e}") from e

        # Concatenate with text tokens
        try:
            combined = torch.cat([encoder_hidden, sketch_tokens], dim=1)
        except Exception as e:
            raise RuntimeError(
                f"Token concatenation failed: {e}\n"
                f"  encoder_hidden: {encoder_hidden.shape}, "
                f"sketch_tokens: {sketch_tokens.shape}"
            ) from e

        # UNet forward — gradient flows back through combined → sketch_tokens
        try:
            noise_pred = self.unet(
                noisy_latents, timesteps,
                encoder_hidden_states = combined,
            ).sample
        except torch.cuda.OutOfMemoryError:
            raise torch.cuda.OutOfMemoryError(
                "OOM during UNet forward. "
                "Try: lower --batch_size or --img_size"
            )
        except Exception as e:
            raise RuntimeError(f"UNet forward failed: {e}") from e

        return noise_pred

    def save(self, path: str) -> None:
        """Save SketchAdapter — raises with clear message on failure."""
        try:
            import os
            os.makedirs(path, exist_ok=True)
            self.sketch_adapter.save(path)
        except PermissionError as e:
            raise PermissionError(f"Cannot save to '{path}': {e}") from e
        except Exception as e:
            raise RuntimeError(f"Model save failed: {e}") from e

    def load_adapter(self, path: str) -> None:
        """Load SketchAdapter — raises with clear message on failure."""
        import os
        adapter_file = os.path.join(path, "sketch_adapter.pt")
        if not os.path.isfile(adapter_file):
            raise FileNotFoundError(
                f"No 'sketch_adapter.pt' found in: '{path}'\n"
                f"  Available files: {list(os.listdir(path)) if os.path.isdir(path) else 'directory not found'}"
            )
        try:
            self.sketch_adapter.load(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter from '{path}': {e}"
            ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SGLDv2InferencePipeline:
    def __init__(
        self,
        adapter_path: str,
        base_model:   str  = DEFAULT_BASE_MODEL,
        device:       str  = "cuda",
        num_tokens:   int  = 16,
        img_size:     int  = 256,
    ):
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        print("  [InferencePipeline] Loading SGLDv2 …")
        self.model = SGLDv2Model(
            base_model = base_model,
            img_size   = img_size,
            num_tokens = num_tokens,
            device     = device,
            dtype      = dtype,
        )
        self.model.load_adapter(adapter_path)
        self.model.eval()

        try:
            self.model.noise_scheduler = UniPCMultistepScheduler.from_config(
                self.model.noise_scheduler.config
            )
        except Exception as e:
            print(f"  [Warning] Could not switch to UniPC scheduler ({e}). "
                  "Using DDPM.")

        print("  [InferencePipeline] Ready.")

    @torch.inference_mode()
    def generate(
        self,
        structure_maps:  torch.Tensor,
        prompt:          str   = "a realistic photo, highly detailed",
        negative_prompt: str   = "low quality, blurry, sketch, cartoon",
        guidance_scale:  float = 7.5,
        num_steps:       int   = 30,
        seed:            int   = 42,
        width:           int   = 256,
        height:          int   = 256,
    ) -> Image.Image:
        if structure_maps.dim() == 3:
            structure_maps = structure_maps.unsqueeze(0)

        structure_maps = structure_maps.to(self.device)
        B = structure_maps.shape[0]

        # Encode sketch tokens
        try:
            adapter_dtype  = self.model.sketch_adapter.token_encoder.patch_embed[0].weight.dtype
            sketch_tokens  = self.model.sketch_adapter(
                structure_maps.to(dtype=adapter_dtype)
            )
        except Exception as e:
            raise RuntimeError(f"Sketch encoding failed: {e}") from e

        # Text embeddings
        try:
            text_emb = self.model.encode_text([prompt]          * B)
            uncond   = self.model.encode_text([negative_prompt] * B)
        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {e}") from e

        cond_combined   = torch.cat([text_emb, sketch_tokens], dim=1)
        uncond_combined = torch.cat([uncond,   sketch_tokens], dim=1)
        combined_emb    = torch.cat([uncond_combined, cond_combined])

        # Initial noise
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            latents   = torch.randn(
                (B, 4, height // 8, width // 8),
                generator = generator,
                device    = self.device,
                dtype     = self.model.dtype,
            )
            latents  *= self.model.noise_scheduler.init_noise_sigma
        except Exception as e:
            raise RuntimeError(f"Latent init failed: {e}") from e

        # Denoising loop
        try:
            self.model.noise_scheduler.set_timesteps(num_steps)
            for t in self.model.noise_scheduler.timesteps:
                latent_in  = torch.cat([latents] * 2)
                latent_in  = self.model.noise_scheduler.scale_model_input(latent_in, t)
                noise_pred = self.model.unet(
                    latent_in, t, encoder_hidden_states=combined_emb
                ).sample
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents    = self.model.noise_scheduler.step(
                    noise_pred, t, latents
                ).prev_sample
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "GPU OOM during denoising.\n"
                "  Try: --steps 20  or  --device cpu"
            )
        except Exception as e:
            raise RuntimeError(f"Denoising loop failed: {e}") from e

        # Decode
        try:
            imgs = self.model.decode_latents(latents)
            img  = (imgs[0].float() * 0.5 + 0.5).clamp(0, 1)
            return Image.fromarray(
                (img.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
            )
        except Exception as e:
            raise RuntimeError(f"VAE decode failed: {e}") from e