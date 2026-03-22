"""
models/diffusion_model.py
==========================
SGLDv2 — Structure-Guided Latent Diffusion v2

Architecture
────────────
  Sketch → StructureExtractor (Canny + MiDaS + SegFormer)
         → 5-channel structure maps
         → StructureInputProjection (Conv64→Conv128→Conv256→Conv3)
         → ControlNetModel → residuals
         → Frozen UNet + residual injection
         → VAE Decoder → RGB image

Speed / memory optimisations
─────────────────────────────
• UNet and VAE use channels_last memory layout.
• xformers memory-efficient attention on UNet (when available).
• SNR-weighted diffusion loss (Min-SNR-γ=5) for faster convergence.
• Inference caches the control hint once and doubles it for CFG,
  avoiding a redundant InputProjection forward pass.
• UniPC multi-step scheduler at inference: 30 steps ≈ quality of 100 DDPM.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional, Tuple

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from models.sketch_adapter import StructureControlNet

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"


# ─────────────────────────────────────────────────────────────────────────────
# SNR helper for Min-SNR-γ loss weighting
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Signal-to-noise ratio for each timestep.
    Based on: Hang et al., "Efficient Diffusion Training via Min-SNR Weighting"
    https://arxiv.org/abs/2303.09556
    """
    alphas_cp = scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_acp  = alphas_cp[timesteps] ** 0.5
    sqrt_1m   = (1.0 - alphas_cp[timesteps]) ** 0.5
    return (sqrt_acp / sqrt_1m) ** 2   # SNR = alpha² / (1-alpha)²


def snr_loss_weights(scheduler, timesteps: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """
    Min-SNR-γ per-sample loss weights.
    Clips SNR at γ to down-weight high-noise (small-t) steps.
    Returns shape (B,).
    """
    snr  = compute_snr(scheduler, timesteps)
    w    = torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1).values
    return w / snr   # normalise so E[w] ≈ 1


# ─────────────────────────────────────────────────────────────────────────────
# SGLDv2Model
# ─────────────────────────────────────────────────────────────────────────────

class SGLDv2Model(nn.Module):
    """
    SGLDv2: Frozen SD backbone + Trained StructureControlNet.

    Trainable  : StructureControlNet  (InputProjection + ControlNet)
    Frozen     : CLIP text encoder, VAE, UNet
    """

    def __init__(
        self,
        base_model:   str         = DEFAULT_BASE_MODEL,
        img_size:     int         = 256,
        device:       str         = "cuda",
        dtype:        torch.dtype = torch.float16,
        compile_unet: bool        = False,   # experimental; set True for extra speed
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype  = dtype

        # ── CLIP ──────────────────────────────────────────────────────────
        print("  [SGLDv2] Loading CLIP tokenizer + text encoder …")
        try:
            self.tokenizer    = CLIPTokenizer.from_pretrained(
                base_model, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                base_model, subfolder="text_encoder"
            )
        except OSError as e:
            raise OSError(
                f"Could not load CLIP from '{base_model}': {e}\n"
                "  Check connection or set HF_TOKEN."
            ) from e

        # ── VAE ───────────────────────────────────────────────────────────
        print("  [SGLDv2] Loading VAE …")
        try:
            self.vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        except OSError as e:
            raise OSError(f"Could not load VAE: {e}") from e

        # ── UNet ──────────────────────────────────────────────────────────
        print("  [SGLDv2] Loading UNet …")
        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                base_model, subfolder="unet"
            )
        except OSError as e:
            raise OSError(f"Could not load UNet: {e}") from e
        except torch.cuda.OutOfMemoryError:
            raise torch.cuda.OutOfMemoryError(
                "GPU OOM loading UNet. Try: --mixed_precision fp16"
            )

        # ── Noise scheduler ───────────────────────────────────────────────
        print("  [SGLDv2] Loading DDPM scheduler …")
        try:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                base_model, subfolder="scheduler"
            )
        except Exception as e:
            raise RuntimeError(f"Could not load scheduler: {e}") from e

        # ── Freeze backbone ───────────────────────────────────────────────
        for m in [self.text_encoder, self.vae, self.unet]:
            m.requires_grad_(False).eval()

        # ── Speed: xformers + channels_last on UNet and VAE ───────────────
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            print("  [SGLDv2] UNet xformers: ENABLED")
        except Exception:
            print("  [SGLDv2] xformers not available for UNet.")
        try:
            self.unet = self.unet.to(memory_format=torch.channels_last)
            self.vae  = self.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

        # ── Optional UNet compile (experimental) ─────────────────────────
        if compile_unet:
            try:
                self.unet = torch.compile(self.unet, mode="reduce-overhead")
                print("  [SGLDv2] UNet torch.compile: ENABLED")
            except Exception:
                pass

        # ── StructureControlNet (trainable) ───────────────────────────────
        print("  [SGLDv2] Building StructureControlNet …")
        try:
            self.control_net = StructureControlNet(
                in_channels = 5,
                unet        = self.unet,
            )
        except Exception as e:
            raise RuntimeError(f"StructureControlNet init failed: {e}") from e

        self.to(device)
        print("  [SGLDv2] Ready. SD backbone frozen.")

    # ── Text encoding ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """(B,) list of strings → (B, 77, 768) embeddings."""
        try:
            ids = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            return self.text_encoder(ids)[0]
        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {e}") from e

    # ── VAE helpers ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [-1,1] → VAE latents (B,4,H//8,W//8)."""
        try:
            imgs = images.to(dtype=self.vae.dtype,
                             memory_format=torch.channels_last)
            return (self.vae.encode(imgs).latent_dist.sample()
                    * self.vae.config.scaling_factor)
        except torch.cuda.OutOfMemoryError:
            raise torch.cuda.OutOfMemoryError("OOM in VAE encode. Lower --batch_size.")
        except Exception as e:
            raise RuntimeError(f"VAE encode failed: {e}") from e

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE latents → (B,3,H,W) in [-1,1]."""
        try:
            latents = (latents / self.vae.config.scaling_factor).to(
                dtype=self.vae.dtype, memory_format=torch.channels_last
            )
            return self.vae.decode(latents).sample
        except Exception as e:
            raise RuntimeError(f"VAE decode failed: {e}") from e

    def add_noise(self, latents, noise, timesteps) -> torch.Tensor:
        try:
            return self.noise_scheduler.add_noise(latents, noise, timesteps)
        except Exception as e:
            raise RuntimeError(f"add_noise failed: {e}") from e

    # ── Trainable parameters ──────────────────────────────────────────────────

    def trainable_parameters(self):
        return list(self.control_net.parameters())

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_latents:  torch.Tensor,
        timesteps:      torch.Tensor,
        encoder_hidden: torch.Tensor,
        structure_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        ControlNet → residuals → frozen UNet → noise_pred.
        Gradient flows: loss ← UNet ← residuals ← ControlNet ← InputProj ✓
        """
        try:
            down_res, mid_res = self.control_net(
                noisy_latents  = noisy_latents,
                timesteps      = timesteps,
                encoder_hidden = encoder_hidden,
                structure_maps = structure_maps,
            )
        except Exception as e:
            raise RuntimeError(f"ControlNet forward failed: {e}") from e

        try:
            return self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states           = encoder_hidden,
                down_block_additional_residuals = down_res,
                mid_block_additional_residual   = mid_res,
            ).sample
        except torch.cuda.OutOfMemoryError:
            raise torch.cuda.OutOfMemoryError(
                "OOM in UNet. Try: --batch_size 2 or --img_size 256"
            )
        except Exception as e:
            raise RuntimeError(f"UNet forward failed: {e}") from e

    # ── SNR-weighted loss ─────────────────────────────────────────────────────

    def diffusion_loss(
        self,
        noise_pred:    torch.Tensor,
        noise_target:  torch.Tensor,
        timesteps:     torch.Tensor,
        snr_gamma:     float = 5.0,
    ) -> torch.Tensor:
        """
        Min-SNR-γ weighted MSE loss.
        Down-weights easy high-noise steps; improves convergence speed.
        """
        per_sample = F.mse_loss(
            noise_pred.float(), noise_target.float(), reduction="none"
        ).mean(dim=[1, 2, 3])   # (B,)
        try:
            weights = snr_loss_weights(self.noise_scheduler, timesteps, snr_gamma)
            return (per_sample * weights).mean()
        except Exception:
            return per_sample.mean()   # plain MSE fallback

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        try:
            self.control_net.save(path)
        except Exception as e:
            raise RuntimeError(f"Model save failed: {e}") from e

    def load_adapter(self, path: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: '{path}'\n"
                "  Train first:  python train.py --data_dir ./dataset"
            )
        try:
            self.control_net.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load from '{path}': {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SGLDv2InferencePipeline:
    """
    Stateless inference: loads SGLDv2 + runs denoising with CFG + ControlNet.

    Optimisations
    ─────────────
    • Control hint computed once and doubled for classifier-free guidance
      (avoids a redundant InputProjection forward pass).
    • UniPC multi-step scheduler: 30 steps gives DDPM-100 quality.
    • Runs in fp16 on CUDA throughout.
    """

    FIXED_PROMPT     = "a realistic photo, highly detailed, sharp focus, 8k"
    FIXED_NEG_PROMPT = (
        "low quality, blurry, sketch, cartoon, line drawing, "
        "deformed, ugly, watermark, text"
    )

    def __init__(
        self,
        adapter_path: str,
        base_model:   str        = DEFAULT_BASE_MODEL,
        device:       str        = "cuda",
        img_size:     int        = 256,
    ) -> None:
        self.device   = device
        self.img_size = img_size
        dtype = torch.float16 if device == "cuda" else torch.float32

        print("  [InferencePipeline] Loading SGLDv2 …")
        self.model = SGLDv2Model(
            base_model = base_model,
            img_size   = img_size,
            device     = device,
            dtype      = dtype,
        )
        self.model.load_adapter(adapter_path)
        self.model.eval()

        # Swap to fast multi-step scheduler
        try:
            self.model.noise_scheduler = UniPCMultistepScheduler.from_config(
                self.model.noise_scheduler.config
            )
            print("  [InferencePipeline] UniPC scheduler: ENABLED")
        except Exception as e:
            print(f"  [InferencePipeline] Could not switch to UniPC ({e}). Using DDPM.")

        print("  [InferencePipeline] Ready.")

    @torch.inference_mode()
    def generate(
        self,
        structure_maps:     torch.Tensor,
        prompt:             Optional[str]   = None,
        negative_prompt:    Optional[str]   = None,
        guidance_scale:     float           = 7.5,
        num_steps:          int             = 30,
        seed:               int             = 42,
        conditioning_scale: float           = 1.0,
    ) -> Image.Image:
        """
        Args
        ----
        structure_maps  : (5, H, W) or (B, 5, H, W) float [0, 1]
        conditioning_scale : ControlNet weight; 0.5–2.0 for style variation
        """
        if structure_maps.dim() == 3:
            structure_maps = structure_maps.unsqueeze(0)

        structure_maps = structure_maps.to(self.device)
        B = structure_maps.shape[0]

        prompt          = prompt          or self.FIXED_PROMPT
        negative_prompt = negative_prompt or self.FIXED_NEG_PROMPT
        width = height  = self.img_size

        # ── Text embeddings ───────────────────────────────────────────────
        cond_emb   = self.model.encode_text([prompt]          * B)
        uncond_emb = self.model.encode_text([negative_prompt] * B)

        # ── Control hint — compute ONCE, reuse for both cond/uncond ───────
        try:
            proj     = self.model.control_net.input_proj
            raw_proj = getattr(proj, "_orig_mod", proj)
            dtype_p  = raw_proj.conv64[0].weight.dtype
            hint     = proj(structure_maps.to(dtype=dtype_p))   # (B, 3, H, W)
            hint_2x  = torch.cat([hint] * 2)                    # (2B, 3, H, W)
        except Exception as e:
            raise RuntimeError(f"Control hint encoding failed: {e}") from e

        # ── Initial latent noise ──────────────────────────────────────────
        gen     = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (B, 4, height // 8, width // 8),
            generator = gen,
            device    = self.device,
            dtype     = self.model.dtype,
        ) * self.model.noise_scheduler.init_noise_sigma

        # ── Denoising loop ────────────────────────────────────────────────
        self.model.noise_scheduler.set_timesteps(num_steps)
        combined_emb = torch.cat([uncond_emb, cond_emb])   # (2B, 77, 768)
        cn = getattr(self.model.control_net.controlnet, "_orig_mod",
                     self.model.control_net.controlnet)

        try:
            for t in self.model.noise_scheduler.timesteps:
                lat_in  = torch.cat([latents] * 2)
                lat_in  = self.model.noise_scheduler.scale_model_input(lat_in, t)

                # ControlNet forward
                down_res, mid_res = cn(
                    sample                = lat_in,
                    timestep              = t,
                    encoder_hidden_states = combined_emb,
                    controlnet_cond       = hint_2x,
                    conditioning_scale    = conditioning_scale,
                    return_dict           = False,
                )

                # Frozen UNet with injected residuals
                noise_pred = self.model.unet(
                    lat_in, t,
                    encoder_hidden_states           = combined_emb,
                    down_block_additional_residuals = down_res,
                    mid_block_additional_residual   = mid_res,
                ).sample

                # Classifier-free guidance
                u, c       = noise_pred.chunk(2)
                noise_pred = u + guidance_scale * (c - u)

                latents = self.model.noise_scheduler.step(
                    noise_pred, t, latents
                ).prev_sample

        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "GPU OOM during denoising. Try: --steps 20 or --device cpu"
            )
        except Exception as e:
            raise RuntimeError(f"Denoising loop failed: {e}") from e

        # ── VAE decode ────────────────────────────────────────────────────
        try:
            imgs = self.model.decode_latents(latents)
            img  = (imgs[0].float() * 0.5 + 0.5).clamp(0, 1)
            arr  = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
            return Image.fromarray(arr)
        except Exception as e:
            raise RuntimeError(f"VAE decode failed: {e}") from e
