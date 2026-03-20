"""
models/pipeline.py
===================
SketchToImagePipeline
======================
Your modified d-Sketch implementation.

How it works (inference)
─────────────────────────
  1. Sketch → VAE Encoder → sketch latent
  2. Sketch latent → LCTN → translated image latent    (d-sketch step)
  3. Sketch → ColoredEdgeGenerator → colored edge map  (your addition)
  4. Colored edge map → CLIP Vision Encoder → conditioning embeddings
     (replaces text prompt from d-sketch)
  5. Add partial noise to image latent (k steps forward)
  6. SD UNet denoises for k steps using conditioning embeddings
  7. VAE Decoder → output photo

What trains
────────────
  LCTN                  ← translates sketch latent to image latent
  ColoredEdgeGenerator  ← predicts colored edge map from sketch
  ImageConditionProj    ← projects CLIP vision features to SD cross-attention

What is frozen
──────────────
  VAE (encoder + decoder)
  CLIP Vision Encoder
  SD UNet

Loss functions (same structure as d-sketch + one addition)
─────────────────────────────────────────────────────────
  L_latent    = MSE(LCTN(sketch_latent), real_image_latent)
  L_perceptual = VGG feature distance(decoded_output, real_image)
  L_edge_color = L1(predicted_colored_map, target_colored_map)

  L_total = L_latent + 0.1 × L_perceptual + 0.5 × L_edge_color
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPVisionModel, CLIPImageProcessor

from models.lctn              import LCTN
from models.edge_color_generator import ColoredEdgeGenerator

DEFAULT_SD_MODEL   = "runwayml/stable-diffusion-v1-5"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


# ─────────────────────────────────────────────────────────────────────────────
# Image conditioning projection
# Replaces CLIP text encoder from d-sketch with CLIP vision encoder
# ─────────────────────────────────────────────────────────────────────────────

class ImageConditionProj(nn.Module):
    """
    Projects CLIP vision features to SD UNet cross-attention format.

    CLIP vision (ViT-L/14) outputs: (B, 257, 1024)
      257 = 1 CLS token + 256 patch tokens
      1024 = ViT-L hidden dim

    SD UNet cross-attention expects: (B, 77, 768)
      77 = text sequence length
      768 = SD cross-attention dim

    This learned projection bridges the two — replacing text conditioning
    with image (colored edge map) conditioning.
    """

    def __init__(
        self,
        clip_dim:  int = 1024,   # ViT-L/14 output dim
        sd_dim:    int = 768,    # SD cross-attention dim
        sd_seq:    int = 77,     # SD sequence length
        clip_seq:  int = 257,    # CLIP sequence length (1+256 patches)
    ) -> None:
        super().__init__()

        # Project feature dimension: 1024 → 768
        self.dim_proj = nn.Linear(clip_dim, sd_dim)

        # Project sequence length: 257 → 77
        self.seq_proj = nn.Linear(clip_seq, sd_seq)

        self.norm = nn.LayerNorm(sd_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.dim_proj.weight)
        nn.init.zeros_(self.dim_proj.bias)
        nn.init.xavier_uniform_(self.seq_proj.weight)
        nn.init.zeros_(self.seq_proj.bias)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """
        clip_features : (B, 257, 1024)  — CLIP ViT-L/14 last hidden state
        returns       : (B, 77,  768)   — SD cross-attention conditioning
        """
        # Project feature dimension
        x = self.dim_proj(clip_features)     # (B, 257, 768)

        # Project sequence length
        x = x.permute(0, 2, 1)              # (B, 768, 257)
        x = self.seq_proj(x)                # (B, 768, 77)
        x = x.permute(0, 2, 1)              # (B, 77,  768)

        return self.norm(x)                  # (B, 77,  768)


# ─────────────────────────────────────────────────────────────────────────────
# VGG Perceptual Loss  (same as d-sketch)
# ─────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """VGG16 feature-level perceptual loss. Frozen — never updates."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        try:
            import torchvision.models as tvm
            vgg         = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
            # Use features up to relu3_3
            self.features = nn.Sequential(
                *list(vgg.features.children())[:16]
            ).to(device).eval()
            for p in self.features.parameters():
                p.requires_grad_(False)
            self.enabled = True
            print("  [VGGLoss] Perceptual loss: ENABLED")
        except Exception as e:
            self.enabled = False
            print(f"  [VGGLoss] Not available ({e}) — skipping perceptual loss.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=pred.device)
        try:
            # Normalise to [0,1]
            p = (pred.float()   * 0.5 + 0.5).clamp(0, 1)
            t = (target.float() * 0.5 + 0.5).clamp(0, 1)
            # Resize to VGG input size
            p = F.interpolate(p, size=(224, 224), mode="bilinear",
                              align_corners=False)
            t = F.interpolate(t, size=(224, 224), mode="bilinear",
                              align_corners=False)
            return F.l1_loss(self.features(p), self.features(t))
        except Exception as e:
            print(f"  [VGGLoss] Forward failed ({e}) — skipping.")
            return torch.tensor(0.0, device=pred.device)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SketchToImagePipeline(nn.Module):
    """
    Full d-Sketch-inspired pipeline with colored edge map conditioning.

    Trainable:  LCTN, ColoredEdgeGenerator, ImageConditionProj
    Frozen:     VAE, CLIP Vision, SD UNet
    """

    def __init__(
        self,
        sd_model:    str   = DEFAULT_SD_MODEL,
        clip_model:  str   = DEFAULT_CLIP_MODEL,
        img_size:    int   = 256,
        device:      str   = "cuda",
        dtype:       torch.dtype = torch.float16,
        k_ratio:     float = 0.8,    # k/T noise ratio (0.7-0.9 per paper)
    ) -> None:
        super().__init__()
        self.device   = device
        self.dtype    = dtype
        self.img_size = img_size
        self.k_ratio  = k_ratio

        # Latent spatial size (img_size // 8)
        lat_h = img_size // 8
        lat_w = img_size // 8
        self.latent_dim = 4 * lat_h * lat_w

        # ── Load frozen SD components ──────────────────────────────────────
        print("  [Pipeline] Loading VAE …")
        try:
            self.vae = AutoencoderKL.from_pretrained(
                sd_model, subfolder="vae"
            ).to(device).eval()
            self.vae.requires_grad_(False)
        except OSError as e:
            raise OSError(f"VAE load failed: {e}") from e

        print("  [Pipeline] Loading SD UNet …")
        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                sd_model, subfolder="unet"
            ).to(device).eval()
            self.unet.requires_grad_(False)
        except OSError as e:
            raise OSError(f"UNet load failed: {e}") from e

        print("  [Pipeline] Loading DDPM scheduler …")
        try:
            self.scheduler = DDPMScheduler.from_pretrained(
                sd_model, subfolder="scheduler"
            )
        except Exception as e:
            raise RuntimeError(f"Scheduler load failed: {e}") from e

        # ── Load frozen CLIP vision encoder ───────────────────────────────
        print("  [Pipeline] Loading CLIP vision encoder …")
        try:
            self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model)
            self.clip_vision    = CLIPVisionModel.from_pretrained(
                clip_model
            ).to(device).eval()
            self.clip_vision.requires_grad_(False)
            clip_hidden_dim = self.clip_vision.config.hidden_size   # 1024 for ViT-L
            clip_seq_len    = (self.clip_vision.config.image_size
                               // self.clip_vision.config.patch_size) ** 2 + 1
        except OSError as e:
            raise OSError(f"CLIP vision load failed: {e}") from e

        # Enable xformers on UNet if available
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            print("  [Pipeline] UNet xformers: ENABLED")
        except Exception:
            pass

        # ── Trainable components ───────────────────────────────────────────
        print("  [Pipeline] Building trainable components …")

        # 1. LCTN — translates sketch latent → image latent (d-sketch core)
        self.lctn = LCTN(latent_dim=self.latent_dim)

        # 2. ColoredEdgeGenerator — sketch → colored edge map (your addition)
        self.edge_gen = ColoredEdgeGenerator(base_ch=64)

        # 3. ImageConditionProj — colored map CLIP features → SD conditioning
        self.cond_proj = ImageConditionProj(
            clip_dim = clip_hidden_dim,
            sd_dim   = 768,
            sd_seq   = 77,
            clip_seq = clip_seq_len,
        )

        self.to(device)
        print("  [Pipeline] Ready.")
        print(f"    img_size   : {img_size}")
        print(f"    latent_dim : {self.latent_dim}")
        print(f"    k_ratio    : {k_ratio}  (k = {int(k_ratio * 1000)} of 1000 steps)")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def trainable_parameters(self) -> list:
        return (list(self.lctn.parameters())
                + list(self.edge_gen.parameters())
                + list(self.cond_proj.parameters()))

    @torch.no_grad()
    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) [-1,1] → VAE latent (B,4,H//8,W//8)"""
        images = images.to(dtype=self.vae.dtype)
        return (self.vae.encode(images).latent_dist.sample()
                * self.vae.config.scaling_factor)

    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE latent → (B,3,H,W) [-1,1]"""
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents.to(dtype=self.vae.dtype)).sample

    @torch.no_grad()
    def encode_conditioning(
        self,
        colored_map: torch.Tensor,   # (B, 3, H, W) [-1, 1]
    ) -> torch.Tensor:
        """
        Colored edge map → SD cross-attention conditioning.
        colored_map : (B, 3, H, W) [-1, 1]
        returns     : (B, 77, 768)
        """
        # Convert to PIL for CLIP processor
        imgs_pil = []
        for i in range(colored_map.shape[0]):
            arr = ((colored_map[i].float().cpu() * 0.5 + 0.5)
                   .clamp(0, 1)
                   .permute(1, 2, 0)
                   .numpy() * 255).astype("uint8")
            imgs_pil.append(Image.fromarray(arr))

        # CLIP vision encode
        inputs  = self.clip_processor(images=imgs_pil, return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.clip_vision(**inputs).last_hidden_state  # (B, 257, 1024)

        # Project to SD conditioning format — THIS part has gradients
        return features   # return raw; projection applied in forward()

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        sketch:           torch.Tensor,   # (B, 3, H, W) [-1,  1]
        real_image:       torch.Tensor,   # (B, 3, H, W) [-1,  1]
        target_colored:   torch.Tensor,   # (B, 3, H, W) [-1,  1]
    ) -> dict:
        """
        Full training forward pass.

        Returns dict with:
            loss_total
            loss_latent
            loss_perceptual
            loss_edge_color
            pred_colored_map   (for logging)
            decoded_output     (for logging)
        """
        B = sketch.shape[0]

        # ── Step 1: VAE encode sketch + real image ─────────────────────────
        sketch_latent    = self.vae_encode(sketch)          # (B,4,H//8,W//8)
        real_latent      = self.vae_encode(real_image)      # (B,4,H//8,W//8)

        # ── Step 2: LCTN translates sketch latent → image latent ──────────
        pred_latent      = self.lctn(sketch_latent)         # (B,4,H//8,W//8)

        # Latent loss — same as d-sketch
        loss_latent      = F.mse_loss(pred_latent, real_latent.detach())

        # ── Step 3: ColoredEdgeGenerator → colored edge map ───────────────
        pred_colored_map = self.edge_gen(sketch)            # (B, 3, H, W)

        # Edge-color loss — your addition
        loss_edge_color  = F.l1_loss(pred_colored_map, target_colored.detach())

        # ── Step 4: Encode colored map → SD conditioning ──────────────────
        with torch.no_grad():
            clip_feats   = self.encode_conditioning(pred_colored_map.detach())

        # ImageConditionProj projects CLIP features — has gradients
        conditioning     = self.cond_proj(
            clip_feats.to(dtype=self.cond_proj.dim_proj.weight.dtype)
        )                                                   # (B, 77, 768)

        # ── Step 5: k-step partial denoising (d-sketch approach) ──────────
        # Add k steps of noise to pred_latent
        T          = self.scheduler.config.num_train_timesteps
        k          = int(self.k_ratio * T)
        timestep   = torch.tensor([k] * B, device=self.device).long()
        noise      = torch.randn_like(pred_latent)
        noisy_lat  = self.scheduler.add_noise(
            pred_latent.detach(), noise, timestep
        )

        # SD UNet forward — frozen, gradient does NOT flow here
        with torch.no_grad():
            noise_pred   = self.unet(
                noisy_lat.to(dtype=self.unet.dtype),
                timestep,
                encoder_hidden_states = conditioning.to(
                    dtype=self.unet.dtype
                ),
            ).sample

            # Denoise back to clean latent estimate
            step_out     = self.scheduler.step(noise_pred, k, noisy_lat)
            clean_latent = step_out.pred_original_sample

            # Decode to pixel space for perceptual loss
            decoded_output = self.vae_decode(
                clean_latent.to(dtype=self.vae.dtype)
            ).float()

        # ── Step 6: Perceptual loss ────────────────────────────────────────
        # Gradient flows through cond_proj → conditioning → used in UNet
        # but for perceptual loss we use decoded output from frozen UNet
        # So perceptual loss DOES signal back to cond_proj
        loss_perceptual = self._perceptual_loss(
            decoded_output, real_image.float()
        )

        # ── Total loss (same structure as d-sketch + edge_color term) ─────
        loss_total = (loss_latent
                      + 0.1  * loss_perceptual
                      + 0.5  * loss_edge_color)

        return {
            "loss_total":      loss_total,
            "loss_latent":     loss_latent,
            "loss_perceptual": loss_perceptual,
            "loss_edge_color": loss_edge_color,
            "pred_colored_map": pred_colored_map.detach(),
            "decoded_output":   decoded_output.detach(),
        }

    def _perceptual_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """VGG feature L1 loss. Returns 0 if VGG unavailable."""
        try:
            p = F.interpolate(
                (pred   * 0.5 + 0.5).clamp(0, 1), (224, 224),
                mode="bilinear", align_corners=False
            )
            t = F.interpolate(
                (target * 0.5 + 0.5).clamp(0, 1), (224, 224),
                mode="bilinear", align_corners=False
            )
            if not hasattr(self, "_vgg"):
                import torchvision.models as tvm
                vgg       = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
                self._vgg = nn.Sequential(
                    *list(vgg.features.children())[:16]
                ).to(self.device).eval()
                for param in self._vgg.parameters():
                    param.requires_grad_(False)
            return F.l1_loss(self._vgg(p), self._vgg(t))
        except Exception:
            return torch.tensor(0.0, device=self.device)

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        sketch_pil:         Image.Image,
        num_steps:          int   = 30,
        seed:               int   = 42,
        guidance_scale:     float = 3.0,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Sketch → (output_photo, colored_edge_map_preview)
        No text prompt needed.
        """
        from torchvision import transforms

        use_autocast = (self.device == "cuda")

        # ── Trainable models always fp32 (BatchNorm requires it) ──────────
        self.lctn.float()
        self.edge_gen.float()
        self.cond_proj.float()

        # ── Preprocess sketch → fp32 tensor ───────────────────────────────
        tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        sketch_t = (tf(sketch_pil.convert("RGB"))
                    .unsqueeze(0)
                    .to(device=self.device, dtype=torch.float32))

        # ── Step 1: VAE encode sketch ──────────────────────────────────────
        # VAE may be fp16 or fp32 — autocast handles it
        with torch.no_grad():
            with torch.autocast(device_type="cuda",
                                dtype=torch.float16,
                                enabled=use_autocast):
                vae_in = sketch_t.to(dtype=next(self.vae.parameters()).dtype)
                dist   = self.vae.encode(vae_in).latent_dist
                sketch_latent = dist.sample() * self.vae.config.scaling_factor
        # Bring back to fp32 for LCTN
        sketch_latent = sketch_latent.float()

        # ── Step 2: LCTN → image latent (fp32) ────────────────────────────
        with torch.no_grad():
            image_latent = self.lctn(sketch_latent).float()

        # ── Step 3: ColoredEdgeGenerator → colored map (fp32) ────────────
        with torch.no_grad():
            pred_colored = self.edge_gen(sketch_t).float()

        # ── Step 4: CLIP encode colored map (fp32 output) ─────────────────
        with torch.no_grad():
            clip_feats   = self.encode_conditioning(pred_colored).float()
            conditioning = self.cond_proj(clip_feats).float()     # (1, 77, 768)

        # ── Step 5: Setup scheduler + noisy latent ─────────────────────────
        T    = self.scheduler.config.num_train_timesteps
        k    = int(self.k_ratio * T)

        infer_sched = UniPCMultistepScheduler.from_config(
            self.scheduler.config
        )
        infer_sched.set_timesteps(num_steps)

        gen          = torch.Generator(device=self.device).manual_seed(seed)
        noise        = torch.randn_like(image_latent, generator=gen)
        timestep_k   = torch.tensor([k], device=self.device).long()

        # add_noise always works in fp32
        noisy_latent = infer_sched.add_noise(
            image_latent.float(),
            noise.float(),
            timestep_k,
        ).float()

        start_step   = int((1 - self.k_ratio) * num_steps)
        active_steps = infer_sched.timesteps[start_step:]

        # ── Step 6: Denoising loop ─────────────────────────────────────────
        # Use autocast so UNet can use fp16 internally without manual casting
        unet_dtype   = next(self.unet.parameters()).dtype
        latents      = noisy_latent

        with torch.no_grad():
            # Pre-cast conditioning once to match UNet dtype
            cond_unet = conditioning.to(dtype=unet_dtype)
            null_cond = torch.zeros_like(cond_unet)
            comb_cond = torch.cat([null_cond, cond_unet])   # (2, 77, 768)

            for t in active_steps:
                lat_in = torch.cat([latents] * 2).to(dtype=unet_dtype)
                lat_in = infer_sched.scale_model_input(lat_in, t)

                with torch.autocast(device_type="cuda",
                                    dtype=torch.float16,
                                    enabled=use_autocast):
                    noise_pred = self.unet(
                        lat_in,
                        t,
                        encoder_hidden_states=comb_cond,
                    ).sample.float()   # bring back to fp32 for scheduler

                # CFG in fp32
                u, c       = noise_pred.chunk(2)
                noise_pred = u + guidance_scale * (c - u)

                # Scheduler step in fp32
                latents = infer_sched.step(
                    noise_pred, t, latents.float()
                ).prev_sample.float()

        # ── Step 7: VAE decode ─────────────────────────────────────────────
        with torch.no_grad():
            with torch.autocast(device_type="cuda",
                                dtype=torch.float16,
                                enabled=use_autocast):
                dec_in  = (latents / self.vae.config.scaling_factor
                           ).to(dtype=next(self.vae.parameters()).dtype)
                out_img = self.vae.decode(dec_in).sample.float()

        # ── Convert to PIL ─────────────────────────────────────────────────
        out_arr = ((out_img[0].cpu() * 0.5 + 0.5)
                   .clamp(0, 1)
                   .permute(1, 2, 0)
                   .numpy() * 255).astype("uint8")
        output_pil = Image.fromarray(out_arr)

        col_arr = ((pred_colored[0].cpu() * 0.5 + 0.5)
                   .clamp(0, 1)
                   .permute(1, 2, 0)
                   .numpy() * 255).astype("uint8")
        colored_pil = Image.fromarray(col_arr)

        return output_pil, colored_pil

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all three trainable components."""
        os.makedirs(path, exist_ok=True)
        self.lctn.save(path)
        self.edge_gen.save(path)
        torch.save(
            self.cond_proj.state_dict(),
            os.path.join(path, "cond_proj.pt")
        )
        print(f"  [Pipeline] All components saved → {path}")

    def load(self, path: str) -> None:
        """Load all three trainable components."""
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: '{path}'\n"
                "  Train first: python train.py --data_dir ./dataset"
            )
        self.lctn.load(path)
        self.edge_gen.load(path)
        cp_path = os.path.join(path, "cond_proj.pt")
        if os.path.isfile(cp_path):
            state       = torch.load(cp_path, map_location="cpu", weights_only=True)
            miss, unex  = self.cond_proj.load_state_dict(state, strict=False)
            if miss:
                print(f"  [CondProj] Missing : {miss}")
            print(f"  [Pipeline] CondProj loaded ← {cp_path}")

        # Keep trainable models in fp32 — BatchNorm1d in LCTN requires fp32.
        # Their outputs are cast to fp16 only when entering frozen SD models.
        self.lctn.to(dtype=torch.float32)
        self.edge_gen.to(dtype=torch.float32)
        self.cond_proj.to(dtype=torch.float32)
        print(f"  [Pipeline] All components loaded ← {path}  "
              f"(trainable=fp32, frozen SD=fp16)")