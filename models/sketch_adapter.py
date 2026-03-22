"""
models/sketch_adapter.py
=========================
StructureControlNet
===================
Architecture (matches diagram):
  Structure Maps (5-ch: edge + depth + SegFormer-RGB)
    → StructureInputProjection  (Conv64 → Conv128 → Conv256 → Conv3)
    → ControlNetModel           (cloned from frozen UNet encoder)
    → down/mid residuals        → injected into frozen UNet skip connections

Speed / memory optimisations
─────────────────────────────
• ControlNet runs with xformers memory-efficient attention when available
  (~20-30 % faster, ~40 % less VRAM for attention).
• ControlNet and InputProjection converted to channels_last memory layout.
• ControlNet gradient checkpointing halves activation memory during training.
• StructureInputProjection is compiled with torch.compile (inductor) when
  torch ≥ 2.0 is detected, giving ~15 % faster forward on CUDA.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from diffusers import ControlNetModel


# ─────────────────────────────────────────────────────────────────────────────
# Input Projection  Conv64 → Conv128 → Conv256 → Conv3
# ─────────────────────────────────────────────────────────────────────────────

class StructureInputProjection(nn.Module):
    """
    Projects 5-channel structure maps to a 3-channel control hint at the
    same spatial resolution (no downsampling).

    in  : (B, 5, H, W)  float [0, 1]
    out : (B, 3, H, W)  float [-1, 1]   — matches SD pixel normalisation
    """

    def __init__(self, in_channels: int = 5) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be ≥ 1, got {in_channels}")

        def _block(ic: int, oc: int, groups: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.GroupNorm(groups, oc),
                nn.SiLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.GroupNorm(groups, oc),
                nn.SiLU(inplace=True),
            )

        self.conv64  = _block(in_channels, 64,  8)
        self.conv128 = _block(64,  128, 8)
        self.conv256 = _block(128, 256, 8)
        self.out     = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,5,H,W) → (B,3,H,W)"""
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input, got {x.shape}")
        dtype = self.conv64[0].weight.dtype
        x     = x.to(dtype=dtype, memory_format=torch.channels_last)
        try:
            return self.out(self.conv256(self.conv128(self.conv64(x))))
        except RuntimeError as e:
            raise RuntimeError(
                f"StructureInputProjection failed: {e}  input={x.shape}"
            ) from e


# ─────────────────────────────────────────────────────────────────────────────
# StructureControlNet
# ─────────────────────────────────────────────────────────────────────────────

class StructureControlNet(nn.Module):
    """
    Full control encoder:
      structure_maps (5-ch)
        → StructureInputProjection  (Conv64 → Conv128 → Conv256 → Conv3)
        → ControlNetModel
        → (down_block_residuals, mid_block_residual)
        → injected into frozen UNet

    Only StructureInputProjection + ControlNet are trained.

    Usage
    -----
        down_res, mid_res = ctrl_net(noisy_lat, t, text_emb, struct_maps)
        noise_pred = unet(
            noisy_lat, t, text_emb,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample
    """

    def __init__(
        self,
        in_channels: int = 5,
        unet              = None,
        compile_proj: bool = True,
    ) -> None:
        super().__init__()

        if unet is None:
            raise ValueError("unet must be provided to build ControlNet.")

        # Input projection: 5-ch → 3-ch control hint
        self.input_proj = StructureInputProjection(in_channels=in_channels)

        # ── Build ControlNet from frozen UNet ─────────────────────────────
        print("  [StructureControlNet] Building ControlNet from UNet …")
        try:
            self.controlnet = ControlNetModel.from_unet(unet)
        except Exception as e:
            raise RuntimeError(
                f"ControlNetModel.from_unet failed: {e}\n"
                "  Ensure diffusers ≥ 0.27 is installed."
            ) from e

        # ── xformers memory-efficient attention ───────────────────────────
        try:
            self.controlnet.enable_xformers_memory_efficient_attention()
            print("  [StructureControlNet] xformers attention: ENABLED")
        except Exception:
            print("  [StructureControlNet] xformers not available — using default attention.")

        # ── channels_last ─────────────────────────────────────────────────
        try:
            self.controlnet = self.controlnet.to(memory_format=torch.channels_last)
            self.input_proj = self.input_proj.to(memory_format=torch.channels_last)
        except Exception:
            pass

        # ── Gradient checkpointing (halves activation memory during train) ─
        try:
            self.controlnet.enable_gradient_checkpointing()
            print("  [StructureControlNet] Gradient checkpointing: ENABLED")
        except Exception:
            pass

        # ── torch.compile InputProjection (inductor, ~15 % faster) ───────
        if compile_proj:
            try:
                self.input_proj = torch.compile(
                    self.input_proj, mode="reduce-overhead", fullgraph=False
                )
                print("  [StructureControlNet] torch.compile InputProjection: ENABLED")
            except Exception:
                pass   # compile is optional

        print("  [StructureControlNet] Ready.")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_latents:      torch.Tensor,   # (B, 4, H//8, W//8)
        timesteps:          torch.Tensor,   # (B,)
        encoder_hidden:     torch.Tensor,   # (B, 77, 768)
        structure_maps:     torch.Tensor,   # (B, 5, H, W)
        conditioning_scale: float = 1.0,
    ) -> tuple:
        """
        Returns (down_block_res_samples, mid_block_res_sample).
        """
        # Replace non-finite values silently
        if not torch.isfinite(structure_maps).all():
            structure_maps = torch.nan_to_num(
                structure_maps, nan=0.0, posinf=1.0, neginf=0.0
            )

        try:
            hint = self.input_proj(structure_maps)   # (B, 3, H, W)
        except Exception as e:
            raise RuntimeError(f"InputProjection failed: {e}") from e

        try:
            down_res, mid_res = self.controlnet(
                sample                = noisy_latents,
                timestep              = timesteps,
                encoder_hidden_states = encoder_hidden,
                controlnet_cond       = hint,
                conditioning_scale    = conditioning_scale,
                return_dict           = False,
            )
            return down_res, mid_res
        except torch.cuda.OutOfMemoryError:
            raise torch.cuda.OutOfMemoryError(
                "OOM in ControlNet. Try: lower --batch_size or --img_size"
            )
        except Exception as e:
            raise RuntimeError(
                f"ControlNet forward failed: {e}  "
                f"noisy={noisy_latents.shape}  hint={hint.shape}"
            ) from e

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Saves:
            {path}/input_proj.pt      — StructureInputProjection state_dict
            {path}/controlnet/        — ControlNet in HuggingFace format
        """
        os.makedirs(path, exist_ok=True)

        ip_path = os.path.join(path, "input_proj.pt")
        try:
            # Unwrap compiled module if necessary
            proj = getattr(self.input_proj, "_orig_mod", self.input_proj)
            torch.save(proj.state_dict(), ip_path)
        except Exception as e:
            raise RuntimeError(f"Could not save input_proj: {e}") from e

        cn_path = os.path.join(path, "controlnet")
        try:
            cn = getattr(self.controlnet, "_orig_mod", self.controlnet)
            cn.save_pretrained(cn_path)
        except Exception as e:
            raise RuntimeError(f"Could not save controlnet: {e}") from e

        print(f"  [StructureControlNet] Saved → {path}")

    def load(self, path: str) -> None:
        """
        Loads:
            {path}/input_proj.pt
            {path}/controlnet/
        """
        ip_path = os.path.join(path, "input_proj.pt")
        cn_path = os.path.join(path, "controlnet")

        if not os.path.isfile(ip_path):
            avail = os.listdir(path) if os.path.isdir(path) else []
            raise FileNotFoundError(
                f"'input_proj.pt' not found in '{path}'.  Available: {avail}"
            )
        if not os.path.isdir(cn_path):
            raise FileNotFoundError(
                f"'controlnet/' directory not found in '{path}'."
            )

        try:
            state = torch.load(ip_path, map_location="cpu", weights_only=True)
            proj  = getattr(self.input_proj, "_orig_mod", self.input_proj)
            miss, unex = proj.load_state_dict(state, strict=False)
            if miss:
                print(f"  [StructureControlNet] input_proj missing keys: {miss}")
            if unex:
                print(f"  [StructureControlNet] input_proj unexpected keys: {unex}")
        except Exception as e:
            raise RuntimeError(f"Could not load input_proj from '{ip_path}': {e}") from e

        try:
            loaded  = ControlNetModel.from_pretrained(cn_path)
            cn      = getattr(self.controlnet, "_orig_mod", self.controlnet)
            miss, _ = cn.load_state_dict(loaded.state_dict(), strict=False)
            if miss:
                print(f"  [StructureControlNet] controlnet missing keys: {miss}")
        except Exception as e:
            raise RuntimeError(f"Could not load controlnet from '{cn_path}': {e}") from e

        print(f"  [StructureControlNet] Loaded ← {path}")
