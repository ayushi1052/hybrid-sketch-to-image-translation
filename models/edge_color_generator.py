"""
models/edge_color_generator.py
================================
ColoredEdgeGenerator
=====================
Learns to generate a colored edge map from a hand-drawn sketch.

What is a colored edge map?
────────────────────────────
It is the BRIDGE between sketch domain and photo domain:
  - Edges   : extracted from the REAL paired photo (sharp, photographic quality)
  - Colors  : average color per region from the REAL paired photo

  Result: an image that has the sketch's structure
          but with real-world colors hinting what the photo should look like.

Training target (built from real image):
  Step 1: Canny edges from real image         → sharp photographic edge map
  Step 2: Heavy Gaussian blur of real image   → smooth color hint per region
  Step 3: Overlay edges (black) on color hint → colored edge map

Architecture: small UNet encoder-decoder with skip connections
  Encoder:  3 → 64 → 128 → 256
  Decoder:  256 → 128 → 64 → 3  (with skip connections)
  Output:   Tanh → [-1, 1]      (same normalisation as SD images)

This replaces the TEXT PROMPT from d-Sketch with a spatially-rich
colored conditioning image — giving SD color + structure without any text.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Target map builder  (used during dataset loading)
# ─────────────────────────────────────────────────────────────────────────────

def build_target_colored_edge_map(
    real_image_np: np.ndarray,
    blur_kernel:   int = 51,
    canny_lo:      int = 80,
    canny_hi:      int = 200,
) -> np.ndarray:
    """
    Build the colored edge map TARGET from a real photo.
    This is what ColoredEdgeGenerator learns to predict from sketch alone.

    real_image_np : (H, W, 3) uint8 RGB
    returns       : (H, W, 3) uint8 RGB — colored edge map
    """
    # Step 1 — Canny edges from real photo  (sharp, photographic quality)
    gray  = cv2.cvtColor(real_image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)           # (H, W) binary

    # Relax if too sparse (can happen on smooth subjects)
    if edges.sum() < 500:
        edges = cv2.Canny(gray, canny_lo // 2, canny_hi // 2)

    # Step 2 — Color hint: heavy blur gives per-region average color
    # Odd kernel required by GaussianBlur
    k           = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    color_hint  = cv2.GaussianBlur(real_image_np, (k, k), 0)

    # Step 3 — Draw edges (black) on color hint
    colored_map             = color_hint.copy()
    colored_map[edges > 0] = [0, 0, 0]     # black edges on top of colors

    return colored_map.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# ColoredEdgeGenerator — small UNet
# ─────────────────────────────────────────────────────────────────────────────

class ColoredEdgeGenerator(nn.Module):
    """
    Small UNet: sketch (3-ch) → colored edge map (3-ch).

    Input  : (B, 3, H, W)  float [-1, 1]   — normalised sketch
    Output : (B, 3, H, W)  float [-1, 1]   — predicted colored edge map
    """

    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()

        def _block(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, oc), oc),
                nn.SiLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, oc), oc),
                nn.SiLU(inplace=True),
            )

        c  = base_ch        # 64
        c2 = base_ch * 2    # 128
        c4 = base_ch * 4    # 256

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = _block(3,  c)    # (B, 64,  H,   W)
        self.enc2 = _block(c,  c2)   # (B, 128, H/2, W/2)
        self.enc3 = _block(c2, c4)   # (B, 256, H/4, W/4)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = _block(c4, c4)

        # ── Decoder with skip connections ─────────────────────────────────────
        self.dec3 = _block(c4 + c4, c2)   # skip from enc3
        self.dec2 = _block(c2 + c2, c)    # skip from enc2
        self.dec1 = _block(c  + c,  c)    # skip from enc1

        # ── Output projection ─────────────────────────────────────────────────
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, 3, kernel_size=1),
            nn.Tanh(),      # → [-1, 1]
        )

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        sketch : (B, 3, H, W) float [-1, 1]
        returns: (B, 3, H, W) float [-1, 1]  — colored edge map
        """
        if sketch.dim() != 4:
            raise ValueError(f"Expected 4-D input, got {sketch.shape}")

        # Encode
        e1 = self.enc1(sketch)              # (B, 64,  H,   W)
        e2 = self.enc2(self.pool(e1))       # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))       # (B, 256, H/4, W/4)

        # Bottleneck
        b  = self.bottleneck(self.pool(e3)) # (B, 256, H/8, W/8)

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up(b),  e3], dim=1))  # (B,256,H/4,W/4)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))  # (B,128,H/2,W/2)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))  # (B, 64, H,  W)

        return self.out_conv(d1)            # (B, 3, H, W) in [-1, 1]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        fpath = os.path.join(path, "edge_color_gen.pt")
        torch.save(self.state_dict(), fpath)
        print(f"  [ColoredEdgeGenerator] Saved → {fpath}")

    def load(self, path: str) -> None:
        fpath = os.path.join(path, "edge_color_gen.pt")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"ColoredEdgeGenerator checkpoint not found: '{fpath}'"
            )
        state       = torch.load(fpath, map_location="cpu", weights_only=True)
        miss, unex  = self.load_state_dict(state, strict=False)
        if miss:
            print(f"  [ColoredEdgeGenerator] Missing : {miss}")
        if unex:
            print(f"  [ColoredEdgeGenerator] Unexpected: {unex}")
        print(f"  [ColoredEdgeGenerator] Loaded ← {fpath}")