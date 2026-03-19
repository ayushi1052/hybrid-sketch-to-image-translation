"""
models/lctn.py
==============
LCTN — Latent Code Translation Network
=======================================
Directly from d-Sketch paper:
  "LCTN consists of a sequence of fully connected layers with
   512, 256, 128, and 64 nodes, with each FC layer followed by
   ReLU activation and batch normalization."

What it does:
  Sketch latent  (from VAE encoder)  →  Image latent  (same space as real photo)

This is the core of d-Sketch — once the sketch latent is translated into
the image latent space, SD can treat it like a real image latent and
denoise it into a photorealistic output.

Architecture:
  Flatten → 512 → 256 → 128 → 64 → 128 → 256 → 512 → Unflatten
  (encoder)     (bottleneck)       (decoder — symmetric)
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn


class LCTN(nn.Module):
    """
    Latent Code Translation Network.

    Translates sketch VAE latent → image VAE latent.
    Both have shape (B, 4, 64, 64) = 16384 values per sample.

    Args
    ----
    latent_dim : flattened latent size  (4 × H//8 × W//8)
                 default = 4 × 64 × 64 = 16384  for 512px images
                 default = 4 × 32 × 32 = 4096   for 256px images
    """

    def __init__(self, latent_dim: int = 16384) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        def _fc_block(in_f: int, out_f: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(inplace=True),
            )

        # Encoder  (same 4-layer structure as d-sketch paper)
        self.encoder = nn.Sequential(
            _fc_block(latent_dim, 512),
            _fc_block(512,        256),
            _fc_block(256,        128),
            _fc_block(128,         64),
        )

        # Decoder  (symmetric, reconstructs full latent from bottleneck)
        self.decoder = nn.Sequential(
            _fc_block(64,  128),
            _fc_block(128, 256),
            _fc_block(256, 512),
            nn.Linear(512, latent_dim),   # no BN/ReLU on final layer
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, sketch_latent: torch.Tensor) -> torch.Tensor:
        """
        sketch_latent : (B, 4, H//8, W//8)
        returns       : (B, 4, H//8, W//8)  — translated image latent
        """
        B = sketch_latent.shape[0]

        # Flatten latent to 1D vector per sample
        x = sketch_latent.reshape(B, -1)           # (B, latent_dim)

        # Translate sketch latent → image latent space
        x = self.decoder(self.encoder(x))          # (B, latent_dim)

        # Reshape back to spatial latent
        return x.reshape(sketch_latent.shape)      # (B, 4, H//8, W//8)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        fpath = os.path.join(path, "lctn.pt")
        torch.save(self.state_dict(), fpath)
        print(f"  [LCTN] Saved → {fpath}")

    def load(self, path: str) -> None:
        fpath = os.path.join(path, "lctn.pt")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"LCTN checkpoint not found: '{fpath}'")
        state = torch.load(fpath, map_location="cpu", weights_only=True)
        miss, unex = self.load_state_dict(state, strict=False)
        if miss:
            print(f"  [LCTN] Missing keys : {miss}")
        if unex:
            print(f"  [LCTN] Unexpected   : {unex}")
        print(f"  [LCTN] Loaded ← {fpath}")