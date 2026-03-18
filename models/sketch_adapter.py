"""
models/sketch_adapter.py
=========================
SketchAdapter — with full exception handling

Structure maps → Token sequence → injected into UNet cross-attention
via simple token concatenation (gradient-safe approach).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SketchTokenEncoder(nn.Module):
    """
    Encodes 5-channel structure maps → (B, num_tokens, 768) token sequence.
    All forward steps have dtype guards and shape validation.
    """

    def __init__(
        self,
        in_channels:  int = 5,
        token_dim:    int = 768,
        num_tokens:   int = 16,
        patch_size:   int = 16,
        img_size:     int = 256,
        num_layers:   int = 4,
        num_heads:    int = 8,
    ):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_tokens = num_tokens
        self.token_dim  = token_dim
        n_patches       = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 256, patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(256, token_dim, 1),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, token_dim) * 0.02
        )

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = token_dim,
            nhead           = num_heads,
            dim_feedforward = token_dim * 4,
            dropout         = 0.0,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Perceiver resampler: variable patches → fixed num_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        self.query_attn   = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads, batch_first=True
        )
        self.query_norm   = nn.LayerNorm(token_dim)
        self.out_proj     = nn.Linear(token_dim, token_dim)
        self.out_norm     = nn.LayerNorm(token_dim)

    def forward(self, structure_maps: torch.Tensor) -> torch.Tensor:
        """
        structure_maps : (B, 5, H, W)
        returns        : (B, num_tokens, 768)
        """
        # ── Input validation ──────────────────────────────────────────────
        if structure_maps.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W), got shape: {structure_maps.shape}"
            )
        if structure_maps.shape[1] != self.patch_embed[0].in_channels:
            raise ValueError(
                f"Expected {self.patch_embed[0].in_channels} channels, "
                f"got {structure_maps.shape[1]}"
            )

        B = structure_maps.shape[0]

        # ── Auto dtype cast: fixes fp16/fp32 mismatch ─────────────────────
        try:
            dtype = self.patch_embed[0].weight.dtype
            x     = structure_maps.to(dtype=dtype)
        except Exception as e:
            raise RuntimeError(f"Dtype cast failed: {e}") from e

        # ── Patch embedding ────────────────────────────────────────────────
        try:
            x = self.patch_embed(x)                          # (B, dim, h, w)
            x = x.flatten(2).permute(0, 2, 1)               # (B, n_patches, dim)
        except RuntimeError as e:
            raise RuntimeError(
                f"Patch embedding failed: {e}\n"
                f"  Input shape: {structure_maps.shape}\n"
                f"  Check that img_size matches the one used at init."
            ) from e

        # ── Positional encoding ────────────────────────────────────────────
        try:
            x = x + self.pos_embed.to(dtype=dtype)
        except RuntimeError as e:
            raise RuntimeError(
                f"Positional encoding failed: {e}\n"
                f"  x: {x.shape}, pos_embed: {self.pos_embed.shape}"
            ) from e

        # ── Transformer encoder ────────────────────────────────────────────
        try:
            x = self.transformer(x)                          # (B, n_patches, dim)
        except Exception as e:
            raise RuntimeError(f"Transformer encoder failed: {e}") from e

        # ── Perceiver resampler ────────────────────────────────────────────
        try:
            q      = self.query_tokens.expand(B, -1, -1).to(dtype=dtype)
            q      = self.query_norm(q)
            out, _ = self.query_attn(q, x, x)               # (B, num_tokens, dim)
        except Exception as e:
            raise RuntimeError(f"Perceiver resampler failed: {e}") from e

        # ── Output projection ──────────────────────────────────────────────
        try:
            return self.out_norm(self.out_proj(out))         # (B, num_tokens, 768)
        except Exception as e:
            raise RuntimeError(f"Output projection failed: {e}") from e


class SketchAdapter(nn.Module):
    """
    Full SketchAdapter: structure maps → sketch tokens.

    Usage:
        adapter       = SketchAdapter()
        sketch_tokens = adapter(structure_maps)          # (B, 16, 768)
        combined      = cat([text_tokens, sketch_tokens], dim=1)
        noise_pred    = unet(..., encoder_hidden_states=combined)

    Only SketchAdapter is trained. UNet stays frozen.
    """

    def __init__(
        self,
        in_channels:    int = 5,
        token_dim:      int = 768,
        num_tokens:     int = 16,
        num_enc_layers: int = 4,
        img_size:       int = 256,
    ):
        super().__init__()

        # Validate init params
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if img_size < 32:
            raise ValueError(f"img_size must be >= 32, got {img_size}")

        try:
            self.token_encoder = SketchTokenEncoder(
                in_channels = in_channels,
                token_dim   = token_dim,
                num_tokens  = num_tokens,
                img_size    = img_size,
                num_layers  = num_enc_layers,
            )
        except Exception as e:
            raise RuntimeError(f"SketchTokenEncoder init failed: {e}") from e

    def forward(self, structure_maps: torch.Tensor) -> torch.Tensor:
        """
        structure_maps : (B, 5, H, W)  float [0,1]
        returns        : (B, num_tokens, 768)
        """
        if structure_maps is None:
            raise ValueError("structure_maps is None.")

        if not torch.isfinite(structure_maps).all():
            # Replace NaN/Inf with zeros rather than crashing
            n_bad = (~torch.isfinite(structure_maps)).sum().item()
            print(f"  [SketchAdapter] Warning: {n_bad} non-finite values in "
                  "structure_maps — replacing with 0.")
            structure_maps = torch.nan_to_num(structure_maps, nan=0.0,
                                              posinf=1.0, neginf=0.0)

        try:
            return self.token_encoder(structure_maps)
        except RuntimeError as e:
            raise RuntimeError(
                f"SketchAdapter forward failed: {e}\n"
                f"  Input shape: {structure_maps.shape}"
            ) from e

    def save(self, path: str) -> None:
        """Save adapter weights — raises with clear message on failure."""
        import os
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create checkpoint directory '{path}': {e}"
            ) from e

        save_path = f"{path}/sketch_adapter.pt"
        try:
            torch.save(self.state_dict(), save_path)
            print(f"  [SketchAdapter] Saved → {save_path}")
        except OSError as e:
            raise OSError(
                f"Could not write checkpoint file '{save_path}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected save error: {e}") from e

    def load(self, path: str) -> None:
        """Load adapter weights — raises with clear message on failure."""
        import os
        load_path = f"{path}/sketch_adapter.pt"

        if not os.path.isfile(load_path):
            files = os.listdir(path) if os.path.isdir(path) else []
            raise FileNotFoundError(
                f"Checkpoint file not found: '{load_path}'\n"
                f"  Files in '{path}': {files}"
            )

        try:
            state = torch.load(load_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"Could not load checkpoint file '{load_path}': {e}\n"
                f"  The file may be corrupt. Try retraining."
            ) from e

        try:
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing:
                print(f"  [SketchAdapter] Warning: missing keys: {missing}")
            if unexpected:
                print(f"  [SketchAdapter] Warning: unexpected keys: {unexpected}")
            print(f"  [SketchAdapter] Loaded ← {load_path}")
        except RuntimeError as e:
            raise RuntimeError(
                f"State dict mismatch loading '{load_path}': {e}\n"
                f"  The checkpoint may have been saved with different --num_tokens "
                f"or --img_size. Check your training config."
            ) from e