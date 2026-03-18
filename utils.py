"""
utils.py
=========
Shared utilities for SGLDv2 — with full exception handling.
All functions are safe to call and never silently corrupt data.
"""

import os
import json
import random
import traceback
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds. Warns but does not crash on failure."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except Exception as e:
        print(f"  [Warning] set_seed({seed}) partially failed: {e}")


def get_device() -> str:
    """Return best available device."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters. Returns 0 on failure."""
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as e:
        print(f"  [Warning] count_parameters failed: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Loss tracking
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Running average tracker. Thread-safe for single-thread use."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.avg   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        try:
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                return   # silently skip NaN/Inf values
            self.val    = float(val)
            self.sum   += float(val) * n
            self.count += n
            self.avg    = self.sum / max(self.count, 1)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Image conversion
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor, denorm: bool = False) -> Image.Image:
    """
    Convert (C, H, W) tensor → PIL Image.
    denorm=True applies ×0.5+0.5 to convert [-1,1] → [0,1].
    Raises ValueError on bad input shape.
    """
    if t is None:
        raise ValueError("tensor_to_pil: received None tensor.")
    if t.dim() not in (3, 4):
        raise ValueError(
            f"tensor_to_pil: expected 3D or 4D tensor, got shape {t.shape}"
        )
    if t.dim() == 4:
        if t.shape[0] != 1:
            raise ValueError(
                f"tensor_to_pil: 4D input must have batch size 1, got {t.shape[0]}"
            )
        t = t.squeeze(0)
    try:
        if denorm:
            t = t * 0.5 + 0.5
        return TF.to_pil_image(t.clamp(0, 1).cpu())
    except Exception as e:
        raise RuntimeError(f"tensor_to_pil failed: {e}") from e


def load_image_uint8(
    path: Union[str, Path],
    size: int = 256,
) -> torch.Tensor:
    """
    Load image → (3, H, W) uint8 tensor [0, 255].
    Used for FID computation.
    Raises FileNotFoundError or IOError with clear message.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: '{path}'")
    try:
        img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
        return TF.to_tensor(img).mul(255).byte()
    except Exception as e:
        raise IOError(f"Could not load image '{path}': {e}") from e


def load_image_float(
    path: Union[str, Path],
    size: int = 256,
) -> torch.Tensor:
    """
    Load image → (3, H, W) float tensor [0, 1].
    Used for SSIM/PSNR/LPIPS computation.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: '{path}'")
    try:
        img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
        return TF.to_tensor(img)
    except Exception as e:
        raise IOError(f"Could not load image '{path}': {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_grid(
    batch:    Dict[str, Any],
    out_path: str,
    max_imgs: int = 4,
) -> None:
    """
    Save 3-col grid: Sketch | Edge Map | Target Photo.
    Non-fatal — logs warning on any failure.
    """
    try:
        # Validate batch keys
        required = {"sketch", "structure", "photo"}
        missing  = required - set(batch.keys())
        if missing:
            print(f"  [Grid] Missing batch keys: {missing} — skipping sample grid.")
            return

        n = min(max_imgs, batch["sketch"].shape[0])
        if n == 0:
            print("  [Grid] Empty batch — skipping sample grid.")
            return

        rows = []
        for i in range(n):
            try:
                sketch = batch["sketch"][i].cpu().clamp(0, 1)
                # Edge channel (ch 0) of structure map → 3-channel grey
                edge   = batch["structure"][i, 0:1].cpu().expand(3, -1, -1).clamp(0, 1)
                photo  = (batch["photo"][i].cpu() * 0.5 + 0.5).clamp(0, 1)
                rows.extend([sketch, edge, photo])
            except Exception as e:
                print(f"  [Grid] Skipping sample {i}: {e}")
                continue

        if not rows:
            print("  [Grid] No valid samples to display.")
            return

        grid = make_grid(torch.stack(rows), nrow=3, padding=4, pad_value=1.0)
        img  = TF.to_pil_image(grid)
        img  = _add_labels(img, ["Sketch", "Edge Map", "Target Photo"])

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        img.save(out_path)

    except PermissionError as e:
        print(f"  [Grid] Permission denied saving to '{out_path}': {e}")
    except Exception as e:
        print(f"  [Grid] save_sample_grid failed: {e}")


def save_comparison(
    sketch:    Image.Image,
    structure: Image.Image,
    generated: Image.Image,
    out_path:  str,
) -> None:
    """
    Save 3-panel: Sketch | Structure Map | Generated.
    Non-fatal — logs warning on failure.
    """
    try:
        if any(img is None for img in [sketch, structure, generated]):
            print("  [Comparison] One or more images is None — skipping.")
            return

        w, h = generated.size
        if w == 0 or h == 0:
            print(f"  [Comparison] Generated image has zero size: {generated.size}")
            return

        sketch    = sketch.resize((w, h), Image.BICUBIC).convert("RGB")
        structure = structure.resize((w, h), Image.BICUBIC).convert("RGB")

        canvas = Image.new("RGB", (w * 3 + 8, h + 25), (255, 255, 255))
        canvas.paste(sketch,    (0,       25))
        canvas.paste(structure, (w + 4,   25))
        canvas.paste(generated, (w*2 + 8, 25))

        draw   = ImageDraw.Draw(canvas)
        labels = ["Sketch", "Structure Map", "Generated"]
        for i, lbl in enumerate(labels):
            draw.text(
                (w * i + w // 2 + i * 4, 12),
                lbl, fill=(40, 40, 40), anchor="mm"
            )

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        canvas.save(out_path)

    except PermissionError as e:
        print(f"  [Comparison] Permission denied saving to '{out_path}': {e}")
    except Exception as e:
        print(f"  [Comparison] save_comparison failed: {e}")


def _add_labels(
    grid_img: Image.Image,
    labels:   List[str],
    pad:      int = 22,
) -> Image.Image:
    """Add column header labels above a grid. Returns original on failure."""
    try:
        w, h = grid_img.size
        out  = Image.new("RGB", (w, h + pad), (255, 255, 255))
        out.paste(grid_img, (0, pad))
        draw = ImageDraw.Draw(out)
        cw   = w // max(len(labels), 1)
        for i, lbl in enumerate(labels):
            draw.text((cw * i + cw // 2, pad // 2), lbl,
                      fill=(30, 30, 30), anchor="mm")
        return out
    except Exception as e:
        print(f"  [Labels] _add_labels failed ({e}) — returning unlabelled grid.")
        return grid_img


# ─────────────────────────────────────────────────────────────────────────────
# Config persistence
# ─────────────────────────────────────────────────────────────────────────────

def log_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save training config to JSON.
    Warns but does not crash if saving fails.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = os.path.join(output_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"  Config saved → {cfg_path}")
    except PermissionError as e:
        print(f"  [Warning] Cannot write config (permission denied): {e}")
    except TypeError as e:
        print(f"  [Warning] Config contains non-serialisable values: {e}")
    except Exception as e:
        print(f"  [Warning] log_config failed: {e}")


def load_config(output_dir: str) -> Dict[str, Any]:
    """
    Load training config from JSON.
    Raises FileNotFoundError if config.json does not exist.
    """
    cfg_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Config file not found: '{cfg_path}'\n"
            f"  Make sure '{output_dir}' is a valid training output directory."
        )
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Config file is malformed JSON: '{cfg_path}'\n  {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Could not read config '{cfg_path}': {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def edge_map_np(img_np: np.ndarray) -> np.ndarray:
    """
    Canny edge map from HxWx3 uint8 → HxW float32 [0,1].
    Falls back to Sobel → zeros on failure.
    """
    try:
        import cv2
        gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return (edges / 255.0).astype(np.float32)
    except ImportError:
        print("  [edge_map_np] OpenCV not available.")
        return np.zeros(img_np.shape[:2], dtype=np.float32)
    except Exception as e:
        print(f"  [edge_map_np] Canny failed ({e}) — returning zeros.")
        return np.zeros(img_np.shape[:2], dtype=np.float32)