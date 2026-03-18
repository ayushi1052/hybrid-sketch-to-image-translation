"""
utils.py
=========
Shared utility functions for SGLDv2.
"""

import os
import json
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Any, Dict, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Loss tracking
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.sum = self.avg = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Image conversion
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor, denorm: bool = False) -> Image.Image:
    if denorm:
        t = t * 0.5 + 0.5
    return TF.to_pil_image(t.clamp(0, 1).cpu())


def pil_to_tensor(
    img:  Image.Image,
    size: Optional[int] = None,
    norm: bool = False,
) -> torch.Tensor:
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    t = TF.to_tensor(img.convert("RGB"))
    if norm:
        t = t * 2.0 - 1.0
    return t


def load_image_uint8(path: Union[str, Path], size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return TF.to_tensor(img).mul(255).byte()


def load_image_float(path: Union[str, Path], size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return TF.to_tensor(img)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_grid(
    batch:    Dict[str, Any],
    out_path: str,
    max_imgs: int = 4,
) -> None:
    """
    Save 3-column grid: Sketch | Structure Map ch0 | Target Photo
    """
    n = min(max_imgs, batch["sketch"].shape[0])
    rows = []
    for i in range(n):
        sketch  = batch["sketch"][i].cpu().clamp(0, 1)
        # Show edge channel of structure map as 3-channel grey
        edge    = batch["structure"][i, 0:1].cpu().expand(3, -1, -1).clamp(0,1)
        photo   = (batch["photo"][i].cpu() * 0.5 + 0.5).clamp(0, 1)
        rows.extend([sketch, edge, photo])

    grid = make_grid(torch.stack(rows), nrow=3, padding=4, pad_value=1.0)
    img  = TF.to_pil_image(grid)
    img  = _add_labels(img, ["Sketch", "Edge Map", "Target Photo"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)


def save_comparison(
    sketch:    Image.Image,
    structure: Image.Image,
    generated: Image.Image,
    out_path:  str,
) -> None:
    """Save 3-panel: Sketch | Structure Map | Generated Photo"""
    size   = generated.size
    sketch = sketch.resize(size, Image.BICUBIC).convert("RGB")
    structure = structure.resize(size, Image.BICUBIC).convert("RGB")

    w, h = size
    canvas = Image.new("RGB", (w * 3 + 8, h + 25), (255, 255, 255))
    canvas.paste(sketch,    (0,         25))
    canvas.paste(structure, (w + 4,     25))
    canvas.paste(generated, (w * 2 + 8, 25))

    draw   = ImageDraw.Draw(canvas)
    labels = ["Sketch", "Structure Map", "Generated"]
    for i, lbl in enumerate(labels):
        draw.text((w * i + w // 2 + i * 4, 12), lbl,
                  fill=(40, 40, 40), anchor="mm")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def _add_labels(grid_img: Image.Image, labels: list, pad: int = 22) -> Image.Image:
    w, h = grid_img.size
    out  = Image.new("RGB", (w, h + pad), (255, 255, 255))
    out.paste(grid_img, (0, pad))
    draw = ImageDraw.Draw(out)
    cw   = w // len(labels)
    for i, lbl in enumerate(labels):
        draw.text((cw * i + cw // 2, pad // 2), lbl,
                  fill=(30, 30, 30), anchor="mm")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def log_config(config: Dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  Config saved → {output_dir}/config.json")


def load_config(output_dir: str) -> Dict[str, Any]:
    with open(os.path.join(output_dir, "config.json")) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def edge_map_np(img_np: np.ndarray) -> np.ndarray:
    import cv2
    gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return (edges / 255.0).astype(np.float32)
