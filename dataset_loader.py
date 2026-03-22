"""
dataset_loader.py
==================
Sketch-to-photo paired dataset — category-free, cache-accelerated.

Folder layouts (auto-detected)
───────────────────────────────
  FLAT          dataset/sketch/*.jpg   dataset/photo/*.jpg
  HIERARCHICAL  dataset/sketch/<sub>/*.jpg   dataset/photo/<sub>/*.jpg

Speed optimisations
────────────────────
• Structure maps are cached to disk as .npz files alongside the source
  images (written by StructureExtractor.extract).  After the first epoch,
  all GPU extraction is skipped — only a fast np.load() per sample.
• DataLoader uses persistent_workers + prefetch_factor=2, keeping worker
  processes alive across epochs and pre-fetching the next batch in parallel.
• Sketch augmentation uses pure cv2 (no PIL round-trip in the hot path).
• Caption is always a fixed generic string — no category string building.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.structure_gan import StructureExtractor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FIXED_CAPTION    = "a realistic photo, highly detailed, sharp focus"


# ─────────────────────────────────────────────────────────────────────────────
# Sketch augmentation  (cv2-only, avoids PIL round-trip overhead)
# ─────────────────────────────────────────────────────────────────────────────

class SketchAugmentor:
    """
    Three sketch-specific augmentations applied with probability p:
      1. Stroke-width variation via morphological dilation
      2. Line breaks (random white patches)
      3. Hand-tremor simulation (Gaussian blur)
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import random
        if random.random() > self.p:
            return img
        for fn in (self._stroke_width, self._line_breaks, self._tremor):
            try:
                img = fn(img)
            except Exception:
                pass
        return img

    def _stroke_width(self, img: np.ndarray) -> np.ndarray:
        import cv2, random
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        k     = random.choice([1, 2, 3])
        if k > 1:
            bw = cv2.dilate(bw, np.ones((k, k), np.uint8))
        return np.stack([255 - bw] * 3, axis=-1)

    def _line_breaks(self, img: np.ndarray, rate: float = 0.025) -> np.ndarray:
        import random
        out  = img.copy()
        h, w = out.shape[:2]
        for _ in range(int(h * w * rate / 9)):
            y = random.randint(0, h - 3)
            x = random.randint(0, w - 3)
            out[y:y+3, x:x+3] = 255
        return out

    def _tremor(self, img: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.GaussianBlur(img, (3, 3), 0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Pair discovery
# ─────────────────────────────────────────────────────────────────────────────

def _index_folder(folder: Path) -> Dict[str, Path]:
    try:
        return {
            p.stem: p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
    except PermissionError as e:
        raise PermissionError(f"Cannot read '{folder}': {e}") from e


def _pair_maps(
    sk_map: Dict[str, Path],
    ph_map: Dict[str, Path],
) -> List[Tuple[Path, Path]]:
    """Match by stem; fall back to sorted-order pairing."""
    common = sorted(set(sk_map) & set(ph_map))
    if common:
        return [(sk_map[s], ph_map[s]) for s in common]
    sk_l = sorted(sk_map.values())
    ph_l = sorted(ph_map.values())
    n    = min(len(sk_l), len(ph_l))
    if n < max(len(sk_l), len(ph_l)):
        print(f"  [Dataset] Stem mismatch — pairing {n} pairs by sorted order.")
    return list(zip(sk_l[:n], ph_l[:n]))


def _discover_pairs(
    sketch_root: Path,
    photo_root:  Path,
) -> Tuple[List[Tuple[Path, Path]], str]:
    """Auto-detect flat vs hierarchical layout."""
    sk_flat = _index_folder(sketch_root)
    ph_flat = _index_folder(photo_root)
    if sk_flat or ph_flat:
        return _pair_maps(sk_flat, ph_flat), "flat"

    sk_dirs = {d.name: d for d in sketch_root.iterdir() if d.is_dir()}
    ph_dirs = {d.name: d for d in photo_root.iterdir()  if d.is_dir()}
    shared  = sorted(set(sk_dirs) & set(ph_dirs))
    if not shared:
        raise RuntimeError(
            "No matching subdirectories found.\n"
            f"  sketch/ : {sorted(sk_dirs)}\n"
            f"  photo/  : {sorted(ph_dirs)}"
        )

    all_pairs: List[Tuple[Path, Path]] = []
    for name in shared:
        try:
            pairs = _pair_maps(
                _index_folder(sk_dirs[name]),
                _index_folder(ph_dirs[name]),
            )
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  [Dataset] Skipping '{name}': {e}")

    if not all_pairs:
        raise RuntimeError("No valid sketch/photo pairs found in any subfolder.")
    return all_pairs, f"hierarchical ({len(shared)} subfolders)"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SketchPhotoDataset(Dataset):
    """
    Paired sketch↔photo dataset.  No category filtering.
    Caption is always FIXED_CAPTION.
    """

    def __init__(
        self,
        data_dir:  str,
        img_size:  int                         = 256,
        extractor: Optional[StructureExtractor] = None,
        augment:   bool                        = True,
    ) -> None:
        self.img_size  = img_size
        self.extractor = extractor
        self.augment   = augment
        self.augmentor = SketchAugmentor(p=0.5) if augment else None

        data = Path(data_dir)
        sketch_root = data / "sketch"
        photo_root  = data / "photo"

        for label, d in [("sketch", sketch_root), ("photo", photo_root)]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Required directory missing: {d}\n"
                    f"  Expected: {data_dir}/{label}/ (flat or with subfolders)"
                )

        try:
            self.pairs, layout = _discover_pairs(sketch_root, photo_root)
        except RuntimeError as e:
            raise RuntimeError(f"Pair discovery failed: {e}") from e

        if not self.pairs:
            raise RuntimeError(f"Dataset is empty: {data_dir}")

        self._print_summary(layout)

        self.sketch_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
        ])
        self.photo_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _print_summary(self, layout: str) -> None:
        print(f"\n[SketchPhotoDataset]")
        print(f"  Layout     : {layout}")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  Image size : {self.img_size}×{self.img_size}")
        print(f"  Augment    : {self.augment}")
        print(f"  Caption    : '{FIXED_CAPTION}'\n")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sketch_path, photo_path = self.pairs[idx]

        # ── Load images ───────────────────────────────────────────────────
        try:
            sketch = Image.open(sketch_path).convert("RGB")
        except UnidentifiedImageError:
            raise IOError(f"Corrupt sketch: '{sketch_path}'")
        except Exception as e:
            raise IOError(f"Cannot load sketch '{sketch_path}': {e}") from e

        try:
            photo = Image.open(photo_path).convert("RGB")
        except UnidentifiedImageError:
            raise IOError(f"Corrupt photo: '{photo_path}'")
        except Exception as e:
            raise IOError(f"Cannot load photo '{photo_path}': {e}") from e

        # ── Resize sketch to numpy ────────────────────────────────────────
        try:
            sketch_np = np.array(
                sketch.resize((self.img_size, self.img_size), Image.BICUBIC)
            )
        except Exception as e:
            raise RuntimeError(f"Sketch resize failed: {e}") from e

        # ── Augmentation (flip + sketch-specific) ─────────────────────────
        if self.augment:
            try:
                if np.random.random() < 0.5:
                    sketch_np = sketch_np[:, ::-1, :].copy()
                    photo     = photo.transpose(Image.FLIP_LEFT_RIGHT)
                if self.augmentor is not None:
                    sketch_np = self.augmentor(sketch_np)
            except Exception:
                pass

        try:
            sketch_pil = Image.fromarray(sketch_np)
        except Exception:
            sketch_pil = sketch.resize((self.img_size, self.img_size))

        # ── Structure maps  (uses disk cache when available) ──────────────
        if self.extractor is not None:
            try:
                structure = self.extractor.extract(
                    sketch_np,
                    (self.img_size, self.img_size),
                    img_path=str(sketch_path),     # enables disk cache
                )
            except Exception:
                structure = self._fallback_structure(sketch_pil)
        else:
            structure = self._fallback_structure(sketch_pil)

        # ── Tensor conversion ─────────────────────────────────────────────
        try:
            sketch_t = self.sketch_tf(sketch_pil)
        except Exception as e:
            raise RuntimeError(f"Sketch→tensor failed at idx {idx}: {e}") from e
        try:
            photo_t  = self.photo_tf(photo)
        except Exception as e:
            raise RuntimeError(f"Photo→tensor failed at idx {idx}: {e}") from e

        return {
            "sketch":    sketch_t,          # (3, H, W) [0, 1]
            "photo":     photo_t,           # (3, H, W) [-1, 1]
            "structure": structure,         # (5, H, W) [0, 1]
            "caption":   FIXED_CAPTION,
            "stem":      sketch_path.stem,
        }

    def _fallback_structure(self, sketch_pil: Image.Image) -> torch.Tensor:
        """3-ch sketch + 2 zero depth/seg channels → (5, H, W)."""
        t = transforms.ToTensor()(sketch_pil)
        return torch.cat([t, torch.zeros(2, self.img_size, self.img_size)], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    data_dir:    str,
    img_size:    int                         = 256,
    batch_size:  int                         = 4,
    num_workers: int                         = 4,
    extractor:   Optional[StructureExtractor] = None,
    augment:     bool                        = True,
    shuffle:     bool                        = True,
) -> DataLoader:
    """Build a DataLoader for SGLDv2 training. No category filtering."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: '{data_dir}'")
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")
    if img_size < 16:
        raise ValueError(f"img_size must be ≥ 16, got {img_size}")

    dataset = SketchPhotoDataset(
        data_dir  = data_dir,
        img_size  = img_size,
        extractor = extractor,
        augment   = augment,
    )

    import multiprocessing
    max_w       = multiprocessing.cpu_count()
    num_workers = min(num_workers, max_w)

    return DataLoader(
        dataset,
        batch_size        = batch_size,
        shuffle           = shuffle,
        num_workers       = num_workers,
        pin_memory        = torch.cuda.is_available(),
        drop_last         = True,
        persistent_workers = (num_workers > 0),
        prefetch_factor    = 2 if num_workers > 0 else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, traceback
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    print(f"Testing dataset loader: {data_dir}")
    try:
        loader = build_dataloader(data_dir, batch_size=4, extractor=None, augment=False)
        batch  = next(iter(loader))
    except Exception as e:
        print(f"[Error] {e}")
        traceback.print_exc()
        sys.exit(1)
    print(f"sketch    : {batch['sketch'].shape}  "
          f"[{batch['sketch'].min():.2f}, {batch['sketch'].max():.2f}]")
    print(f"photo     : {batch['photo'].shape}   "
          f"[{batch['photo'].min():.2f}, {batch['photo'].max():.2f}]")
    print(f"structure : {batch['structure'].shape}")
    print(f"caption   : {batch['caption'][0]}")
    print("Dataset OK.")
