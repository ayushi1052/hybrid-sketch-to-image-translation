"""
dataset_loader.py
==================
Sketch-Photo paired dataset for the modified d-Sketch pipeline.

Each sample returns:
  sketch         : (3, H, W) float [-1, 1]  — hand-drawn sketch
  real_image     : (3, H, W) float [-1, 1]  — ground truth photo
  target_colored : (3, H, W) float [-1, 1]  — colored edge map built from real image
                   (this is what ColoredEdgeGenerator learns to predict)

Target colored edge map (built automatically from real image):
  Step 1: Canny edges from real photo    → sharp photographic edges
  Step 2: Heavy blur of real photo       → per-region color hint
  Step 3: Draw black edges on color hint → colored edge map

Folder layout (auto-detected):
  FLAT          dataset/sketch/*.jpg    dataset/photo/*.jpg
  HIERARCHICAL  dataset/sketch/<sub>/*.jpg   dataset/photo/<sub>/*.jpg
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.edge_color_generator import build_target_colored_edge_map

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Sketch augmentation
# ─────────────────────────────────────────────────────────────────────────────

class SketchAugmentor:
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
        import random
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        k     = random.choice([1, 2, 3])
        if k > 1:
            bw = cv2.dilate(bw, np.ones((k, k), np.uint8))
        return np.stack([255 - bw] * 3, axis=-1)

    def _line_breaks(self, img: np.ndarray) -> np.ndarray:
        import random
        out  = img.copy()
        h, w = out.shape[:2]
        for _ in range(int(h * w * 0.025 / 9)):
            y = random.randint(0, h - 3)
            x = random.randint(0, w - 3)
            out[y:y+3, x:x+3] = 255
        return out

    def _tremor(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (3, 3), 0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Pair discovery
# ─────────────────────────────────────────────────────────────────────────────

def _index_folder(folder: Path) -> Dict[str, Path]:
    try:
        return {p.stem: p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    except PermissionError as e:
        raise PermissionError(f"Cannot read '{folder}': {e}") from e


def _normalize_stem(stem: str) -> str:
    """
    Strip sketch variant suffixes to get the base photo ID.

    Your naming convention:
      photo  : abc..10105          → base ID = abc..10105
      sketch : abc..10105-1        → base ID = abc..10105
      sketch : abc..10105-2        → base ID = abc..10105
      sketch : abc..10105_1        → base ID = abc..10105

    Matches your notebook logic:
      stem.replace("-1", "").replace("_1", "")
    But generalised to handle -2, -3, _2, _3 etc. as well.
    """
    # Remove trailing -N or _N suffix (any single digit variant number)
   # remove extension (extra safety)
    name = stem.split('.')[0]

    # remove sketch suffix (-6, -3 etc.)
    if '-' in name:
        name = name.split('-')[0]

    return name


def _pair_maps(
    sk_map: Dict[str, Path],
    ph_map: Dict[str, Path],
) -> List[Tuple[Path, Path]]:
    """
    Match sketches to photos using normalized base IDs.

    sketch stem : abc..10105-1  →  normalize  →  abc..10105
    sketch stem : abc..10105-2  →  normalize  →  abc..10105
    photo  stem : abc..10105    →  normalize  →  abc..10105  (unchanged)

    All sketch variants correctly map to their one real photo.
    """
    # Build photo lookup: normalized_stem → photo_path
    # Photos have no suffix so normalize is a no-op for them
    photo_lookup: Dict[str, Path] = {}
    for stem, path in ph_map.items():
        norm = _normalize_stem(stem)
        photo_lookup[norm] = path

    pairs:     List[Tuple[Path, Path]] = []
    unmatched: List[str]               = []

    for sk_stem, sk_path in sk_map.items():
        base_id = _normalize_stem(sk_stem)
        if base_id in photo_lookup:
            pairs.append((sk_path, photo_lookup[base_id]))
        else:
            unmatched.append(sk_stem)

    if unmatched:
        preview = unmatched[:3]
        more    = f"... (+{len(unmatched)-3} more)" if len(unmatched) > 3 else ""
        print(f"  [Dataset] {len(unmatched)} sketches had no matching photo: "
              f"{preview}{more}")

    return pairs


def _discover_pairs(
    sketch_root: Path,
    photo_root:  Path,
) -> Tuple[List[Tuple[Path, Path]], str]:
    sk_flat = _index_folder(sketch_root)
    ph_flat = _index_folder(photo_root)
    if sk_flat or ph_flat:
        return _pair_maps(sk_flat, ph_flat), "flat"

    sk_dirs = {d.name: d for d in sketch_root.iterdir() if d.is_dir()}
    ph_dirs = {d.name: d for d in photo_root.iterdir()  if d.is_dir()}
    shared  = sorted(set(sk_dirs) & set(ph_dirs))
    if not shared:
        raise RuntimeError(
            f"No matching subdirectories.\n"
            f"  sketch/: {sorted(sk_dirs)}\n"
            f"  photo/ : {sorted(ph_dirs)}"
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
        raise RuntimeError("No valid sketch/photo pairs found.")
    return all_pairs, f"hierarchical ({len(shared)} subfolders)"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SketchPhotoDataset(Dataset):
    """
    Returns (sketch, real_image, target_colored_edge_map) triplets.
    target_colored_edge_map is built automatically from the real image.
    """

    def __init__(
        self,
        data_dir:  str,
        img_size:  int  = 256,
        augment:   bool = True,
    ) -> None:
        self.img_size  = img_size
        self.augment   = augment
        self.augmentor = SketchAugmentor(p=0.5) if augment else None

        data        = Path(data_dir)
        sketch_root = data / "sketch"
        photo_root  = data / "photo"

        for label, d in [("sketch", sketch_root), ("photo", photo_root)]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Required directory missing: {d}\n"
                    f"  Expected: {data_dir}/{label}/"
                )

        try:
            self.pairs, layout = _discover_pairs(sketch_root, photo_root)
        except RuntimeError as e:
            raise RuntimeError(f"Pair discovery failed: {e}") from e

        if not self.pairs:
            raise RuntimeError(f"Dataset is empty: {data_dir}")

        print(f"\n[SketchPhotoDataset]")
        print(f"  Layout     : {layout}")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  Image size : {img_size}×{img_size}")
        print(f"  Augment    : {augment}\n")

        # Transforms — normalize to [-1, 1] for SD compatibility
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

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

        # ── Resize to numpy ────────────────────────────────────────────────
        s = img_size = self.img_size
        sketch_np = np.array(sketch.resize((s, s), Image.BICUBIC))
        photo_np  = np.array(photo.resize((s, s),  Image.BICUBIC))

        # ── Augmentation — consistent flip on both sketch and photo ───────
        if self.augment:
            try:
                if np.random.random() < 0.5:
                    sketch_np = sketch_np[:, ::-1, :].copy()
                    photo_np  = photo_np[:,  ::-1, :].copy()
                if self.augmentor is not None:
                    sketch_np = self.augmentor(sketch_np)
            except Exception:
                pass

        # ── Build target colored edge map from real photo ──────────────────
        # This is what ColoredEdgeGenerator must learn to predict from sketch
        try:
            target_colored_np = build_target_colored_edge_map(photo_np)
        except Exception:
            target_colored_np = photo_np.copy()   # fallback: raw photo

        # ── Convert to tensors ────────────────────────────────────────────
        try:
            sketch_pil         = Image.fromarray(sketch_np)
            photo_pil          = Image.fromarray(photo_np)
            target_colored_pil = Image.fromarray(target_colored_np)
        except Exception as e:
            raise RuntimeError(f"Array→PIL failed at idx {idx}: {e}") from e

        try:
            sketch_t  = self.tf(sketch_pil)          # (3,H,W) [-1,1]
            photo_t   = self.tf(photo_pil)           # (3,H,W) [-1,1]
            colored_t = self.tf(target_colored_pil)  # (3,H,W) [-1,1]
        except Exception as e:
            raise RuntimeError(f"Transform failed at idx {idx}: {e}") from e

        return {
            "sketch":          sketch_t,    # (3,H,W) [-1,1]
            "real_image":      photo_t,     # (3,H,W) [-1,1]
            "target_colored":  colored_t,   # (3,H,W) [-1,1]
            "stem":            sketch_path.stem,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    data_dir:    str,
    img_size:    int  = 256,
    batch_size:  int  = 8,
    num_workers: int  = 4,
    augment:     bool = True,
    shuffle:     bool = True,
) -> DataLoader:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: '{data_dir}'")
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1")

    dataset = SketchPhotoDataset(
        data_dir = data_dir,
        img_size = img_size,
        augment  = augment,
    )

    import multiprocessing
    num_workers = min(num_workers, multiprocessing.cpu_count())

    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = shuffle,
        num_workers        = num_workers,
        pin_memory         = torch.cuda.is_available(),
        drop_last          = True,
        persistent_workers = (num_workers > 0),
        prefetch_factor    = 2 if num_workers > 0 else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    print(f"Testing: {data_dir}")
    try:
        loader = build_dataloader(data_dir, batch_size=4, augment=False)
        batch  = next(iter(loader))
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
    print(f"sketch         : {batch['sketch'].shape}  "
          f"[{batch['sketch'].min():.2f}, {batch['sketch'].max():.2f}]")
    print(f"real_image     : {batch['real_image'].shape}")
    print(f"target_colored : {batch['target_colored'].shape}")
    print(f"stem           : {batch['stem'][0]}")
    print("Dataset OK.")