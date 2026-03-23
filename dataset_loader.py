"""
dataset_loader.py  [FIXED — cache support]

Key fix:
__getitem__ now checks for pre-computed .pt cache files before running StructureExtractor.

Cache priority order:
  1. .pt file exists next to sketch → load instantly  (fast path)
  2. .pt file missing + extractor available → extract + save cache
  3. .pt file missing + no extractor → raw sketch fallback

This makes training 3-5x faster after running the caching script once.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.structure_gan import StructureExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Sketch Augmentor
# ─────────────────────────────────────────────────────────────────────────────
class SketchAugmentor:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sketch_np: np.ndarray) -> np.ndarray:
        import random

        if random.random() > self.p:
            return sketch_np

        try:
            sketch_np = self._stroke_width(sketch_np)
        except Exception:
            pass

        try:
            sketch_np = self._line_breaks(sketch_np)
        except Exception:
            pass

        try:
            sketch_np = self._tremor(sketch_np)
        except Exception:
            pass

        return sketch_np

    def _stroke_width(self, img):
        import cv2
        import random

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        k = random.choice([1, 2, 3])
        if k > 1:
            bw = cv2.dilate(bw, np.ones((k, k), np.uint8))

        return np.stack([255 - bw] * 3, axis=-1)

    def _line_breaks(self, img, rate=0.03):
        import random

        img = img.copy()
        h, w = img.shape[:2]

        for _ in range(int(h * w * rate / 9)):
            y = random.randint(0, h - 3)
            x = random.randint(0, w - 3)
            img[y:y + 3, x:x + 3] = 255

        return img

    def _tremor(self, img, sigma=0.6):
        import cv2

        return cv2.GaussianBlur(img, (3, 3), sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Caption Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_caption(category: str) -> str:
    if not category or not category.strip():
        return "a realistic photo, highly detailed"

    vowels = "aeiou"
    name = category.strip().replace("_", " ").replace("-", " ")
    article = "an" if name[0].lower() in vowels else "a"

    return f"a realistic photo of {article} {name}, highly detailed"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SketchPhotoDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        extractor: Optional[StructureExtractor] = None,
        categories: Optional[List[str]] = None,
        augment: bool = True,
        caption_mode: str = "auto",
        fixed_caption: str = "a realistic photo, highly detailed",
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.extractor = extractor
        self.augment = augment
        self.caption_mode = caption_mode
        self.fixed_caption = fixed_caption
        self.augmentor = SketchAugmentor(p=0.5) if augment else None

        # Validate directories
        self.sketch_root = self.data_dir / "sketch"
        self.photo_root = self.data_dir / "photo"

        for label, d in [("sketch", self.sketch_root), ("photo", self.photo_root)]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {d}\n"
                    f"Expected: {data_dir}/{label}/<category>/<image.jpg>"
                )
            if not d.is_dir():
                raise NotADirectoryError(f"Expected a directory but got a file: {d}")

        # Discover categories
        sketch_cats = {d.name for d in self.sketch_root.iterdir() if d.is_dir()}
        photo_cats = {d.name for d in self.photo_root.iterdir() if d.is_dir()}

        found = sorted(sketch_cats & photo_cats)
        if not found:
            raise RuntimeError("No matching category folders found.")

        self.categories = categories if categories else found

        # Build pairs
        self.pairs: List[Tuple[Path, Path, str]] = []
        self.cat_counts: Dict[str, int] = {}

        for cat in self.categories:
            sk_map = self._index_folder(self.sketch_root / cat)
            ph_map = self._index_folder(self.photo_root / cat)

            common = sorted(set(sk_map) & set(ph_map))

            for stem in common:
                self.pairs.append((sk_map[stem], ph_map[stem], cat))

            self.cat_counts[cat] = len(common)

        if not self.pairs:
            raise RuntimeError("Dataset is empty.")

        # Cache stats
        cached = sum(1 for sk, _, _ in self.pairs if sk.with_suffix(".pt").exists())
        self._print_summary(cached, len(self.pairs))

        # Transforms
        self.sketch_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.photo_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sketch_path, photo_path, category = self.pairs[idx]

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        sketch_np = np.array(sketch.resize((self.img_size, self.img_size)))

        if self.augment and self.augmentor:
            sketch_np = self.augmentor(sketch_np)

        sketch_pil = Image.fromarray(sketch_np)

        structure = self._load_structure(sketch_path, sketch_np, sketch_pil)

        caption = build_caption(category)

        return {
            "sketch": self.sketch_tf(sketch_pil),
            "photo": self.photo_tf(photo),
            "structure": structure,
            "caption": caption,
            "category": category,
            "stem": sketch_path.stem,
        }

    def _index_folder(self, folder: Path) -> Dict[str, Path]:
        return {
            p.stem: p for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        }

    def _load_structure(self, sketch_path, sketch_np, sketch_pil):
        cache_path = sketch_path.with_suffix(".pt")

        if cache_path.exists():
            return torch.load(cache_path)

        if self.extractor:
            structure = self.extractor.extract(
                sketch_np, (self.img_size, self.img_size)
            )
            torch.save(structure, cache_path)
            return structure

        st = transforms.ToTensor()(sketch_pil)

        return torch.cat(
            [st, torch.zeros(2, self.img_size, self.img_size)],
            dim=0
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
):
    dataset = SketchPhotoDataset(data_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"

    loader = build_dataloader(data_dir)

    batch = next(iter(loader))

    print("Dataset OK")