"""
dataset_loader.py
==================
Dataset loader for your exact folder structure:

    dataset/
        sketch/
            aeroplane/   img1.jpg  img2.jpg ...
            apple/       img1.jpg  img2.jpg ...
            ball/        img1.jpg  img2.jpg ...
            ...
        photo/
            aeroplane/   img1.jpg  img2.jpg ...
            apple/       img1.jpg  img2.jpg ...
            ball/        img1.jpg  img2.jpg ...
            ...

Pairing logic:
  sketch/aeroplane/img1.jpg  ↔  photo/aeroplane/img1.jpg  (same name, same category)

If filenames don't match within a category, pairs are made by sorted order.

Each item returned:
    sketch_tensor   : (3, 256, 256)  float [0, 1]
    photo_tensor    : (3, 256, 256)  float [-1, 1]   (SD normalisation)
    structure_tensor: (5, 256, 256)  float [0, 1]    (edge+depth+seg)
    caption         : "a realistic photo of a <category>"
    category        : str  e.g. "aeroplane"
    stem            : filename stem
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from models.structure_gan import StructureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Sketch Augmentor  (closes the edge-map → freehand-sketch domain gap)
# ─────────────────────────────────────────────────────────────────────────────

class SketchAugmentor:
    """
    Applies sketch-style augmentations during training.
    Simulates real freehand drawing characteristics so the model
    handles real user sketches better at inference.

    This is something D-Sketch does NOT do — it's trained only on
    clean Canny edge maps and struggles on rough freehand sketches.
    """

    def __init__(self, p: float = 0.5):
        """p: probability of each augmentation being applied."""
        self.p = p

    def __call__(self, sketch_np: np.ndarray) -> np.ndarray:
        import random
        if random.random() > self.p:
            return sketch_np
        sketch_np = self._random_stroke_width(sketch_np)
        sketch_np = self._random_line_breaks(sketch_np)
        sketch_np = self._add_hand_tremor(sketch_np)
        return sketch_np

    def _random_stroke_width(self, img: np.ndarray) -> np.ndarray:
        """Vary stroke thickness to simulate different pen/pencil sizes."""
        import cv2, random
        gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bw  = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        k      = random.choice([1, 2, 3])
        if k > 1:
            bw = cv2.dilate(bw, np.ones((k, k), np.uint8))
        result = 255 - bw
        return np.stack([result] * 3, axis=-1)

    def _random_line_breaks(
        self, img: np.ndarray, drop_rate: float = 0.03
    ) -> np.ndarray:
        """Randomly remove small patches (simulates incomplete strokes)."""
        import random
        img  = img.copy()
        h, w = img.shape[:2]
        n    = int(h * w * drop_rate / 9)
        for _ in range(n):
            y = random.randint(0, h - 3)
            x = random.randint(0, w - 3)
            img[y:y+3, x:x+3] = 255
        return img

    def _add_hand_tremor(
        self, img: np.ndarray, sigma: float = 0.6
    ) -> np.ndarray:
        """Slight Gaussian blur to simulate shaky hand."""
        import cv2
        return cv2.GaussianBlur(img, (3, 3), sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Caption builder
# ─────────────────────────────────────────────────────────────────────────────

def build_caption(category: str) -> str:
    """
    Auto-generate a descriptive text prompt from the category name.
    e.g. 'aeroplane' → 'a realistic photo of an aeroplane'
    """
    vowels     = "aeiou"
    article    = "an" if category[0].lower() in vowels else "a"
    clean_name = category.replace("_", " ").replace("-", " ")
    return f"a realistic photo of {article} {clean_name}, highly detailed"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SketchPhotoDataset(Dataset):
    """
    Sketch-Photo dataset supporting your exact category subfolder structure.

        dataset/
          sketch/category/img.jpg  ↔  photo/category/img.jpg

    Parameters
    ----------
    data_dir       : root directory (contains 'sketch/' and 'photo/')
    img_size       : resize target (your images are 256×256, set to 256)
    extractor      : StructureExtractor; if None uses raw sketch channels
    categories     : list of category names to use; None = all categories
    augment        : apply SketchAugmentor during training
    caption_mode   : 'auto' = generate from category name
                     'fixed' = use a single fixed caption
    fixed_caption  : used when caption_mode='fixed'
    """

    def __init__(
        self,
        data_dir:       str,
        img_size:       int  = 256,
        extractor:      Optional[StructureExtractor] = None,
        categories:     Optional[List[str]] = None,
        augment:        bool = True,
        caption_mode:   str  = "auto",
        fixed_caption:  str  = "a realistic photo, highly detailed",
    ):
        self.data_dir      = Path(data_dir)
        self.img_size      = img_size
        self.extractor     = extractor
        self.augment       = augment
        self.caption_mode  = caption_mode
        self.fixed_caption = fixed_caption

        self.augmentor = SketchAugmentor(p=0.5) if augment else None

        # ── Validate folder structure ─────────────────────────────────────
        self.sketch_root = self.data_dir / "sketch"
        self.photo_root  = self.data_dir / "photo"

        for d in [self.sketch_root, self.photo_root]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Expected directory not found: {d}\n"
                    f"Make sure your dataset has 'sketch/' and 'photo/' subfolders."
                )

        # ── Discover categories ────────────────────────────────────────────
        sketch_cats = {d.name for d in self.sketch_root.iterdir() if d.is_dir()}
        photo_cats  = {d.name for d in self.photo_root.iterdir()  if d.is_dir()}
        found_cats  = sorted(sketch_cats & photo_cats)

        if not found_cats:
            raise RuntimeError(
                "No matching category folders found between sketch/ and photo/."
            )

        if categories is not None:
            # Filter to requested categories
            missing = set(categories) - set(found_cats)
            if missing:
                print(f"  [Dataset] Warning: categories not found: {missing}")
            self.categories = [c for c in categories if c in found_cats]
        else:
            self.categories = found_cats

        # ── Build paired list ─────────────────────────────────────────────
        self.pairs: List[Tuple[Path, Path, str]] = []
        self.category_counts: Dict[str, int] = {}

        for cat in self.categories:
            sketch_dir = self.sketch_root / cat
            photo_dir  = self.photo_root  / cat

            sketch_map = self._index_folder(sketch_dir)
            photo_map  = self._index_folder(photo_dir)

            # Strategy 1: pair by same filename stem
            common = sorted(set(sketch_map) & set(photo_map))

            if common:
                for stem in common:
                    self.pairs.append((sketch_map[stem], photo_map[stem], cat))
            else:
                # Strategy 2: pair by sorted order (different filenames)
                sk_list = sorted(sketch_map.values())
                ph_list = sorted(photo_map.values())
                n = min(len(sk_list), len(ph_list))
                for i in range(n):
                    self.pairs.append((sk_list[i], ph_list[i], cat))

            cat_count = len([p for p in self.pairs if p[2] == cat])
            self.category_counts[cat] = cat_count

        self._print_summary()

        # ── Transforms ────────────────────────────────────────────────────
        self.sketch_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),                                # [0, 1]
        ])
        self.photo_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _index_folder(folder: Path) -> Dict[str, Path]:
        """Returns {stem: path} for all images in folder."""
        return {
            p.stem: p
            for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        }

    def _print_summary(self):
        print(f"\n[SketchPhotoDataset] Summary:")
        print(f"  Total pairs : {len(self.pairs)}")
        print(f"  Categories  : {len(self.categories)}")
        print(f"  Augment     : {self.augment}")
        print(f"  {'Category':<20} {'Pairs':>6}")
        print(f"  {'-'*28}")
        for cat, count in self.category_counts.items():
            print(f"  {cat:<20} {count:>6}")
        print()

    def _random_hflip(
        self, sk: np.ndarray, ph: Image.Image
    ) -> Tuple[np.ndarray, Image.Image]:
        import random
        if random.random() < 0.5:
            sk = sk[:, ::-1, :].copy()
            ph = ph.transpose(Image.FLIP_LEFT_RIGHT)
        return sk, ph

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sketch_path, photo_path, category = self.pairs[idx]

        # Load
        sketch = Image.open(sketch_path).convert("RGB")
        photo  = Image.open(photo_path).convert("RGB")

        # Resize sketch to numpy (for structure extraction + augmentation)
        sketch_np = np.array(
            sketch.resize((self.img_size, self.img_size), Image.BICUBIC)
        )

        # Augmentation (applied to both consistently)
        if self.augment:
            sketch_np, photo = self._random_hflip(sketch_np, photo)
            if self.augmentor:
                sketch_np = self.augmentor(sketch_np)

        sketch_pil = Image.fromarray(sketch_np)

        # ── Structure maps ────────────────────────────────────────────────
        if self.extractor is not None:
            structure = self.extractor.extract(
                sketch_np, target_size=(self.img_size, self.img_size)
            )                                          # (5, H, W) float [0,1]
        else:
            # Fallback: duplicate sketch channels (3ch) + two zero channels
            st = transforms.ToTensor()(sketch_pil)    # (3, H, W)
            structure = torch.cat([
                st,
                torch.zeros(2, self.img_size, self.img_size)
            ], dim=0)                                  # (5, H, W)

        # ── Caption ───────────────────────────────────────────────────────
        if self.caption_mode == "auto":
            caption = build_caption(category)
        else:
            caption = self.fixed_caption

        return {
            "sketch":    self.sketch_tf(sketch_pil),  # (3, H, W) [0, 1]
            "photo":     self.photo_tf(photo),         # (3, H, W) [-1, 1]
            "structure": structure,                    # (5, H, W) [0, 1]
            "caption":   caption,
            "category":  category,
            "stem":      sketch_path.stem,
        }

    def get_categories(self) -> List[str]:
        return list(self.categories)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    data_dir:       str,
    img_size:       int  = 256,
    batch_size:     int  = 4,
    num_workers:    int  = 4,
    extractor:      Optional[StructureExtractor] = None,
    categories:     Optional[List[str]] = None,
    augment:        bool = True,
    caption_mode:   str  = "auto",
    shuffle:        bool = True,
) -> DataLoader:
    """
    Build a DataLoader for your category-structured dataset.

    Parameters
    ----------
    data_dir     : path to dataset root (with sketch/ and photo/ subfolders)
    img_size     : 256 for your dataset
    batch_size   : 4 recommended for 12GB VRAM (SGLDv2 uses less memory)
    categories   : list of specific categories; None = all
    augment      : apply SketchAugmentor
    """
    dataset = SketchPhotoDataset(
        data_dir      = data_dir,
        img_size      = img_size,
        extractor     = extractor,
        categories    = categories,
        augment       = augment,
        caption_mode  = caption_mode,
    )
    return DataLoader(
        dataset,
        batch_size   = batch_size,
        shuffle      = shuffle,
        num_workers  = num_workers,
        pin_memory   = True,
        drop_last    = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    print(f"Testing dataset at: {data_dir}\n")

    loader = build_dataloader(
        data_dir    = data_dir,
        img_size    = 256,
        batch_size  = 4,
        extractor   = None,      # skip structure extraction for quick test
        augment     = False,
    )

    batch = next(iter(loader))

    print("Batch contents:")
    print(f"  sketch    : {batch['sketch'].shape}    "
          f"[{batch['sketch'].min():.2f}, {batch['sketch'].max():.2f}]")
    print(f"  photo     : {batch['photo'].shape}     "
          f"[{batch['photo'].min():.2f}, {batch['photo'].max():.2f}]")
    print(f"  structure : {batch['structure'].shape} "
          f"[{batch['structure'].min():.2f}, {batch['structure'].max():.2f}]")
    print(f"  caption   : {batch['caption'][0]}")
    print(f"  category  : {batch['category']}")
    print(f"  stem      : {batch['stem']}")
    print("\n Dataset loader working correctly.")
