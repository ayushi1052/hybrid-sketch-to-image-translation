"""
dataset_loader.py
==================
Dataset loader — with full exception handling

Handles your exact structure:
    dataset/
        sketch/  airplane/  apple/  ...
        photo/   airplane/  apple/  ...
"""

import os
import sys
import traceback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Optional, List, Tuple, Dict

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
        import cv2, random
        gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bw  = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        k      = random.choice([1, 2, 3])
        if k > 1:
            bw = cv2.dilate(bw, np.ones((k, k), np.uint8))
        return np.stack([255 - bw] * 3, axis=-1)

    def _line_breaks(self, img, rate=0.03):
        import random
        img  = img.copy()
        h, w = img.shape[:2]
        for _ in range(int(h * w * rate / 9)):
            y = random.randint(0, h - 3)
            x = random.randint(0, w - 3)
            img[y:y+3, x:x+3] = 255
        return img

    def _tremor(self, img, sigma=0.6):
        import cv2
        return cv2.GaussianBlur(img, (3, 3), sigma)


def build_caption(category: str) -> str:
    if not category or not category.strip():
        return "a realistic photo, highly detailed"
    vowels  = "aeiou"
    name    = category.strip().replace("_", " ").replace("-", " ")
    article = "an" if name[0].lower() in vowels else "a"
    return f"a realistic photo of {article} {name}, highly detailed"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SketchPhotoDataset(Dataset):
    def __init__(
        self,
        data_dir:      str,
        img_size:      int  = 256,
        extractor:     Optional[StructureExtractor] = None,
        categories:    Optional[List[str]] = None,
        augment:       bool = True,
        caption_mode:  str  = "auto",
        fixed_caption: str  = "a realistic photo, highly detailed",
    ):
        self.data_dir      = Path(data_dir)
        self.img_size      = img_size
        self.extractor     = extractor
        self.augment       = augment
        self.caption_mode  = caption_mode
        self.fixed_caption = fixed_caption
        self.augmentor     = SketchAugmentor(p=0.5) if augment else None

        # ── Validate root structure ────────────────────────────────────────
        self.sketch_root = self.data_dir / "sketch"
        self.photo_root  = self.data_dir / "photo"

        for label, d in [("sketch", self.sketch_root), ("photo", self.photo_root)]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {d}\n"
                    f"  Expected: {data_dir}/{label}/<category>/<image.jpg>"
                )
            if not d.is_dir():
                raise NotADirectoryError(f"Expected a directory but got a file: {d}")

        # ── Discover categories ────────────────────────────────────────────
        try:
            sketch_cats = {d.name for d in self.sketch_root.iterdir() if d.is_dir()}
            photo_cats  = {d.name for d in self.photo_root.iterdir()  if d.is_dir()}
        except PermissionError as e:
            raise PermissionError(f"Cannot read dataset directory: {e}") from e

        found = sorted(sketch_cats & photo_cats)
        if not found:
            raise RuntimeError(
                f"No matching category folders found.\n"
                f"  sketch/ has: {sorted(sketch_cats)}\n"
                f"  photo/  has: {sorted(photo_cats)}\n"
                f"  They must share at least one folder name."
            )

        if categories is not None:
            missing = set(categories) - set(found)
            if missing:
                print(f"  [Dataset] Warning: requested categories not found: {sorted(missing)}")
            self.categories = [c for c in categories if c in found]
            if not self.categories:
                raise RuntimeError(
                    f"None of the requested categories exist: {categories}\n"
                    f"  Available: {found}"
                )
        else:
            self.categories = found

        # ── Build pairs ────────────────────────────────────────────────────
        self.pairs: List[Tuple[Path, Path, str]] = []
        self.cat_counts: Dict[str, int] = {}
        skipped_cats = []

        for cat in self.categories:
            try:
                sk_map = self._index_folder(self.sketch_root / cat)
                ph_map = self._index_folder(self.photo_root  / cat)
            except Exception as e:
                print(f"  [Dataset] Warning: could not read category '{cat}': {e}")
                skipped_cats.append(cat)
                continue

            if not sk_map:
                print(f"  [Dataset] Warning: no images in sketch/{cat}/ — skipping.")
                skipped_cats.append(cat)
                continue
            if not ph_map:
                print(f"  [Dataset] Warning: no images in photo/{cat}/ — skipping.")
                skipped_cats.append(cat)
                continue

            common = sorted(set(sk_map) & set(ph_map))
            if common:
                for stem in common:
                    self.pairs.append((sk_map[stem], ph_map[stem], cat))
            else:
                # Fallback: pair by sorted order
                sk_list = sorted(sk_map.values())
                ph_list = sorted(ph_map.values())
                n = min(len(sk_list), len(ph_list))
                for i in range(n):
                    self.pairs.append((sk_list[i], ph_list[i], cat))

            self.cat_counts[cat] = len([p for p in self.pairs if p[2] == cat])

        if skipped_cats:
            print(f"  [Dataset] Skipped {len(skipped_cats)} categories: {skipped_cats}")

        if not self.pairs:
            raise RuntimeError(
                "Dataset is empty — no valid sketch/photo pairs found.\n"
                f"  Checked: {self.data_dir}"
            )

        self._print_summary()

        self.sketch_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
        ])
        self.photo_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    @staticmethod
    def _index_folder(folder: Path) -> Dict[str, Path]:
        try:
            return {
                p.stem: p for p in folder.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
            }
        except PermissionError as e:
            raise PermissionError(f"Cannot read folder '{folder}': {e}") from e

    def _print_summary(self):
        print(f"\n[SketchPhotoDataset] Summary:")
        print(f"  Total pairs : {len(self.pairs)}")
        print(f"  Categories  : {len(self.categories)}")
        print(f"  Augment     : {self.augment}")
        print(f"  {'Category':<20} {'Pairs':>6}")
        print(f"  {'-'*28}")
        for cat, count in self.cat_counts.items():
            print(f"  {cat:<20} {count:>6}")
        print()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sketch_path, photo_path, category = self.pairs[idx]

        # ── Load images with clear error messages ──────────────────────────
        try:
            sketch = Image.open(sketch_path).convert("RGB")
        except UnidentifiedImageError:
            raise IOError(
                f"Corrupt or unreadable sketch image: '{sketch_path}'\n"
                f"  Delete or replace this file and restart."
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Sketch file missing: '{sketch_path}'")
        except Exception as e:
            raise IOError(f"Error loading sketch '{sketch_path}': {e}") from e

        try:
            photo = Image.open(photo_path).convert("RGB")
        except UnidentifiedImageError:
            raise IOError(
                f"Corrupt or unreadable photo image: '{photo_path}'\n"
                f"  Delete or replace this file and restart."
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Photo file missing: '{photo_path}'")
        except Exception as e:
            raise IOError(f"Error loading photo '{photo_path}': {e}") from e

        # ── Resize sketch to numpy ─────────────────────────────────────────
        try:
            sketch_np = np.array(
                sketch.resize((self.img_size, self.img_size), Image.BICUBIC)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to resize sketch '{sketch_path}': {e}") from e

        # ── Augmentation ───────────────────────────────────────────────────
        if self.augment:
            try:
                if np.random.random() < 0.5:
                    sketch_np = sketch_np[:, ::-1, :].copy()
                    photo     = photo.transpose(Image.FLIP_LEFT_RIGHT)
                if self.augmentor:
                    sketch_np = self.augmentor(sketch_np)
            except Exception as e:
                # Augmentation failure is non-fatal
                pass

        try:
            sketch_pil = Image.fromarray(sketch_np)
        except Exception as e:
            sketch_pil = sketch.resize((self.img_size, self.img_size))

        # ── Structure maps ─────────────────────────────────────────────────
        if self.extractor is not None:
            try:
                structure = self.extractor.extract(
                    sketch_np, (self.img_size, self.img_size)
                )
            except Exception as e:
                # Structure extraction failure → fallback silently
                st = transforms.ToTensor()(sketch_pil)
                structure = torch.cat([
                    st, torch.zeros(2, self.img_size, self.img_size)
                ], dim=0)
        else:
            st = transforms.ToTensor()(sketch_pil)
            structure = torch.cat([
                st, torch.zeros(2, self.img_size, self.img_size)
            ], dim=0)

        # ── Caption ────────────────────────────────────────────────────────
        try:
            caption = build_caption(category) if self.caption_mode == "auto" \
                      else self.fixed_caption
        except Exception:
            caption = self.fixed_caption

        # ── Tensor conversion ──────────────────────────────────────────────
        try:
            sketch_tensor = self.sketch_tf(sketch_pil)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert sketch to tensor at index {idx}: {e}"
            ) from e

        try:
            photo_tensor  = self.photo_tf(photo)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert photo to tensor at index {idx}: {e}"
            ) from e

        return {
            "sketch":    sketch_tensor,
            "photo":     photo_tensor,
            "structure": structure,
            "caption":   caption,
            "category":  category,
            "stem":      sketch_path.stem,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    data_dir:      str,
    img_size:      int  = 256,
    batch_size:    int  = 4,
    num_workers:   int  = 2,
    extractor:     Optional[StructureExtractor] = None,
    categories:    Optional[List[str]] = None,
    augment:       bool = True,
    caption_mode:  str  = "auto",
    shuffle:       bool = True,
) -> DataLoader:
    """Build DataLoader with validation."""

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: '{data_dir}'")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if img_size < 16:
        raise ValueError(f"img_size must be >= 16, got {img_size}")

    # Build dataset
    dataset = SketchPhotoDataset(
        data_dir     = data_dir,
        img_size     = img_size,
        extractor    = extractor,
        categories   = categories,
        augment      = augment,
        caption_mode = caption_mode,
    )

    # Clamp num_workers to available CPUs
    import multiprocessing
    max_workers = multiprocessing.cpu_count()
    if num_workers > max_workers:
        print(f"  [DataLoader] Clamping num_workers from {num_workers} "
              f"to {max_workers} (system max).")
        num_workers = max_workers

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    print(f"Testing dataset: {data_dir}")

    try:
        loader = build_dataloader(
            data_dir, batch_size=4, extractor=None, augment=False
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[Error] {e}")
        sys.exit(1)

    try:
        batch = next(iter(loader))
    except StopIteration:
        print("[Error] Dataset is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to load first batch: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"sketch    : {batch['sketch'].shape}    "
          f"[{batch['sketch'].min():.2f}, {batch['sketch'].max():.2f}]")
    print(f"photo     : {batch['photo'].shape}     "
          f"[{batch['photo'].min():.2f}, {batch['photo'].max():.2f}]")
    print(f"structure : {batch['structure'].shape}")
    print(f"caption   : {batch['caption'][0]}")
    print(f"category  : {batch['category'][0]}")
    print("\nDataset OK.")