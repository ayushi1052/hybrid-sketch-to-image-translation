"""
dataset_loader.py
==================
Sketch-Photo Dataset Loader for SGLD training.

Expected folder structure
─────────────────────────
data_dir/
    sketches/
        image_001.png
        image_002.png
        ...
    photos/
        image_001.png   ← same filename as sketch
        image_002.png
        ...

Each sketch is paired with a ground-truth photo sharing the same filename.
The loader returns:
    • sketch_tensor   : (3, H, W)  float [0, 1]
    • photo_tensor    : (3, H, W)  float [-1, 1]  (Stable Diffusion norm)
    • control_tensor  : (3, H, W)  float [0, 1]   (structure map for ControlNet)
    • caption         : str
    • stem            : filename without extension
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

from models.structure_gan import StructureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Supported image extensions
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# SketchPhotoDataset
# ─────────────────────────────────────────────────────────────────────────────

class SketchPhotoDataset(Dataset):
    """
    PyTorch Dataset for sketch → photo generation.

    Parameters
    ----------
    data_dir        : root directory containing 'sketches/' and 'photos/'
    img_size        : square resize target (default 512)
    extractor       : StructureExtractor instance; if None uses sketch as-is
    caption         : text prompt applied to all samples
    ctrl_mode       : 'edge' | 'depth' | 'combined'  (structure map type)
    augment         : apply random horizontal flip augmentation
    """

    def __init__(
        self,
        data_dir:    str,
        img_size:    int = 512,
        extractor:   Optional[StructureExtractor] = None,
        caption:     str = "a realistic photo of the object",
        ctrl_mode:   str = "combined",
        augment:     bool = True,
    ):
        self.data_dir  = Path(data_dir)
        self.img_size  = img_size
        self.extractor = extractor
        self.caption   = caption
        self.ctrl_mode = ctrl_mode
        self.augment   = augment

        # ── Collect matched pairs ─────────────────────────────────────────
        sketch_dir = self.data_dir / "sketches"
        photo_dir  = self.data_dir / "photos"

        self._validate_dirs(sketch_dir, photo_dir)

        # Build stem → path mapping for both sides
        sketch_map = self._index_folder(sketch_dir)
        photo_map  = self._index_folder(photo_dir)

        common_stems = sorted(set(sketch_map) & set(photo_map))
        if not common_stems:
            raise RuntimeError(
                f"No matching filenames found between:\n"
                f"  {sketch_dir}\n  {photo_dir}\n"
                "Make sure sketch and photo files share the same name."
            )

        self.pairs = [
            (sketch_map[s], photo_map[s]) for s in common_stems
        ]
        print(
            f"[SketchPhotoDataset] {len(self.pairs)} pairs "
            f"| size={img_size} | mode={ctrl_mode} | augment={augment}"
        )

        # ── Transforms ────────────────────────────────────────────────────
        # Sketch: [0, 1] float
        self.sketch_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
        ])

        # Photo: [-1, 1] float (Stable Diffusion VAE normalisation)
        self.photo_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_dirs(*dirs):
        for d in dirs:
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")

    @staticmethod
    def _index_folder(folder: Path) -> dict:
        """Returns {stem: path} for all images in a folder."""
        return {
            p.stem: p
            for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        }

    def _random_hflip(
        self,
        sketch_np: np.ndarray,
        photo: Image.Image,
        p: float = 0.5,
    ) -> Tuple[np.ndarray, Image.Image]:
        """Random horizontal flip applied consistently to both images."""
        if np.random.random() < p:
            sketch_np = sketch_np[:, ::-1, :].copy()
            photo = photo.transpose(Image.FLIP_LEFT_RIGHT)
        return sketch_np, photo

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sketch_path, photo_path = self.pairs[idx]

        # Load images
        sketch = Image.open(sketch_path).convert("RGB")
        photo  = Image.open(photo_path).convert("RGB")

        # Resize sketch to numpy for structure extraction
        sketch_resized = sketch.resize((self.img_size, self.img_size),
                                        Image.BICUBIC)
        sketch_np = np.array(sketch_resized)         # HxWx3 uint8

        # Optional augmentation (applied to both)
        if self.augment:
            sketch_np, photo = self._random_hflip(sketch_np, photo)
            sketch_resized = Image.fromarray(sketch_np)

        # ── Structure map (control image for ControlNet) ──────────────────
        if self.extractor is not None:
            ctrl_pil   = self.extractor.build_control_pil(
                sketch_np, size=self.img_size, mode=self.ctrl_mode
            )
            ctrl_tensor = transforms.ToTensor()(ctrl_pil)   # (3, H, W) [0,1]
        else:
            # Fallback: use greyscale sketch repeated as 3-channel
            gray = transforms.Grayscale(num_output_channels=3)(sketch_resized)
            ctrl_tensor = transforms.ToTensor()(gray)       # (3, H, W) [0,1]

        # ── Tensors ───────────────────────────────────────────────────────
        sketch_tensor = self.sketch_tf(sketch_resized)      # (3, H, W) [0,1]
        photo_tensor  = self.photo_tf(photo)                # (3, H, W) [-1,1]

        return {
            "sketch":  sketch_tensor,           # input (for logging/display)
            "photo":   photo_tensor,            # target (encoded to latent)
            "control": ctrl_tensor,             # conditioning image
            "caption": self.caption,
            "stem":    sketch_path.stem,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    data_dir:    str,
    img_size:    int  = 512,
    batch_size:  int  = 2,
    num_workers: int  = 4,
    extractor:   Optional[StructureExtractor] = None,
    caption:     str  = "a realistic photo",
    ctrl_mode:   str  = "combined",
    augment:     bool = True,
    shuffle:     bool = True,
) -> DataLoader:
    """
    Convenience function to build a DataLoader for SGLD training.

    Parameters
    ----------
    data_dir    : path to folder with 'sketches/' and 'photos/' subfolders
    img_size    : image resolution (must match model input, default 512)
    batch_size  : training batch size (2–4 for 12 GB VRAM)
    num_workers : DataLoader parallel workers
    extractor   : StructureExtractor (pass None to skip structure maps)
    caption     : text prompt for all training samples
    ctrl_mode   : 'edge' | 'depth' | 'combined'
    augment     : random horizontal flip
    shuffle     : shuffle training data

    Returns
    -------
    DataLoader
    """
    dataset = SketchPhotoDataset(
        data_dir  = data_dir,
        img_size  = img_size,
        extractor = extractor,
        caption   = caption,
        ctrl_mode = ctrl_mode,
        augment   = augment,
    )
    return DataLoader(
        dataset,
        batch_size   = batch_size,
        shuffle      = shuffle,
        num_workers  = num_workers,
        pin_memory   = True,
        drop_last    = True,          # keeps batch size consistent
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    print(f"Testing dataset at: {data_dir}")

    loader = build_dataloader(
        data_dir   = data_dir,
        img_size   = 512,
        batch_size = 2,
        extractor  = None,           # skip MiDaS for quick test
        augment    = False,
    )

    batch = next(iter(loader))

    print(f"\nBatch contents:")
    print(f"  sketch  : {batch['sketch'].shape}   range [{batch['sketch'].min():.2f}, {batch['sketch'].max():.2f}]")
    print(f"  photo   : {batch['photo'].shape}    range [{batch['photo'].min():.2f}, {batch['photo'].max():.2f}]")
    print(f"  control : {batch['control'].shape}  range [{batch['control'].min():.2f}, {batch['control'].max():.2f}]")
    print(f"  caption : {batch['caption']}")
    print(f"  stem    : {batch['stem']}")
    print("\n Dataset loader is working correctly.")"""Dataset loader module."""
