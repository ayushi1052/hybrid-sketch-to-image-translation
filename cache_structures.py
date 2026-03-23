"""
cache_structures.py
====================
Run this ONCE before training to pre-compute all structure maps.
Saves a .pt file next to each sketch image.

Usage:
    python cache_structures.py --data_dir ./dataset
    python cache_structures.py --data_dir ./dataset --device cpu
    python cache_structures.py --data_dir ./dataset --overwrite   # re-cache all
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from models.structure_gan import StructureExtractor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=str, required=True)
    p.add_argument("--img_size",  type=int, default=256)
    p.add_argument("--device",    type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-compute even if .pt file already exists")
    return p.parse_args()


def main():
    args = parse_args()

    sketch_root = Path(args.data_dir) / "sketch"
    if not sketch_root.exists():
        print(f"[Error] sketch/ folder not found in: {args.data_dir}")
        sys.exit(1)

    # Collect all sketch images
    sketches = sorted([
        p for p in sketch_root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ])

    if not sketches:
        print(f"[Error] No images found in: {sketch_root}")
        sys.exit(1)

    # Count already cached
    already_cached = sum(1 for p in sketches if p.with_suffix(".pt").exists())
    to_process     = len(sketches) if args.overwrite else len(sketches) - already_cached

    print(f"\n  Total sketches : {len(sketches)}")
    print(f"  Already cached : {already_cached}")
    print(f"  To process     : {to_process}")

    if to_process == 0:
        print("\n  ✅ All structure maps already cached. Nothing to do.")
        print("     Use --overwrite to re-compute all.")
        return

    # Load extractor
    print(f"\n  Loading StructureExtractor on {args.device} ...")
    try:
        extractor = StructureExtractor(device=args.device)
    except Exception as e:
        print(f"[Error] Could not load StructureExtractor: {e}")
        sys.exit(1)

    # Process
    success = 0
    failed  = 0
    skipped = 0

    for sketch_path in tqdm(sketches, desc="Caching"):
        cache_path = sketch_path.with_suffix(".pt")

        # Skip if already cached and not overwriting
        if cache_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            img       = np.array(Image.open(sketch_path).convert("RGB"))
            structure = extractor.extract(img, (args.img_size, args.img_size))
            torch.save(structure, cache_path)
            success += 1

        except Exception as e:
            print(f"\n  [Failed] {sketch_path.name}: {e}")
            failed += 1

    print(f"\n  ✅ Done.")
    print(f"     Cached  : {success}")
    print(f"     Skipped : {skipped}  (already existed)")
    print(f"     Failed  : {failed}")
    if failed > 0:
        print(f"     Failed images will fall back to raw sketch during training.")
    print(f"\n  Training will now be 3-5x faster.")


if __name__ == "__main__":
    main()
