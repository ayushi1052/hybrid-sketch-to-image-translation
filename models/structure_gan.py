"""
models/structure_gan.py
========================
StructureExtractor
==================
Extracts a 5-channel structure tensor from a sketch:
  ch 0   : Canny edge map
  ch 1   : MiDaS monocular depth  (fp16, channels_last)
  ch 2-4 : SegFormer-B0 semantic segmentation → ADE20K palette RGB

Speed / memory optimisations
─────────────────────────────
• MiDaS and SegFormer run fp16 on CUDA with channels_last memory layout.
• extract_batch() processes a list of images in one MiDaS + one SegFormer
  forward pass, amortising Python/GPU overhead across the whole batch.
• Disk cache: structure maps saved as .npz alongside source images;
  reloaded on next epoch, skipping all GPU work for seen images.
• Canny is CPU-only (OpenCV) — zero GPU memory.

Fallback chain
──────────────
  Depth  : MiDaS → gradient pseudo-depth
  Seg    : SegFormer → SLIC → K-means → raw colour channels
  Edge   : Canny → Sobel → grayscale
"""

from __future__ import annotations

import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# ADE20K colour palette (150 classes)
# ─────────────────────────────────────────────────────────────────────────────
_ADE_PALETTE = np.array([
    [120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],
    [120,120,80],[140,140,140],[204,5,255],[230,230,230],[4,250,7],
    [224,5,255],[235,255,7],[150,5,61],[120,120,70],[8,255,51],
    [255,6,82],[143,255,140],[204,255,4],[255,51,7],[204,70,3],
    [0,102,200],[61,230,250],[255,6,51],[11,102,255],[255,7,71],
    [255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],
    [8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],
    [7,255,255],[224,255,8],[102,8,255],[255,61,6],[255,194,7],
    [255,122,8],[0,255,20],[255,8,41],[255,5,153],[6,51,255],
    [235,12,255],[160,150,20],[0,163,255],[140,140,140],[250,10,15],
    [20,255,0],[31,255,0],[255,31,0],[255,224,0],[153,255,0],
    [0,0,255],[255,71,0],[0,235,255],[0,173,255],[31,0,255],
    [11,200,200],[255,82,0],[0,255,245],[0,61,255],[0,255,112],
    [0,255,133],[255,0,0],[255,163,0],[255,102,0],[194,255,0],
    [0,143,255],[51,255,0],[0,82,255],[0,255,41],[0,255,173],
    [10,0,255],[173,255,0],[0,255,153],[255,92,0],[255,0,255],
    [255,0,245],[255,0,102],[255,173,0],[255,0,20],[255,184,184],
    [0,31,255],[0,255,61],[0,71,255],[255,0,204],[0,255,194],
    [0,255,82],[0,10,255],[0,112,255],[51,0,255],[0,194,255],
    [0,122,255],[0,255,163],[255,153,0],[0,255,10],[255,112,0],
    [143,255,0],[82,0,255],[163,255,0],[255,235,0],[8,184,170],
    [133,0,255],[0,255,92],[184,0,255],[255,0,31],[0,184,255],
    [0,214,255],[255,0,112],[92,255,0],[0,224,255],[112,224,255],
    [70,184,160],[163,0,255],[153,0,255],[71,255,0],[255,0,163],
    [255,204,0],[255,0,143],[0,255,235],[133,255,0],[255,0,235],
    [245,0,255],[255,0,122],[255,245,0],[10,190,212],[214,255,0],
    [0,204,255],[20,0,255],[255,255,0],[0,153,255],[0,41,255],
    [0,255,204],[41,0,255],[41,255,0],[173,0,255],[0,245,255],
    [71,0,255],[122,0,255],[0,255,184],[0,92,255],[184,255,0],
    [0,133,255],[255,214,0],[25,194,194],[102,255,0],[92,0,255],
], dtype=np.uint8)   # (150, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Disk-cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(img_path: str, h: int) -> str:
    p = Path(img_path)
    return str(p.parent / f"{p.stem}_struct{h}.npz")

def _load_cache(path: str) -> Optional[np.ndarray]:
    try:
        if os.path.isfile(path):
            arr = np.load(path)["s"]
            return arr if arr.shape[0] == 5 else None
    except Exception:
        return None

def _save_cache(path: str, arr: np.ndarray) -> None:
    try:
        np.savez_compressed(path, s=arr)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# StructureExtractor
# ─────────────────────────────────────────────────────────────────────────────

class StructureExtractor:
    """
    Extracts 5-channel structure maps from sketches or photos.

    Args
    ----
    device    : "cuda" | "mps" | "cpu"
    use_cache : persist .npz maps next to source images
    half      : run neural models in fp16 (CUDA only)
    """

    def __init__(
        self,
        device:    str  = "cuda",
        use_cache: bool = True,
        half:      bool = True,
    ) -> None:
        self.device    = device
        self.use_cache = use_cache
        self.half      = half and (device == "cuda")
        self._dtype    = torch.float16 if self.half else torch.float32

        self._depth_backend = None
        self._seg_backend   = None

        self._load_midas()
        self._load_segformer()
        print("  [StructureExtractor] Ready.")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_midas(self) -> None:
        try:
            print("  [StructureExtractor] Loading MiDaS-small …")
            midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            ).to(self.device).eval()
            if self.half:
                midas = midas.half()
            self._midas    = midas.to(memory_format=torch.channels_last)
            tf             = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._midas_tf = tf.small_transform
            self._depth_backend = "midas"
            suffix = " (fp16)" if self.half else ""
            print(f"  [StructureExtractor] MiDaS: OK{suffix}")
        except Exception as e:
            print(f"  [StructureExtractor] MiDaS failed ({e}) → gradient fallback.")
            self._depth_backend = "gradient"

    def _load_segformer(self) -> None:
        try:
            from transformers import (
                SegformerImageProcessor,
                SegformerForSemanticSegmentation,
            )
            mid = "nvidia/segformer-b0-finetuned-ade-512-512"
            print("  [StructureExtractor] Loading SegFormer-B0 …")
            self._seg_proc  = SegformerImageProcessor.from_pretrained(mid)
            seg = SegformerForSemanticSegmentation.from_pretrained(mid).to(self.device).eval()
            if self.half:
                seg = seg.half()
            self._seg_model   = seg
            self._seg_backend = "segformer"
            suffix = " (fp16)" if self.half else ""
            print(f"  [StructureExtractor] SegFormer-B0: OK{suffix}")
        except ImportError:
            print("  [StructureExtractor] transformers not found → SLIC fallback.")
            self._seg_backend = "slic"
        except Exception as e:
            print(f"  [StructureExtractor] SegFormer failed ({e}) → SLIC fallback.")
            self._seg_backend = "slic"

    # ── Edge ─────────────────────────────────────────────────────────────────

    def get_edge_map(self, img_np: np.ndarray) -> np.ndarray:
        """Canny edge map with sketch-aware thresholds. (H,W) float32 [0,1]."""
        try:
            gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            median = float(np.median(gray))
            if median > 200:                       # white-background sketch
                lo, hi = 30, 100
            elif median < 30:
                lo, hi = 10, 50
            else:
                lo = int(max(10,  0.33 * median))
                hi = int(min(250, 1.00 * median))
                if hi - lo < 20:
                    lo = max(10, hi - 40)
            edges = cv2.Canny(gray, lo, hi)
            if edges.sum() == 0:
                edges = cv2.Canny(gray, max(5, lo // 3), max(30, hi // 2))
            return (edges / 255.0).astype(np.float32)
        except Exception:
            pass
        # Sobel fallback
        try:
            g  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
            sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            m  = np.hypot(sx, sy)
            return ((m - m.min()) / (m.max() - m.min() + 1e-8)).astype(np.float32)
        except Exception:
            return np.zeros(img_np.shape[:2], dtype=np.float32)

    # ── Depth ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_depth_map(self, img_np: np.ndarray) -> np.ndarray:
        """MiDaS monocular depth. (H,W) float32 [0,1]."""
        if self._depth_backend == "midas":
            try:
                inp  = self._midas_tf(img_np).to(self.device)
                if self.half:
                    inp = inp.half()
                inp  = inp.to(memory_format=torch.channels_last)
                pred = self._midas(inp)
                pred = F.interpolate(
                    pred.unsqueeze(1), size=img_np.shape[:2],
                    mode="bicubic", align_corners=False,
                ).squeeze().float().cpu().numpy()
                return ((pred - pred.min()) / (pred.max() - pred.min() + 1e-8)).astype(np.float32)
            except torch.cuda.OutOfMemoryError:
                print("  [Depth] MiDaS OOM → gradient fallback.")
                self._depth_backend = "gradient"
            except Exception as e:
                print(f"  [Depth] MiDaS failed ({e}) → gradient fallback.")
                self._depth_backend = "gradient"
        # Gradient pseudo-depth
        h, w = img_np.shape[:2]
        return np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w))

    # ── Segmentation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_seg_map(self, img_np: np.ndarray) -> np.ndarray:
        """SegFormer semantic segmentation → ADE20K palette RGB. (H,W,3) float32 [0,1]."""
        h, w = img_np.shape[:2]

        if self._seg_backend == "segformer":
            try:
                pil    = Image.fromarray(img_np)
                inputs = self._seg_proc(images=pil, return_tensors="pt")
                inputs = {k: v.to(self.device,
                                  dtype=self._dtype if v.is_floating_point() else v.dtype)
                          for k, v in inputs.items()}
                logits = self._seg_model(**inputs).logits
                logits = F.interpolate(logits.float(), size=(h, w),
                                       mode="bilinear", align_corners=False)
                labels = logits.argmax(1).squeeze().cpu().numpy()
                labels = np.clip(labels, 0, len(_ADE_PALETTE) - 1)
                return (_ADE_PALETTE[labels].astype(np.float32) / 255.0)
            except torch.cuda.OutOfMemoryError:
                print("  [Seg] SegFormer OOM → SLIC fallback.")
                self._seg_backend = "slic"
            except Exception as e:
                print(f"  [Seg] SegFormer failed ({e}) → SLIC fallback.")
                self._seg_backend = "slic"

        if self._seg_backend == "slic":
            try:
                from skimage.segmentation import slic
                from skimage.color import label2rgb
                segs    = slic(img_np, n_segments=50, compactness=10,
                               sigma=1, start_label=1)
                seg_img = label2rgb(segs, img_np, kind="avg", bg_label=0)
                return (seg_img / 255.0).astype(np.float32)
            except Exception:
                self._seg_backend = "kmeans"

        if self._seg_backend == "kmeans":
            try:
                return self._kmeans_seg(img_np)
            except Exception:
                self._seg_backend = "colour"

        return (img_np / 255.0).astype(np.float32)

    def _kmeans_seg(self, img: np.ndarray, k: int = 8) -> np.ndarray:
        h, w   = img.shape[:2]
        px     = img.reshape(-1, 3).astype(np.float32)
        crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, lbl, ctr = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        return (ctr[lbl.flatten()].reshape(h, w, 3) / 255.0).astype(np.float32)

    # ── Single extraction ─────────────────────────────────────────────────────

    def extract(
        self,
        img_np:      np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
        img_path:    Optional[str]   = None,
    ) -> torch.Tensor:
        """
        Returns (5, H, W) float32 tensor.
        img_path enables disk caching.
        """
        h, w = target_size

        if self.use_cache and img_path:
            cf  = _cache_path(img_path, h)
            arr = _load_cache(cf)
            if arr is not None:
                return torch.from_numpy(arr)

        if img_np is None:
            raise ValueError("img_np is None.")
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) uint8, got {img_np.shape}")
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_AREA)

        edge  = self._safe_edge(img_np, h, w)
        depth = self._safe_depth(img_np, h, w)
        seg   = self._safe_seg(img_np, h, w)

        struct = np.concatenate([edge[None], depth[None],
                                 seg.transpose(2, 0, 1)],
                                axis=0).astype(np.float32)

        if self.use_cache and img_path:
            _save_cache(_cache_path(img_path, h), struct)

        return torch.from_numpy(struct)

    # ── Batch extraction ──────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_batch(
        self,
        images:      List[np.ndarray],
        target_size: Tuple[int, int]      = (256, 256),
        img_paths:   Optional[List[str]]  = None,
    ) -> List[torch.Tensor]:
        """
        Batch version — single MiDaS + single SegFormer forward pass.
        Returns list of (5, H, W) float32 tensors.
        """
        h, w = target_size
        n    = len(images)
        results  : List[Optional[torch.Tensor]] = [None] * n
        todo_idx : List[int] = []

        # Load cached
        if self.use_cache and img_paths:
            for i, p in enumerate(img_paths):
                if p:
                    arr = _load_cache(_cache_path(p, h))
                    if arr is not None:
                        results[i] = torch.from_numpy(arr)
                        continue
                todo_idx.append(i)
        else:
            todo_idx = list(range(n))

        if not todo_idx:
            return results

        # Resize
        resized = []
        for i in todo_idx:
            img = images[i]
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            resized.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))

        edges  = [self._safe_edge(im, h, w)  for im in resized]
        depths = self._batch_midas(resized, h, w)
        segs   = self._batch_segformer(resized, h, w)

        for k, i in enumerate(todo_idx):
            struct = np.concatenate([
                edges[k][None],
                depths[k][None],
                segs[k].transpose(2, 0, 1),
            ], axis=0).astype(np.float32)
            results[i] = torch.from_numpy(struct)
            if self.use_cache and img_paths and img_paths[i]:
                _save_cache(_cache_path(img_paths[i], h), struct)

        return results

    @torch.no_grad()
    def _batch_midas(self, imgs: List[np.ndarray], h: int, w: int) -> List[np.ndarray]:
        if self._depth_backend != "midas":
            return [self._safe_depth(im, h, w) for im in imgs]
        try:
            tensors = [self._midas_tf(im).to(self.device) for im in imgs]
            batch   = torch.cat(tensors, 0)
            if self.half:
                batch = batch.half()
            batch = batch.to(memory_format=torch.channels_last)
            pred  = self._midas(batch)                        # (B, H', W')
            pred  = F.interpolate(pred.unsqueeze(1), size=(h, w),
                                  mode="bicubic", align_corners=False
                                  ).squeeze(1).float().cpu().numpy()
            return [((d - d.min()) / (d.max() - d.min() + 1e-8)).astype(np.float32)
                    for d in pred]
        except Exception as e:
            print(f"  [Depth] Batch MiDaS failed ({e}) — per-image.")
            return [self._safe_depth(im, h, w) for im in imgs]

    @torch.no_grad()
    def _batch_segformer(self, imgs: List[np.ndarray], h: int, w: int) -> List[np.ndarray]:
        if self._seg_backend != "segformer":
            return [self._safe_seg(im, h, w) for im in imgs]
        try:
            pils   = [Image.fromarray(im) for im in imgs]
            inputs = self._seg_proc(images=pils, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device,
                               dtype=self._dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            logits = self._seg_model(**inputs).logits
            logits = F.interpolate(logits.float(), size=(h, w),
                                   mode="bilinear", align_corners=False)
            labels = np.clip(logits.argmax(1).cpu().numpy(), 0, len(_ADE_PALETTE) - 1)
            return [(_ADE_PALETTE[lbl].astype(np.float32) / 255.0) for lbl in labels]
        except Exception as e:
            print(f"  [Seg] Batch SegFormer failed ({e}) — per-image.")
            return [self._safe_seg(im, h, w) for im in imgs]

    # ── Safe wrappers ─────────────────────────────────────────────────────────

    def _safe_edge(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        try:
            e = self.get_edge_map(img)
            return cv2.resize(e, (w, h)) if e.shape != (h, w) else e
        except Exception:
            return np.zeros((h, w), dtype=np.float32)

    def _safe_depth(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        try:
            d = self.get_depth_map(img)
            return cv2.resize(d, (w, h)) if d.shape != (h, w) else d
        except Exception:
            return np.zeros((h, w), dtype=np.float32)

    def _safe_seg(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        try:
            s = self.get_seg_map(img)
            return cv2.resize(s, (w, h)) if s.shape[:2] != (h, w) else s
        except Exception:
            return np.zeros((h, w, 3), dtype=np.float32)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def build_control_pil(
        self,
        img_np: np.ndarray,
        size:   int = 256,
        mode:   str = "combined",
    ) -> Image.Image:
        """mode: 'edge' | 'depth' | 'seg' | 'combined'"""
        mode = mode if mode in {"edge", "depth", "seg", "combined"} else "combined"
        try:
            img_np = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            return Image.new("RGB", (size, size), (128, 128, 128))
        try:
            if mode == "edge":
                e = (self.get_edge_map(img_np) * 255).astype(np.uint8)
                out = np.stack([e] * 3, -1)
            elif mode == "depth":
                d = (self.get_depth_map(img_np) * 255).astype(np.uint8)
                out = np.stack([d] * 3, -1)
            elif mode == "seg":
                out = (self.get_seg_map(img_np) * 255).astype(np.uint8)
            else:
                e   = self.get_edge_map(img_np)  * 255
                d   = self.get_depth_map(img_np) * 255
                s   = self.get_seg_map(img_np)   * 255
                out = np.clip(np.stack([e]*3,-1)*0.5
                              + np.stack([d]*3,-1)*0.2
                              + s*0.3, 0, 255).astype(np.uint8)
            return Image.fromarray(out)
        except Exception as e:
            print(f"  [Control] build_control_pil failed ({e}).")
            return Image.fromarray(img_np) if img_np is not None \
                   else Image.new("RGB", (size, size), (128, 128, 128))
