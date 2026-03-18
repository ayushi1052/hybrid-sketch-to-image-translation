"""
models/structure_gan.py
========================
Structure Extractor — upgraded pipeline:
  • Edge Map   → Canny (fast, reliable)
  • Depth Map  → Depth Anything V2  (far superior to MiDaS)
  • Seg Map    → SLIC superpixels   (better than K-means, no checkpoint needed)
                 (SAM optional upgrade — see get_seg_map_sam())

Final output: 5-channel tensor (edge + depth + seg_rgb) → fed to SketchAdapter
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Structure Extractor
# ─────────────────────────────────────────────────────────────────────────────

class StructureExtractor:
    """
    Extracts three complementary structure maps from a sketch/image.

    Maps produced:
      edge  : (H, W)    float32 [0,1]  – boundary information
      depth : (H, W)    float32 [0,1]  – spatial layout / 3-D order
      seg   : (H, W, 3) float32 [0,1]  – region grouping

    Final 5-channel tensor: (edge, depth, seg_R, seg_G, seg_B)
    """

    def __init__(self, device: str = "cuda", use_depth_anything: bool = True):
        self.device = device
        self.use_depth_anything = use_depth_anything

        if use_depth_anything:
            self._load_depth_anything()
        else:
            self._load_midas()

        print("  [StructureExtractor] Ready.")

    # ── Depth model loading ───────────────────────────────────────────────────

    def _load_depth_anything(self):
        """Load Depth Anything V2 (much better than MiDaS)."""
        try:
            from transformers import pipeline as hf_pipeline
            print("  [StructureExtractor] Loading Depth Anything V2 …")
            self.depth_pipe = hf_pipeline(
                task            = "depth-estimation",
                model           = "depth-anything/Depth-Anything-V2-Small-hf",
                device          = 0 if self.device == "cuda" else -1,
            )
            self._depth_backend = "depth_anything"
        except Exception as e:
            print(f"  [StructureExtractor] Depth Anything V2 failed ({e}), "
                  "falling back to MiDaS …")
            self._load_midas()

    def _load_midas(self):
        """Fallback: MiDaS small."""
        print("  [StructureExtractor] Loading MiDaS …")
        self.midas = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        ).to(self.device).eval()
        t = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.midas_transform = t.small_transform
        self._depth_backend = "midas"

    # ── Edge extraction ───────────────────────────────────────────────────────

    def get_edge_map(self, img_np: np.ndarray) -> np.ndarray:
        """
        Canny edge detection.
        img_np : HxWx3 uint8
        returns: HxW float32 [0,1]
        """
        gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Adaptive thresholds based on image statistics
        median = np.median(gray)
        lo     = int(max(0,   0.67 * median))
        hi     = int(min(255, 1.33 * median))
        edges  = cv2.Canny(gray, lo, hi)
        return (edges / 255.0).astype(np.float32)

    # ── Depth extraction ──────────────────────────────────────────────────────

    @torch.no_grad()
    def get_depth_map(self, img_np: np.ndarray) -> np.ndarray:
        """
        Monocular depth estimation.
        img_np : HxWx3 uint8
        returns: HxW float32 [0,1]
        """
        if self._depth_backend == "depth_anything":
            pil    = Image.fromarray(img_np)
            result = self.depth_pipe(pil)
            depth  = np.array(result["depth"], dtype=np.float32)
        else:
            inp    = self.midas_transform(img_np).to(self.device)
            pred   = self.midas(inp)
            pred   = F.interpolate(
                pred.unsqueeze(1), size=img_np.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
            depth  = pred.cpu().numpy()

        # Normalise to [0,1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)

    # ── Segmentation extraction ────────────────────────────────────────────────

    def get_seg_map(self, img_np: np.ndarray, n_segments: int = 50) -> np.ndarray:
        """
        SLIC superpixel segmentation — much better than K-means.
        Each superpixel gets the mean colour of its region.

        img_np : HxWx3 uint8
        returns: HxWx3 float32 [0,1]
        """
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb

            segments = slic(
                img_np,
                n_segments = n_segments,
                compactness= 10,
                sigma      = 1,
                start_label= 1,
            )
            # Replace each superpixel with its mean colour
            seg_img = label2rgb(segments, img_np, kind="avg", bg_label=0)
            return (seg_img / 255.0).astype(np.float32)

        except ImportError:
            # Fallback: K-means if skimage not available
            return self._get_seg_kmeans(img_np)

    def _get_seg_kmeans(self, img_np: np.ndarray, k: int = 10) -> np.ndarray:
        """K-means fallback segmentation."""
        h, w   = img_np.shape[:2]
        pixels = img_np.reshape(-1, 3).astype(np.float32)
        crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, crit, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        seg = centers[labels.flatten()].reshape(h, w, 3)
        return (seg / 255.0).astype(np.float32)

    def get_seg_map_sam(self, img_np: np.ndarray) -> np.ndarray:
        """
        Optional: SAM (Segment Anything Model) segmentation.
        Requires: pip install segment-anything
                  Download: sam_vit_b_01ec64.pth from Meta

        Much better than SLIC for complex scenes.
        """
        try:
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
            )
            if not hasattr(self, "_sam"):
                sam = sam_model_registry["vit_b"](
                    checkpoint="sam_vit_b_01ec64.pth"
                ).to(self.device)
                self._sam_gen = SamAutomaticMaskGenerator(sam)

            masks  = self._sam_gen.generate(img_np)
            h, w   = img_np.shape[:2]
            seg    = np.zeros((h, w), dtype=np.float32)
            for i, m in enumerate(masks):
                seg[m["segmentation"]] = (i + 1) / (len(masks) + 1)
            return np.stack([seg] * 3, axis=-1)

        except (ImportError, FileNotFoundError):
            print("  SAM not available, using SLIC instead.")
            return self.get_seg_map(img_np)

    # ── Combined extraction ────────────────────────────────────────────────────

    def extract(
        self,
        img_np:      np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
        use_sam:     bool = False,
    ) -> torch.Tensor:
        """
        Full extraction pipeline → 5-channel control tensor.

        img_np      : HxWx3 uint8
        target_size : (H, W) output size
        use_sam     : use SAM instead of SLIC (requires checkpoint)

        Returns: torch.Tensor (5, H, W) float [0,1]
          ch 0   : edge map
          ch 1   : depth map
          ch 2-4 : seg map (RGB)
        """
        h, w   = target_size
        img_np = cv2.resize(img_np, (w, h))

        edge  = self.get_edge_map(img_np)
        depth = self.get_depth_map(img_np)
        seg   = self.get_seg_map_sam(img_np) if use_sam \
                else self.get_seg_map(img_np)

        # Resize seg if shape mismatch (Depth Anything may return diff size)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h))
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h))

        edge_t  = torch.tensor(edge ).unsqueeze(0)           # (1, H, W)
        depth_t = torch.tensor(depth).unsqueeze(0)           # (1, H, W)
        seg_t   = torch.tensor(seg  ).permute(2, 0, 1)       # (3, H, W)

        return torch.cat([edge_t, depth_t, seg_t], dim=0)    # (5, H, W)

    def build_control_pil(
        self,
        img_np: np.ndarray,
        size:   int  = 256,
        mode:   str  = "combined",
    ) -> Image.Image:
        """
        Build a 3-channel PIL image for visualisation / ControlNet fallback.
        mode: 'edge' | 'depth' | 'combined'
        """
        img_np = cv2.resize(img_np, (size, size))

        if mode == "edge":
            e   = (self.get_edge_map(img_np)  * 255).astype(np.uint8)
            out = np.stack([e] * 3, axis=-1)

        elif mode == "depth":
            d   = (self.get_depth_map(img_np) * 255).astype(np.uint8)
            out = np.stack([d] * 3, axis=-1)

        elif mode == "combined":
            e   = (self.get_edge_map(img_np)  * 255).astype(np.float32)
            d   = (self.get_depth_map(img_np) * 255).astype(np.float32)
            e3  = np.stack([e] * 3, axis=-1)
            d3  = np.stack([d] * 3, axis=-1)
            out = np.clip(e3 * 0.6 + d3 * 0.4, 0, 255).astype(np.uint8)

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        return Image.fromarray(out)
