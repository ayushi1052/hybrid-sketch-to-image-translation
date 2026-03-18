"""
models/structure_gan.py
========================
Structure Extractor — with full exception handling

Each extractor (edge, depth, seg) fails gracefully:
  - Depth Anything V2 → falls back to MiDaS → falls back to gradient-based depth
  - SLIC segmentation → falls back to K-means → falls back to colour channels
  - Canny edge        → falls back to Sobel → falls back to grayscale
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple


class StructureExtractor:
    """
    Extracts 5-channel structure tensor (edge + depth + seg) from an image.
    Every step has fallbacks so extraction never fully crashes.
    """

    def __init__(self, device: str = "cuda", use_depth_anything: bool = True):
        self.device = device
        self._depth_backend = None

        if use_depth_anything:
            self._load_depth_anything()
        else:
            self._load_midas()

        print("  [StructureExtractor] Ready.")

    # ── Depth model loading ───────────────────────────────────────────────────

    def _load_depth_anything(self):
        """Try Depth Anything V2 → MiDaS → gradient fallback."""
        try:
            from transformers import pipeline as hf_pipeline
            print("  [StructureExtractor] Loading Depth Anything V2 …")
            self.depth_pipe = hf_pipeline(
                task   = "depth-estimation",
                model  = "depth-anything/Depth-Anything-V2-Small-hf",
                device = 0 if self.device == "cuda" else -1,
            )
            self._depth_backend = "depth_anything"
            print("  [StructureExtractor] Depth Anything V2: OK")
        except ImportError:
            print("  [StructureExtractor] transformers not installed — trying MiDaS.")
            self._load_midas()
        except Exception as e:
            print(f"  [StructureExtractor] Depth Anything V2 failed ({e}) — trying MiDaS.")
            self._load_midas()

    def _load_midas(self):
        """Try MiDaS → gradient fallback."""
        try:
            print("  [StructureExtractor] Loading MiDaS …")
            self.midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            ).to(self.device).eval()
            t = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.midas_transform = t.small_transform
            self._depth_backend = "midas"
            print("  [StructureExtractor] MiDaS: OK")
        except Exception as e:
            print(f"  [StructureExtractor] MiDaS failed ({e}).")
            print("  [StructureExtractor] Using gradient-based depth fallback.")
            self._depth_backend = "gradient"

    # ── Edge extraction ───────────────────────────────────────────────────────

    def get_edge_map(self, img_np: np.ndarray) -> np.ndarray:
        """
        Canny edge detection with adaptive thresholds.
        Falls back to Sobel → grayscale on failure.
        img_np : HxWx3 uint8
        returns: HxW float32 [0,1]
        """
        # Primary: Canny
        try:
            gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            median = np.median(gray)
            lo     = int(max(0,   0.67 * median))
            hi     = int(min(255, 1.33 * median))
            # Avoid degenerate case where lo == hi
            if lo == hi:
                lo, hi = 50, 150
            edges  = cv2.Canny(gray, lo, hi)
            return (edges / 255.0).astype(np.float32)
        except cv2.error as e:
            print(f"  [Edge] Canny failed ({e}) — trying Sobel fallback.")
        except Exception as e:
            print(f"  [Edge] Unexpected error ({e}) — trying Sobel fallback.")

        # Fallback 1: Sobel
        try:
            gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
            sx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag   = np.sqrt(sx**2 + sy**2)
            mag   = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
            return mag.astype(np.float32)
        except Exception as e:
            print(f"  [Edge] Sobel failed ({e}) — using grayscale fallback.")

        # Fallback 2: grayscale
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return (gray / 255.0).astype(np.float32)
        except Exception:
            return np.zeros(img_np.shape[:2], dtype=np.float32)

    # ── Depth extraction ──────────────────────────────────────────────────────

    @torch.no_grad()
    def get_depth_map(self, img_np: np.ndarray) -> np.ndarray:
        """
        Monocular depth estimation.
        Falls back across: Depth Anything V2 → MiDaS → gradient → zeros.
        img_np : HxWx3 uint8
        returns: HxW float32 [0,1]
        """
        # Primary backend
        if self._depth_backend == "depth_anything":
            try:
                result = self.depth_pipe(Image.fromarray(img_np))
                depth  = np.array(result["depth"], dtype=np.float32)
                depth  = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                return depth.astype(np.float32)
            except Exception as e:
                print(f"  [Depth] Depth Anything failed ({e}) — falling back to MiDaS.")
                self._load_midas()
                return self.get_depth_map(img_np)   # retry with MiDaS

        if self._depth_backend == "midas":
            try:
                inp  = self.midas_transform(img_np).to(self.device)
                pred = self.midas(inp)
                pred = F.interpolate(
                    pred.unsqueeze(1), size=img_np.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze()
                depth = pred.cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                return depth.astype(np.float32)
            except torch.cuda.OutOfMemoryError:
                print("  [Depth] MiDaS OOM — using gradient fallback.")
                self._depth_backend = "gradient"
            except Exception as e:
                print(f"  [Depth] MiDaS failed ({e}) — using gradient fallback.")
                self._depth_backend = "gradient"

        # Fallback: gradient-based pseudo-depth (bottom of image = closer)
        try:
            h, w  = img_np.shape[:2]
            depth = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
            return depth.astype(np.float32)
        except Exception:
            return np.zeros(img_np.shape[:2], dtype=np.float32)

    # ── Segmentation extraction ────────────────────────────────────────────────

    def get_seg_map(self, img_np: np.ndarray, n_segments: int = 50) -> np.ndarray:
        """
        Segmentation via SLIC → K-means → colour channels.
        img_np : HxWx3 uint8
        returns: HxWx3 float32 [0,1]
        """
        # Primary: SLIC superpixels
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb
            segments = slic(
                img_np, n_segments=n_segments,
                compactness=10, sigma=1, start_label=1
            )
            seg_img = label2rgb(segments, img_np, kind="avg", bg_label=0)
            return (seg_img / 255.0).astype(np.float32)
        except ImportError:
            pass   # scikit-image not installed — try K-means
        except Exception as e:
            print(f"  [Seg] SLIC failed ({e}) — trying K-means.")

        # Fallback 1: K-means
        try:
            return self._seg_kmeans(img_np)
        except cv2.error as e:
            print(f"  [Seg] K-means failed ({e}) — using colour channels.")
        except Exception as e:
            print(f"  [Seg] K-means failed ({e}) — using colour channels.")

        # Fallback 2: normalised colour channels
        try:
            return (img_np / 255.0).astype(np.float32)
        except Exception:
            return np.zeros((*img_np.shape[:2], 3), dtype=np.float32)

    def _seg_kmeans(self, img_np: np.ndarray, k: int = 10) -> np.ndarray:
        h, w   = img_np.shape[:2]
        pixels = img_np.reshape(-1, 3).astype(np.float32)
        if len(pixels) < k:
            raise ValueError(f"Image too small for K-means (k={k}, pixels={len(pixels)})")
        crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, crit, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        return (centers[labels.flatten()].reshape(h, w, 3) / 255.0).astype(np.float32)

    # ── Combined extraction ────────────────────────────────────────────────────

    def extract(
        self,
        img_np:      np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Full pipeline → 5-channel tensor (edge + depth + seg_rgb).
        Returns zeros for any channel that fails.

        img_np      : HxWx3 uint8
        target_size : (H, W)
        returns     : torch.Tensor (5, H, W) float [0,1]
        """
        h, w = target_size

        # Validate input
        if img_np is None:
            raise ValueError("img_np is None — cannot extract structure maps.")
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError(
                f"Expected HxWx3 image, got shape: {img_np.shape}"
            )
        if img_np.dtype != np.uint8:
            # Auto-convert float images
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        try:
            img_np = cv2.resize(img_np, (w, h))
        except Exception as e:
            raise RuntimeError(f"Image resize failed: {e}") from e

        # Edge
        try:
            edge = self.get_edge_map(img_np)
        except Exception as e:
            print(f"  [Extract] Edge extraction failed ({e}) — using zeros.")
            edge = np.zeros((h, w), dtype=np.float32)

        # Depth
        try:
            depth = self.get_depth_map(img_np)
        except Exception as e:
            print(f"  [Extract] Depth extraction failed ({e}) — using zeros.")
            depth = np.zeros((h, w), dtype=np.float32)

        # Seg
        try:
            seg = self.get_seg_map(img_np)
        except Exception as e:
            print(f"  [Extract] Seg extraction failed ({e}) — using zeros.")
            seg = np.zeros((h, w, 3), dtype=np.float32)

        # Resize outputs if shape differs (e.g. Depth Anything returns different size)
        try:
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h))
        except Exception:
            depth = np.zeros((h, w), dtype=np.float32)

        try:
            if seg.shape[:2] != (h, w):
                seg = cv2.resize(seg, (w, h))
        except Exception:
            seg = np.zeros((h, w, 3), dtype=np.float32)

        # Assemble tensor
        try:
            return torch.cat([
                torch.tensor(edge ).unsqueeze(0),          # (1, H, W)
                torch.tensor(depth).unsqueeze(0),          # (1, H, W)
                torch.tensor(seg  ).permute(2, 0, 1),      # (3, H, W)
            ], dim=0)                                      # (5, H, W)
        except Exception as e:
            raise RuntimeError(
                f"Failed to assemble structure tensor: {e}\n"
                f"  edge: {edge.shape}, depth: {depth.shape}, seg: {seg.shape}"
            ) from e

    def build_control_pil(
        self,
        img_np: np.ndarray,
        size:   int  = 256,
        mode:   str  = "combined",
    ) -> Image.Image:
        """
        Build 3-channel PIL image for visualisation.
        mode: 'edge' | 'depth' | 'combined'
        Falls back to grayscale sketch on any failure.
        """
        valid_modes = {"edge", "depth", "combined"}
        if mode not in valid_modes:
            print(f"  [Control] Unknown mode '{mode}' — using 'combined'.")
            mode = "combined"

        try:
            img_np = cv2.resize(img_np, (size, size))
        except Exception as e:
            print(f"  [Control] Resize failed ({e}) — returning blank image.")
            return Image.new("RGB", (size, size), (128, 128, 128))

        try:
            if mode == "edge":
                e   = (self.get_edge_map(img_np)  * 255).astype(np.uint8)
                out = np.stack([e] * 3, axis=-1)

            elif mode == "depth":
                d   = (self.get_depth_map(img_np) * 255).astype(np.uint8)
                out = np.stack([d] * 3, axis=-1)

            else:  # combined
                e   = (self.get_edge_map(img_np)  * 255).astype(np.float32)
                d   = (self.get_depth_map(img_np) * 255).astype(np.float32)
                out = np.clip(
                    np.stack([e]*3, -1)*0.6 + np.stack([d]*3, -1)*0.4, 0, 255
                ).astype(np.uint8)

            return Image.fromarray(out)

        except Exception as e:
            print(f"  [Control] build_control_pil failed ({e}) — returning sketch.")
            try:
                return Image.fromarray(img_np)
            except Exception:
                return Image.new("RGB", (size, size), (128, 128, 128))