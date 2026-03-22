import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

# ---------- EDGE ----------
def get_edge(image):
    img = np.array(image.convert("L"))
    edge = cv2.Canny(img, 100, 200)
    edge = torch.tensor(edge).unsqueeze(0).float() / 255.0
    return edge

# ---------- DEPTH (MiDaS) ----------
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def get_depth(image, device):
    inp = midas_transforms(image).to(device)
    with torch.no_grad():
        depth = midas(inp)

    depth = F.interpolate(
        depth.unsqueeze(1),
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )
    return depth

# ---------- COLOR (HEURISTIC) ----------
def get_color(sketch_tensor):
    # simple blur-like heuristic (low frequency color)
    return torch.nn.functional.avg_pool2d(sketch_tensor, kernel_size=7, stride=1, padding=3)# Utility functions for the hybrid sketch-to-image model