"""utils.py — shared utilities"""

from __future__ import annotations

import os
import json
import random
import numpy as np
import torch
from typing import Any, Dict


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        return 0


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.sum = self.avg = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        if not np.isfinite(val):
            return
        self.val    = float(val)
        self.sum   += float(val) * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


def log_config(config: Dict[str, Any], output_dir: str) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
    except Exception as e:
        print(f"  [Warning] Could not save config: {e}")