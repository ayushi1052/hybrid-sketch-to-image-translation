import torch
import torch.nn as nn
from models.condition_adapter import ConditionAdapter
from models.lctn import LCTN

class SketchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = ConditionAdapter()
        self.lctn = LCTN()

    def forward(self, latent, cond):

        # Condition injection
        cond_feat = self.adapter(cond)
        latent = latent + 0.1 * cond_feat

        # LCTN refinement
        b, c, h, w = latent.shape
        flat = latent.permute(0,2,3,1).reshape(-1, c)
        flat = self.lctn(flat)
        latent = flat.view(b,h,w,c).permute(0,3,1,2)

        return latent