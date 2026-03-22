import torch
import torch.nn as nn

from models.structure_generator import StructureGenerator
from models.condition_adapter import ConditionAdapter
from models.lctn import LCTN

class SketchToImageModel(nn.Module):
    def __init__(self, unet, vae):
        super().__init__()

        self.structure_net = StructureGenerator()
        self.adapter = ConditionAdapter()
        self.lctn = LCTN()

        self.unet = unet.eval()
        self.vae = vae.eval()

        for p in self.unet.parameters():
            p.requires_grad = False

    def forward(self, sketch, latent, t):

        # Structure maps
        edge, depth, color = self.structure_net(sketch)
        cond = torch.cat([edge, depth, color], dim=1)

        # Condition features
        cond_feats = self.adapter(cond)

        x = latent

        # Inject into UNet (simplified)
        for i, block in enumerate(self.unet.down_blocks):
            x = block(x)
            if i < len(cond_feats):
                x = x + cond_feats[i]

        # LCTN refinement
        b, c, h, w = x.shape
        x_flat = x.permute(0,2,3,1).reshape(-1, c)
        x_refined = self.lctn(x_flat)
        x = x_refined.view(b, h, w, -1).permute(0,3,1,2)

        return x