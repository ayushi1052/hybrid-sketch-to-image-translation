import torch
import torch.nn as nn

class StructureGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=5):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = block(in_ch, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.dec3 = block(256, 128)
        self.dec2 = block(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d3 = self.up(e3)
        d3 = self.dec3(d3)

        d2 = self.up(d3)
        d2 = self.dec2(d2)

        out = self.final(d2)

        edge = out[:, 0:1]
        depth = out[:, 1:2]
        color = out[:, 2:5]

        return edge, depth, color