import torch
import torch.nn as nn

class ConditionAdapter(nn.Module):
    def __init__(self, in_ch=5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)

        self.act = nn.ReLU()

    def forward(self, cond):
        f1 = self.act(self.conv1(cond))
        f2 = self.act(self.conv2(f1))
        f3 = self.act(self.conv3(f2))

        return [f1, f2, f3]