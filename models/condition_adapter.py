import torch.nn as nn

class ConditionAdapter(nn.Module):
    def __init__(self, in_ch=5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 4, 3, padding=1)  # match latent channels

        self.act = nn.ReLU()

    def forward(self, cond):
        x = self.act(self.conv1(cond))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x