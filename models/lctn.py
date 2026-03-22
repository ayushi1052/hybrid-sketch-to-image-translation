import torch.nn as nn

class LCTN(nn.Module):
    def __init__(self, dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x):
        return self.net(x)