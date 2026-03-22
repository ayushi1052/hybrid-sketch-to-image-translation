import torch
import torch.nn as nn

class LCTN(nn.Module):
    def __init__(self, input_dim=4, hidden=256, output_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)