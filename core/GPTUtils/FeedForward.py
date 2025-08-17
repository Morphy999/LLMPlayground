import torch.nn as nn

from .Gelu import Gelu


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["emb_dim"]
        self.d_ff = self.emb_dim * 4
        self.layers = nn.Sequential(
            nn.Linear(self.emb_dim, self.d_ff), Gelu(), nn.Linear(self.d_ff, self.emb_dim)
        )

    def forward(self, x):
        return self.layers(x)
