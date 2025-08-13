import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_in, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_in))
        self.shift = nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * x_normalized + self.shift
