"""
Camadas básicas do modelo GPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer Normalization customizada"""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * ((x - mean) / torch.sqrt(var + self.eps)) + self.beta


class FeedForward(nn.Module):
    """Camada Feed Forward com ativação GELU"""

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.fc2 = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Gelu(nn.Module):
    """Ativação GELU customizada"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)
