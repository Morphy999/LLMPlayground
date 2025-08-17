import torch.nn as nn

from ..Attention.MultiHeadAttention import MultiHeadAttention
from ..GPTUtils.FeedForward import FeedForward
from ..GPTUtils.LayerNorm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn_layer = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["n_heads"], cfg["dropout"]
        )
        self.ff_layer = FeedForward(cfg)
        self.norm_layer1 = LayerNorm(cfg["emb_dim"])
        self.norm_layer2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):

        shortcut = x

        x = self.norm_layer1(x)
        x = self.attn_layer(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x

        x = self.norm_layer2(x)
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
