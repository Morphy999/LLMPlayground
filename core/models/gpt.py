import torch
import torch.nn as nn

from .components.layers import LayerNorm
from .components.transformer import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.emb_layer = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb_layer = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.ln_f = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape

        tok_embeddings = self.emb_layer(in_idx)

        pos_embeddings = self.pos_emb_layer(
            torch.arange(seq_len).expand(batch_size, seq_len).to(in_idx.device)
        )

        x = tok_embeddings + pos_embeddings

        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.out_head(x)

        return logits
