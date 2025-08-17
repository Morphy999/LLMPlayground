import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, dropout=0.1, qvk_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.n_heads = n_heads
        self.d_k = d_out // n_heads
        self.Wq = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k)

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)

        attn_scores = q @ k.transpose(2, 3)  # (batch_size, n_heads, d_k, seq_len)

        attn_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / k.shape[-1] ** 0.5, dim=-1
        )  # normalize the scores

        attn_weights = self.dropout(attn_weights)  # apply dropout to the weights

        context_vectors = (attn_weights @ v).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)

        context_vectors = context_vectors.reshape(
            batch_size, seq_len, self.d_out
        )  # (batch_size, seq_len, d_out)

        context_vectors = self.out_proj(context_vectors)

        return context_vectors  # (batch_size, seq_len, d_out)
