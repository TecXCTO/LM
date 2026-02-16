# Recursive Context Transformer


import torch
import torch.nn as nn

class RecursiveBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        # Shared weights for all "thinking" steps
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, steps=4):
        h = x
        for _ in range(steps):
            # The same layers are used recursively to refine the hidden state
            attn_out, _ = self.attn(self.norm(h), h, h)
            h = h + attn_out
            h = h + self.mlp(self.norm(h))
        return h

class LongContextR_LLM(nn.Module):
    def __init__(self, vocab_size, dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.recursive_engine = RecursiveBlock(dim, n_heads=8)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        # Process through the recursive loop
        x = self.recursive_engine(x, steps=6) 
        return self.head(x)
