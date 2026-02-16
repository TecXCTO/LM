# The Full LLM System Wrapper

import torch.nn as nn

class CustomLLM(nn.Module):
    """
    The complete LLM architecture: 
    Embeddings -> N x Transformer Blocks -> Final Norm -> LM Head
    """
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, mlp_dim, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 1. Token Embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # 2. Rotary Positional Embeddings (Shared across layers)
        self.rotary_emb = RotaryEmbedding(dim // n_heads, max_seq_len)
        
        # 3. Stack of Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        
        # 4. Final Output Normalization
        self.norm = RMSNorm(dim)
        
        # 5. Output Head (Maps hidden states back to vocabulary)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight Tying (Standard practice to save parameters and improve performance)
        self.tok_embeddings.weight = self.output.weight

    def forward(self, tokens, cos=None, sin=None):
        _batch, seq_len = tokens.shape
        
        # Generate RoPE embeddings if not provided
        if cos is None or sin is None:
            cos, sin = self.rotary_emb(tokens, seq_len)
        
        h = self.tok_embeddings(tokens)
        
        # Pass through all layers
        for layer in self.layers:
            h = layer(h, cos, sin)
            
        h = self.norm(h)
        logits = self.output(h)
        
        return logits

# --- Hyperparameters for a "Small" 2026-style Model ---
config = {
    "vocab_size": 50257, # GPT-2 Standard
    "dim": 768,          # Embedding Dimension
    "n_layers": 12,      # Number of blocks
    "n_heads": 12,       # Attention heads
    "n_kv_heads": 4,     # GQA: 4 KV heads for 12 Query heads
    "mlp_dim": 3072,     # SwiGLU hidden dimension
    "max_seq_len": 2048
}

# Initialize the model
model = CustomLLM(**config)
print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
