class KVCacheTransformerBlock(TransformerBlock):
    """Modified block to handle Key-Value caching for fast inference."""
    def forward(self, x, cos, sin, kv_cache=None):
        # Normal path for training
        if kv_cache is None:
            return super().forward(x, cos, sin)
        
        # Inference path
        x_norm = self.attention_norm(x)
        
        # Accessing the Attention layer's projections
        q = self.attention.q_proj(x_norm).view(1, -1, self.attention.n_heads, self.attention.head_dim)
        k = self.attention.k_proj(x_norm).view(1, -1, self.attention.n_kv_heads, self.attention.head_dim)
        v = self.attention.v_proj(x_norm).view(1, -1, self.attention.n_kv_heads, self.attention.head_dim)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update Cache (Append new K,V to history)
        k_prev, v_prev = kv_cache
        k = torch.cat([k_prev, k], dim=1)
        v = torch.cat([v_prev, v], dim=1)
        new_cache = (k, v)

        # GQA Repeat logic
        k_rep = k.repeat_interleave(self.attention.n_rep, dim=2)
        v_rep = v.repeat_interleave(self.attention.n_rep, dim=2)

        # Efficient attention on the latest token only
        out = F.scaled_dot_product_attention(q.transpose(1,2), k_rep.transpose(1,2), v_rep.transpose(1,2))
        out = out.transpose(1,2).reshape(1, -1, config["dim"])
        
        # Final residual path
        h = x + self.attention.o_proj(out)
        return h + self.feed_forward(self.ffn_norm(h)), new_cache
