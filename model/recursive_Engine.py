# Recursive Engine

import torch
import torch.nn as nn

class RecursiveTransformer(nn.Module):
    """
    A model that 'thinks' by passing data through the 
    SAME layers multiple times (Recursive Refinement).
    """
    def __init__(self, config):
        super().__init__()
        self.depth = config.depth # Number of 'thought' loops
        self.layer = SharedTransformerBlock(config)
        self.norm = nn.RMSNorm(config.dim)
        
    def forward(self, x, context_memory=None):
        # Initial state
        h = x 
        for i in range(self.depth):
            # Pass through the SAME weights repeatedly
            # Optional: Add a 'thought' embedding for each step
            h = self.layer(h, context_memory)
            
        return self.norm(h)
