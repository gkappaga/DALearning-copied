import torch
import torch.nn as nn
import numpy as np
from config import Config

class SinusoidalEmbedding(nn.Module):
    """
    Standard sinusoidal positional embedding.
    AMBIGUITY FLAG: Paper says "embedding of dimension 4" for s. 
    It doesn't specify if it is learnable or fixed sinusoidal.
    We implement a small MLP projection on top of sinusoidal features 
    to match the flexibility implied by "embedding".
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Frequencies for sinusoidal embedding
        self.register_buffer('freqs', 2 * np.pi * torch.arange(1, dim // 2 + 1).float())

    def forward(self, x):
        # x: [Batch, 1]
        x = x.float()
        emb = x * self.freqs[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class DriftNetwork(nn.Module):
    """
    Appendix D.2.4: 
    - MLP, 5 hidden layers, 256 hidden dim.
    - Inputs: X_s (3), Condition X_0 (embed 4), Time s (embed 4).
    """
    def __init__(self):
        super().__init__()
        
        # Embeddings
        # For s (scalar) -> 4 dim
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(4), # Base features
            nn.Linear(4, Config.EMBED_DIM),
            nn.SiLU()
        )
        
        # For X0 (vector 3) -> 4 dim
        # "embedding X0 similar to how s is embedded" (App D.1.3/D.2.4)
        # This implies a projection layer.
        self.cond_embed = nn.Sequential(
            nn.Linear(3, Config.EMBED_DIM),
            nn.SiLU()
        )
        
        input_dim = 3 + Config.EMBED_DIM + Config.EMBED_DIM # X_s + Emb(X0) + Emb(s)
        
        layers = []
        layers.append(nn.Linear(input_dim, Config.HIDDEN_DIM))
        layers.append(nn.SiLU())
        
        for _ in range(Config.LAYERS - 1):
            layers.append(nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(Config.HIDDEN_DIM, 3)) # Output dim 3 (velocity)
        self.net = nn.Sequential(*layers)
        
    def forward(self, x_s, s, x_0):
        # x_s: [B, 3]
        # s: [B, 1]
        # x_0: [B, 3]
        
        s_emb = self.time_embed(s)
        x0_emb = self.cond_embed(x_0)
        
        net_input = torch.cat([x_s, s_emb, x0_emb], dim=1)
        return self.net(net_input)