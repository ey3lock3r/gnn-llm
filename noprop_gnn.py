import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EntropyBalancedAttention(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
    def forward(self, h_i, h_j):
        scores = torch.matmul(h_i, h_j.t()) / self.tau
        return torch.softmax(scores, dim=0)

class NoPropBlock(nn.Module):
    """
    GigaGraph v9.0: Block-wise Denoising (March 2025).
    Each block learns to map the previous latent state (Z_t-1) 
    to a cleaner state (Z_t) conditioned on the original input X.
    """
    def __init__(self, d_model, device="cpu"):
        super().__init__()
        self.d_model = d_model
        
        # Internal Weights (Local BP)
        self.W = nn.Linear(d_model * 2, d_model, bias=False, device=device)
        self.norm = nn.LayerNorm(d_model, device=device)
        self.eba = EntropyBalancedAttention(tau=0.02)
        
        # Local Optimizer (Adam) for this specific block
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x, z_prev):
        """
        x: Original data input (Embedding)
        z_prev: The latent state being denoised
        """
        # Feature concat: Input + Noisy Latent
        combined = torch.cat([x, z_prev], dim=-1)
        h = self.W(combined)
        h_norm = self.norm(h)
        
        # Dynamic Adjacency (EBA Graph)
        att = self.eba(h_norm.mean(dim=1), h_norm.mean(dim=1))
        gated_att = torch.diagonal(att).view(-1, 1, 1)
        
        z_next = z_prev + h * gated_att * 0.1
        return z_next

    def train_block(self, x, z_prev, z_target):
        """
        Local Training: MSE(Z_next, Z_target)
        Eliminates the global BP pass.
        """
        self.optimizer.zero_grad()
        z_pred = self.forward(x, z_prev)
        loss = F.mse_loss(z_pred, z_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class NoPropGNN(nn.Module):
    def __init__(self, vocab_size, d_model, depth, device="cpu"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, device=device)
        self.blocks = nn.ModuleList([NoPropBlock(d_model, device=device) for _ in range(depth)])
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=device)
        
    def generate_noise_path(self, y_embed, depth):
        """
        Generates a sequence of latents with decreasing noise 
        levels (Diffusion path) to be used as local targets.
        """
        path = []
        for d in range(depth + 1):
            noise_factor = (depth - d) / depth
            noise = torch.randn_like(y_embed) * noise_factor
            path.append(y_embed + noise)
        return path

    def inference(self, ids):
        x = self.embed(ids)
        # Start with maximal noise (Initial Latent)
        z = torch.randn_like(x)
        for block in self.blocks:
            z = block(x, z)
        return self.head(z)
