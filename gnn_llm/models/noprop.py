import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .base import GigaModel

class EntropyBalancedAttention(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
    def forward(self, h_i, h_j):
        scores = torch.matmul(h_i, h_j.t()) / self.tau
        return torch.softmax(scores, dim=0)

class NoPropBlock(nn.Module):
    def __init__(self, d_model, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.W = nn.Linear(d_model * 2, d_model, bias=False, device=device)
        self.norm = nn.LayerNorm(d_model, device=device)
        self.eba = EntropyBalancedAttention(tau=0.02)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x, z_prev):
        combined = torch.cat([x, z_prev], dim=-1)
        h = self.W(combined)
        h_norm = self.norm(h)
        att = self.eba(h_norm.mean(dim=1), h_norm.mean(dim=1))
        gated_att = torch.diagonal(att).view(-1, 1, 1)
        return z_prev + h * gated_att * 0.1

    def train_block(self, x, z_prev, z_target):
        self.optimizer.zero_grad()
        z_pred = self.forward(x, z_prev)
        loss = F.mse_loss(z_pred, z_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class NoPropModel(GigaModel):
    def __init__(self, vocab_size, d_model, depth, device="cpu"):
        super().__init__(vocab_size, d_model, device)
        self.depth = depth
        self.embed = nn.Embedding(vocab_size, d_model, device=device)
        self.blocks = nn.ModuleList([NoPropBlock(d_model, device=device) for _ in range(depth)])
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=device)
        
    def generate_noise_path(self, y_embed, depth):
        path = []
        for d in range(depth + 1):
            noise_factor = (depth - d) / depth
            noise = torch.randn_like(y_embed) * noise_factor
            path.append(y_embed + noise)
        return path

    def train_step(self, x, y, **kwargs):
        x_embed = self.embed(x).detach()  # detach: conditioning only, each block owns its own graph
        y_embed = self.embed(y).detach()
        path = self.generate_noise_path(y_embed, self.depth)
        
        total_loss = 0
        for d in range(self.depth):
            loss = self.blocks[d].train_block(x_embed, path[d].detach(), path[d+1].detach())
            total_loss += loss
        return total_loss / self.depth

    def save_checkpoint(self, path, step):
        checkpoint = {
            'step': step,
            'state_dict': self.state_dict(),
            'd_model': self.d_model,
            'depth': self.depth
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return 0
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        self.to(self.device)
        return checkpoint['step']
