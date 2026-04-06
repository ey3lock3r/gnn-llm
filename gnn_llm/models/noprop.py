import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from .base import GigaModel

class EntropyBalancedAttention(nn.Module):
    def __init__(self, tau=0.01):
        super().__init__()
        self.tau = tau
    def forward(self, h_i, h_j):
        scores = torch.matmul(h_i, h_j.t()) / self.tau
        return torch.softmax(scores, dim=0)

class NoPropBlock(nn.Module):
    def __init__(self, d_model, device="cpu", use_fp16=True):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.use_fp16 = use_fp16
        
        self.W = nn.Linear(d_model * 2, d_model, bias=False, device=device)
        self.norm = nn.LayerNorm(d_model, device=device)
        self.eba = EntropyBalancedAttention(tau=0.01)
        
        if use_fp16:
            self.W.half()
            # LayerNorm usually stays in FP32 for stability
            
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler('cuda') if 'cuda' in str(device) and use_fp16 else None

    def forward(self, x, z_prev):
        # x and z_prev must be on self.device
        if self.use_fp16:
            x = x.half()
            z_prev = z_prev.half()
            
        combined = torch.cat([x, z_prev], dim=-1)
        h = self.W(combined)
        h_norm = self.norm(h.float()).to(h.dtype) # Stabilize with FP32 Norm
        
        att = self.eba(h_norm.mean(dim=1), h_norm.mean(dim=1))
        gated_att = torch.diagonal(att).view(-1, 1, 1)
        
        # v10.1 High-Gain: 0.4 residual gain
        z_next = z_prev + h * gated_att * 0.4
        return z_next

    def train_block(self, x, z_prev, z_target):
        # Ensure inputs are on correct device and dtype
        x = x.to(self.device).detach()
        z_prev = z_prev.to(self.device).detach()
        z_target = z_target.to(self.device).detach().to(self.W.weight.dtype)
        
        self.optimizer.zero_grad()
        
        if self.scaler:
            with torch.amp.autocast('cuda'):
                z_pred = self.forward(x, z_prev)
                loss = F.mse_loss(z_pred, z_target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            z_pred = self.forward(x, z_prev)
            loss = F.mse_loss(z_pred, z_target)
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

class NoPropModel(GigaModel):
    def __init__(self, vocab_size, d_model, depth, device="cpu", use_fp16=True):
        super().__init__(vocab_size, d_model, device)
        self.depth = depth
        self.use_fp16 = use_fp16
        
        # Sharding logic (similar to APTP)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device0 = "cuda:0"
            self.device1 = "cuda:1"
        else:
            self.device0 = device
            self.device1 = device
            
        self.embed = nn.Embedding(vocab_size, d_model, device=self.device0)
        
        self.blocks = nn.ModuleList()
        for d in range(depth):
            block_device = self.device0 if d < depth // 2 else self.device1
            self.blocks.append(NoPropBlock(d_model, device=block_device, use_fp16=use_fp16))
            
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=self.device1)
        if use_fp16:
            self.head.half()
        
    def generate_noise_path(self, y_embed, depth):
        # y_embed is on device0
        path = []
        for d in range(depth + 1):
            # Cosine Noise Schedule: noise_factor = 0.5 * (1 + cos(pi * d / depth))
            noise_factor = 0.5 * (1 + math.cos(math.pi * d / depth))
            noise = torch.randn_like(y_embed) * noise_factor
            path.append(y_embed + noise)
        return path

    def train_step(self, x, y, **kwargs):
        # x, y are typically on device0
        x_embed = self.embed(x).detach()
        y_embed = self.embed(y).detach()
        
        # Ground truth path (asynchronous targets)
        path = self.generate_noise_path(y_embed, self.depth)
        
        total_loss = 0
        for d in range(self.depth):
            # NoProp Blocks are independent – they each learn to denoise one step of the path
            loss = self.blocks[d].train_block(x_embed, path[d], path[d+1])
            total_loss += loss
            
        return total_loss / self.depth

    def save_checkpoint(self, path, step):
        checkpoint = {
            'step': step,
            'state_dict': self.state_dict(),
            'd_model': self.d_model,
            'depth': self.depth,
            'use_fp16': self.use_fp16
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return 0
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint['step']
