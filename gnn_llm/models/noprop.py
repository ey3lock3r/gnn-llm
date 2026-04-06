import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from .base import GigaModel

class EntropyBalancedAttention(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
    def forward(self, h_i, h_j):
        # Hyper-Stable v11.3: Native FP16 with score clamp avoids saturation
        # safely without the massive FP32 conversion overhead.
        scores = torch.matmul(h_i, h_j.t()) / self.tau
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        return torch.softmax(scores, dim=0)

class NoPropBlock(nn.Module):
    def __init__(self, d_model, device="cpu", use_fp16=True):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.use_fp16 = use_fp16
        
        self.W = nn.Linear(d_model * 2, d_model, bias=False, device=device)
        # Hyper-Stable v11.2: Hardened initialization and LayerNorm epsilon
        nn.init.trunc_normal_(self.W.weight, std=0.01)
        self.norm = nn.LayerNorm(d_model, device=device, eps=1e-4) # Hardened for FP16
        self.eba = EntropyBalancedAttention(tau=0.02)
        
        if use_fp16:
            self.W.half()
            self.norm.half()

    def forward(self, x, z_prev):
        if self.use_fp16:
            x = x.half()
            z_prev = z_prev.half()
            
        combined = torch.cat([x, z_prev], dim=-1)
        h = self.W(combined)
        h_norm = self.norm(h) # FP16 native (eps=1e-4 protects against underflow)
        
        att = self.eba(h_norm.mean(dim=1), h_norm.mean(dim=1))
        gated_att = torch.diagonal(att).view(-1, 1, 1)
        
        # Stability Gain
        z_next = z_prev + h * gated_att * 0.1
        return z_next

class NoPropModel(GigaModel):
    def __init__(self, vocab_size, d_model, depth, device="cpu", use_fp16=True):
        super().__init__(vocab_size, d_model, device)
        self.depth = depth
        self.use_fp16 = use_fp16
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device0 = "cuda:0"
            self.device1 = "cuda:1"
        else:
            self.device0 = device
            self.device1 = device
            
        self.embed = nn.Embedding(vocab_size, d_model, device=self.device0)
        # Hyper-Stable v11.2: Scaled embedding initialization
        nn.init.trunc_normal_(self.embed.weight, std=0.01)
        if use_fp16:
            self.embed.half()

        self.blocks = nn.ModuleList()
        for d in range(depth):
            block_device = self.device0 if d < depth // 2 else self.device1
            self.blocks.append(NoPropBlock(d_model, device=block_device, use_fp16=use_fp16))
            
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=self.device1)
        
        # Hyper-Stable v11.4: Weight Tying + Global Optimizer
        self.head.weight = self.embed.weight
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, eps=1e-4)
        
    def generate_noise_path(self, y_embed, depth):
        path = []
        for d in range(depth + 1):
            noise_factor = 0.5 * (1 + math.cos(math.pi * d / depth))
            noise = torch.randn_like(y_embed) * noise_factor
            path.append(y_embed + noise)
        return path

    def train_step(self, x, y, **kwargs):
        # Hyper-Stable v11.2: Input Scaling (Transformer best practice)
        x_embed = self.embed(x).detach() / math.sqrt(self.d_model)
        y_embed = self.embed(y).detach() / math.sqrt(self.d_model)
        path = self.generate_noise_path(y_embed, self.depth)
        
        self.optimizer.zero_grad()
        total_loss = 0
        
        for d in range(self.depth):
            block = self.blocks[d]
            target_device = next(block.parameters()).device
            
            x_d = x_embed.to(target_device)
            z_prev = path[d].to(target_device)
            z_target = path[d+1].to(target_device).to(block.W.weight.dtype)
            
            z_pred = block.forward(x_d, z_prev)
            loss = F.mse_loss(z_pred, z_target)
            loss.backward()
            total_loss += loss.item()
            
        # Global Kernel Fusion (v11.4): One clip & step across all blocks
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
        self.optimizer.step()
            
        return total_loss / self.depth

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """
        Denoising Diffusion Generation (v11.1).
        Iteratively denoises noise tokens conditioned on the prompt.
        """
        device = self.device0
        current_context = prompt_tokens.to(device) # [Batch, Seq]
        
        generated = []
        for _ in range(max_new_tokens):
            # 1. Condition on prompt (Hyper-Stable scaling applied)
            x_embed = (self.embed(current_context).detach() / math.sqrt(self.d_model))
            
            # 2. Sample initial noise
            z = torch.randn(x_embed.size(0), 1, self.d_model, device=device)
            if self.use_fp16:
                z = z.half()
                x_embed = x_embed.half()
                
            # 3. Denoising Chain
            for d in range(self.depth):
                block = self.blocks[d]
                target_device = next(block.parameters()).device
                if z.device != target_device:
                    z = z.to(target_device)
                
                x_pool = x_embed.to(target_device).mean(dim=1, keepdim=True)
                z = block.forward(x_pool, z)
            
            # 4. Final Decode
            z = z.to(self.device1)
            logits = self.head(z[:, -1, :]) / (temperature + 1e-6)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated.append(next_token)
            current_context = torch.cat([current_context, next_token.to(device)], dim=1)
            
            if next_token.item() == 128009: 
                break
                
        return torch.cat(generated, dim=1)

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
