import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from .base import GigaModel

class EntropyBalancedAttention(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
    def forward(self, h_i, h_j):
        scores = torch.matmul(h_i, h_j.t()) / self.tau
        return torch.softmax(scores, dim=0)

class APTPBlock(nn.Module):
    def __init__(self, d_model, device="cpu"):
        super().__init__()
        std = 0.5 * math.sqrt(1.0 / (d_model))
        self.W = nn.Parameter(torch.empty(d_model, d_model, device=device).uniform_(-std, std), requires_grad=False)
        self.R = nn.Parameter(torch.empty(d_model, d_model, device=device).uniform_(-std * 0.1, std * 0.1), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(1, device=device) * 60.0, requires_grad=False)
        self.norm = nn.LayerNorm(d_model, device=device)
        self.norm.weight.requires_grad = False
        self.norm.bias.requires_grad = False
        self.eba = EntropyBalancedAttention(tau=0.02)
        
    def forward_pass(self, h):
        h_norm = self.norm(h)
        att = self.eba(h_norm.mean(dim=1), h_norm.mean(dim=1))
        gated_att = torch.diagonal(att).view(-1, 1, 1)
        z = F.linear(h_norm, self.W)
        h_out = F.gelu(z) * gated_att
        return h + h_out * 0.08

    def compute_drm_error(self, e):
        mod = torch.matmul(torch.tanh(e * self.gamma), self.R.t())
        norm = mod.norm(p=2, dim=-1, keepdim=True)
        max_norm = 1.0
        return mod * (max_norm / torch.clamp(norm, min=max_norm))

    def update_weights(self, h_p1, h_p2, lr=1e-4, weight_decay=1e-6):
        error_signal = h_p2 - h_p1
        error_flat = error_signal.view(-1, error_signal.size(-1)).to(torch.float32)
        h_flat = h_p1.view(-1, h_p1.size(-1)).to(torch.float32)
        delta_W = torch.matmul(error_flat.t(), h_flat)
        update_norm = delta_W.norm()
        max_update_norm = 1.0
        if update_norm > max_update_norm:
            delta_W *= (max_update_norm / (update_norm + 1e-6))
        norm_factor = math.sqrt(h_p1.size(1))
        self.W.add_(delta_W.to(self.W.dtype), alpha=lr / norm_factor)
        self.W.sub_(self.W, alpha=weight_decay)

class APTPModel(GigaModel):
    def __init__(self, vocab_size=128256, depth=32, d_model=3072, device="cpu"):
        super().__init__(vocab_size, d_model, device)
        self.depth = depth
        
        # Simple sharding logic
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device0 = "cuda:0"
            self.device1 = "cuda:1"
        else:
            self.device0 = device
            self.device1 = device
            
        self.embeddings = nn.Embedding(vocab_size, d_model).to(self.device0)
        self.pos_emb = nn.Parameter(torch.zeros(1, 4096, d_model, device=self.device0), requires_grad=False)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            device = self.device0 if i < (depth // 2) else self.device1
            self.blocks.append(APTPBlock(d_model, device=device))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(self.device1)
        for param in self.parameters():
            param.requires_grad = False

    def train_step(self, x, y, lr=1e-4, **kwargs):
        h = self.embeddings(x) + self.pos_emb[:, :x.size(1), :]
        p1_activations = []
        for i, block in enumerate(self.blocks):
            if i == (self.depth // 2) and self.device0 != self.device1:
                h = h.to(self.device1)
            h = block.forward_pass(h)
            p1_activations.append(h)
            
        logits = self.lm_head(h)
        
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            target = F.one_hot(y, num_classes=logits.size(-1)).float().to(logits.device)
            error_logits = target - probs
            global_error = torch.matmul(error_logits, self.lm_head.weight)
            
        for i, block in enumerate(self.blocks):
            h_p1 = p1_activations[i]
            e_g = global_error if i >= (self.depth // 2) else global_error.to(self.device0)
            e_l = block.compute_drm_error(e_g)
            h_p2 = block.forward_pass(h_p1 + e_l)
            block.update_weights(h_p1, h_p2, lr=lr)
            
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1).to(logits.device))

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
        self.embeddings.to(self.device0)
        for i, block in enumerate(self.blocks):
            device = self.device0 if i < (self.depth // 2) else self.device1
            block.to(device)
        self.lm_head.to(self.device1)
        return checkpoint['step']
