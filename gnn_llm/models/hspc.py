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
        # Hyper-Stable v11.3: Native FP16 with score clamp avoids saturation
        scores = torch.matmul(h_i, h_j.t()) / (self.tau + 1e-6)
        scores = torch.clamp(scores, min=-8.0, max=8.0)
        # Ensure at least one element for batch_size=1
        return torch.softmax(scores, dim=-1)

class HSPCBlock(nn.Module):
    def __init__(self, d_model, device="cpu", use_fp16=True):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.use_fp16 = use_fp16
        
        # Predictive weight W: Predicts the next state from current + input
        self.W = nn.Linear(d_model * 2, d_model, bias=False, device=device)
        nn.init.trunc_normal_(self.W.weight, std=0.01)
        self.norm = nn.LayerNorm(d_model, device=device, eps=1e-4)
        self.eba = EntropyBalancedAttention(tau=0.02)
        
        if use_fp16:
            self.W.half()
            self.norm.half()

    def predict(self, x, z_prev):
        """Generates prediction for the next state z_l."""
        # Defensive device migration for Dual-T4 sharding
        target_device = self.W.weight.device
        dtype = self.W.weight.dtype
        x = x.to(target_device, dtype=dtype)
        z_prev = z_prev.to(target_device, dtype=dtype)
            
        combined = torch.cat([x, z_prev], dim=-1)
        h = self.W(combined)
        h_norm = self.norm(h)
        
        # Attention gating for stability
        att = self.eba(h_norm.mean(dim=1), h_norm.mean(dim=1))
        gated_att = torch.diagonal(att).view(-1, 1, 1)
        
        # Residual step: prediction of the delta
        return z_prev + h * gated_att * 0.1

class HSPCModel(GigaModel):
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
        nn.init.trunc_normal_(self.embed.weight, std=0.01)
        if use_fp16:
            self.embed.half()

        self.blocks = nn.ModuleList()
        for d in range(depth):
            block_device = self.device0 if d < depth // 2 else self.device1
            self.blocks.append(HSPCBlock(d_model, device=block_device, use_fp16=use_fp16))
            
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=self.device0)
        self.head.weight = self.embed.weight
        
        # Default Optimizer & Scaler for FP16 Stability
        self.optimizer = None 
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    def init_optimizer(self, opt_type='adam', lr=1e-4):
        if opt_type == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def inference_relaxation(self, x_embed, y_target_embed, iters=10, noise_std=0.01, relaxation='parallel', threshold=0.0, state_lr=0.1):
        """
        Iterative state relaxation (Pure HS-PC).
        Adjusts internal states z_l to minimize local prediction errors.
        """
        # 1. Sequential Forward Pass to initialize states
        # Move inputs to dev0 for start of chain
        z = [x_embed.to(self.blocks[0].W.weight.device)]
        for d in range(self.depth):
            block = self.blocks[d]
            target_device = block.W.weight.device
            z_in = z[-1].to(target_device)
            z.append(block.predict(x_embed, z_in).detach())
        
        # Final target state is the ground truth embedding
        z[self.depth] = y_target_embed.to(self.head.weight.device)
        
        # 2. Relaxation Iterations
        actual_iters = 0
        total_delta = 0
        
        for i in range(iters):
            actual_iters = i + 1
            max_delta = 0
            
            if relaxation == 'parallel':
                new_z = [z[0]] 
                for d in range(1, self.depth):
                    block_prev, block_next = self.blocks[d-1], self.blocks[d]
                    z_old = z[d]
                    z_d = z_old.clone().detach().requires_grad_(True)
                    
                    # Prediction dynamics
                    pred_d = block_prev.predict(x_embed, z[d-1])
                    err_in = F.mse_loss(z_d.to(pred_d.device), pred_d)
                    pred_next = block_next.predict(x_embed, z_d)
                    err_out = F.mse_loss(z[d+1].to(pred_next.device), pred_next)
                    
                    energy = err_in + err_out.to(err_in.device)
                    grads = torch.autograd.grad(energy, z_d, retain_graph=False, allow_unused=True)[0]
                    
                    with torch.no_grad():
                        grad_val = grads if grads is not None else torch.zeros_like(z_d)
                        # Scale shocks relative to state magnitude for deep stability
                        z_std = z_d.std().item() + 1e-6
                        shock = torch.randn_like(z_d) * (noise_std * z_std) if noise_std > 0 else 0
                        
                        # NOISE FLOOR FIX: Delta is measured based on the INTENTIONAL gradient move
                        delta = torch.abs(grad_val * state_lr).mean().item()
                        max_delta = max(max_delta, delta)
                        
                        # Apply update
                        z_d_new = z_d - (grad_val + shock) * state_lr
                    
                    new_z.append(z_d_new.detach())
                new_z.append(z[self.depth])
                z = new_z
            else:
                # Sequential Sweep (Biologically Plausible)
                for d in range(1, self.depth):
                    block_prev, block_next = self.blocks[d-1], self.blocks[d]
                    z_old = z[d]
                    z_d = z_old.clone().detach().requires_grad_(True)
                    pred_d = block_prev.predict(x_embed, z[d-1])
                    err_in = F.mse_loss(z_d.to(pred_d.device), pred_d)
                    pred_next = block_next.predict(x_embed, z_d)
                    err_out = F.mse_loss(z[d+1].to(pred_next.device), pred_next)
                    energy = err_in + err_out.to(err_in.device)
                    grads = torch.autograd.grad(energy, z_d, retain_graph=False, allow_unused=True)[0]
                    with torch.no_grad():
                        grad_val = grads if grads is not None else torch.zeros_like(z_d)
                        z_std = z_d.std().item() + 1e-6
                        shock = torch.randn_like(z_d) * (noise_std * z_std) if noise_std > 0 else 0
                        
                        delta = torch.abs(grad_val * state_lr).mean().item()
                        max_delta = max(max_delta, delta)
                        
                        z[d] = (z_d - (grad_val + shock) * state_lr).detach()
            
            total_delta += max_delta
            # Early Stopping Check
            if threshold > 0 and max_delta < threshold:
                break
                
        avg_delta = total_delta / actual_iters
        return z, actual_iters, avg_delta

    def train_step(self, x, y, **kwargs):
        config = kwargs.get('config', {})
        iters = config.get('hspc_iters', 10)
        noise = config.get('hspc_noise', 0.01)
        relaxation = config.get('hspc_relaxation', 'parallel')
        threshold = config.get('hspc_convergence_threshold', 0.0)
        state_lr = config.get('hspc_state_lr', 0.1)
        opt_type = config.get('hspc_optimizer', 'adam')
        lr = config.get('hspc_lr', 1e-4)

        if self.optimizer is None:
            self.init_optimizer(opt_type, lr)
            # Suppress intentional stream mismatch warnings in sharded Dual-T4 setups
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        else:
            # Sync learning rate for warmup compatibility
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Scale inputs (v11.2 best practice)
        x_embed = self.embed(x).detach() / math.sqrt(self.d_model)
        y_embed = self.embed(y).detach() / math.sqrt(self.d_model)

        # Mixed Precision Autocast for HS-PC stability
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # 1. Find equilibrium states (with dynamic early stopping)
            z_refined, actual_iters, avg_delta = self.inference_relaxation(
                x_embed, y_embed, iters, noise, relaxation, 
                threshold=threshold, state_lr=state_lr
            )

            # Defensive sync to prevent stream mismatch warnings on Dual-T4
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # 2. Local weight updates
            self.optimizer.zero_grad()
            total_loss = 0
            
            # Block-wise predictive loss
            for d in range(self.depth):
                block = self.blocks[d]
                target_device = block.W.weight.device
                z_pred = block.predict(x_embed, z_refined[d])
                loss = F.mse_loss(z_pred, z_refined[d+1].to(target_device))
                # Local scaling for predictive updates
                self.scaler.scale(loss).backward()
                total_loss += loss.item()

            # LM Head supervised loss (Essential for non-zero signal)
            z_final = z_refined[self.depth].to(self.head.weight.device)
            logits = self.head(z_final)
            target_tokens = y.to(logits.device)
            loss_lm = F.cross_entropy(logits.view(-1, self.vocab_size), target_tokens.view(-1))
            self.scaler.scale(loss_lm).backward()
            total_loss += loss_lm.item()

        # SUPER-AGGRESSIVE MEMORY CLEANUP
        # We must clear activations BEFORE optimizer.step() allocations for 3.2B+ models
        del z_refined, z_final, logits, target_tokens, loss, loss_lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        final_loss = total_loss / (self.depth + 1)
        
        # Return metrics dict for advanced logging
        return {
            'loss': final_loss,
            'hspc_actual_iters': actual_iters,
            'hspc_avg_delta': avg_delta
        }

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """Standard Forward-Pass Token Generation for HS-PC."""
        device0 = self.device0
        current_context = prompt_tokens.to(device0) 
        generated = []
        for _ in range(max_new_tokens):
            x_embed = (self.embed(current_context).detach() / math.sqrt(self.d_model))
            z = x_embed
            for d in range(self.depth):
                z = self.blocks[d].predict(x_embed, z)
            z = z.to(self.head.weight.device)
            logits = self.head(z[:, -1, :]) / (temperature + 1e-6)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            current_context = torch.cat([current_context, next_token.to(device0)], dim=1)
            if next_token.item() == 128009: break
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
