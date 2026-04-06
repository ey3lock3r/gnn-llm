import torch
from tqdm import tqdm
import wandb
from .utils import get_latest_checkpoint, get_next_checkpoint_slot, SAVE_INTERVAL

def run_training(model, loader, config):
    """
    Unified Training Loop (v10.0).
    Works with any GigaModel (APTP, NoProp, future models).
    """
    ignore_checkpoint = config.get('ignore_checkpoint', False)
    resume_step = 0
    if not ignore_checkpoint:
        latest_cp = get_latest_checkpoint()
        if latest_cp:
            resume_step = model.load_checkpoint(latest_cp)

    lr = config.get('lr', 1e-3)
    warmup_steps = config.get('warmup_steps', 500)

    pbar = tqdm(loader)
    for i, batch in enumerate(pbar):
        global_step = resume_step + i
        curr_lr = lr * min(1.0, (global_step + 1) / warmup_steps)

        x = batch.to(next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu')
        y = torch.roll(x, -1, dims=1)

        loss = model.train_step(x, y, lr=curr_lr)

        if torch.is_tensor(loss):
            loss_val = loss.item()
            if loss_val > 200 or torch.isnan(loss):
                print(f"⚠️ Loss spike at step {global_step}: {loss_val:.4f}")
                break
        else:
            loss_val = loss

        if global_step % 5 == 0:
            wandb.log({'loss': loss_val, 'lr': curr_lr, 'step': global_step})
            pbar.set_postfix({'step': global_step, 'loss': f'{loss_val:.4f}', 'lr': f'{curr_lr:.2e}'})

        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            model.save_checkpoint(get_next_checkpoint_slot(global_step), global_step)
