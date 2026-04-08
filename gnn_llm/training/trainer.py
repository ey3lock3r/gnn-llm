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
    run_id = config.get('wandb_run_id', 'default_run')
    resume_step = 0
    if not ignore_checkpoint:
        latest_cp = get_latest_checkpoint(run_id)
        if latest_cp:
            resume_step = model.load_checkpoint(latest_cp)

    lr = config.get('lr', 1e-3)
    warmup_steps = config.get('warmup_steps', 500)
    max_steps = config.get('max_steps', float('inf'))
    # max_steps = additional steps to train this session, not an absolute global cap
    stop_at_step = resume_step + max_steps

    pbar = tqdm(loader)
    for i, batch in enumerate(pbar):
        global_step = resume_step + i
        if global_step >= stop_at_step:
            print(f"✅ Reached max_steps ({max_steps}) from resume point {resume_step}. Stopping.")
            break
            
        # Warmup is per-session (local step i), not global_step.
        # This ensures LR ramps correctly even when resuming from late checkpoints.
        curr_lr = lr * min(1.0, (i + 1) / warmup_steps)

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
            model.save_checkpoint(get_next_checkpoint_slot(global_step, run_id), global_step)
