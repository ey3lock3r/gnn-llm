import os
import glob
import wandb
import torch

SAVE_INTERVAL = 500
base_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "/tmp"

def get_cp_paths(run_id="default"):
    return [f"{base_dir}/{run_id}_checkpoint_A.pt", f"{base_dir}/{run_id}_checkpoint_B.pt"]

def get_latest_checkpoint(run_id="default"):
    candidates = []
    for p in get_cp_paths(run_id):
        if os.path.exists(p):
            # Safe checkpoint retrieval
            try:
                ck = torch.load(p, map_location='cpu')
                candidates.append((ck.get('step', 0), p))
            except Exception:
                pass
    return max(candidates, key=lambda x: x[0])[1] if candidates else None

def get_next_checkpoint_slot(global_step, run_id="default"):
    paths = get_cp_paths(run_id)
    return paths[0] if (global_step // SAVE_INTERVAL) % 2 == 1 else paths[1]

def init_wandb(project, run_id, resume='allow'):
    wandb.init(project=project, resume=resume, id=run_id)
    wandb.define_metric('step')
    wandb.define_metric('*', step_metric='step')
