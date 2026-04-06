import os
import glob
import wandb
import torch

CP_PATH_A = "/kaggle/working/checkpoint_A.pt"
CP_PATH_B = "/kaggle/working/checkpoint_B.pt"
SAVE_INTERVAL = 500

def get_latest_checkpoint():
    candidates = []
    for p in [CP_PATH_A, CP_PATH_B]:
        if os.path.exists(p):
            ck = torch.load(p, map_location='cpu')
            candidates.append((ck.get('step', 0), p))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None

def get_next_checkpoint_slot(global_step):
    return CP_PATH_A if (global_step // SAVE_INTERVAL) % 2 == 1 else CP_PATH_B

def init_wandb(project, run_id, resume='allow'):
    wandb.define_metric('step')
    wandb.define_metric('*', step_metric='step')
    wandb.init(project=project, resume=resume, id=run_id)
