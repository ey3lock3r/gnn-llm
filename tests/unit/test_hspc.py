import pytest
import torch
import sys
sys.path.insert(0, '/home/ey3lock3r/Source/gnn-llm')

from gnn_llm import build_model

VOCAB, D_MODEL, DEPTH = 512, 64, 2

def make_batch():
    return torch.randint(0, VOCAB, (2, 16))

def test_hspc_build():
    model = build_model("hspc", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu", use_fp16=False)
    assert model is not None

def test_hspc_train_step_parallel():
    model = build_model("hspc", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu", use_fp16=False)
    x = make_batch()
    y = torch.roll(x, -1, dims=1)
    
    config = {
        'hspc_iters': 2,
        'hspc_relaxation': 'parallel',
        'hspc_optimizer': 'adam'
    }
    loss = model.train_step(x, y, config=config)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))

def test_hspc_train_step_sequential():
    model = build_model("hspc", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu", use_fp16=False)
    x = make_batch()
    y = torch.roll(x, -1, dims=1)
    
    config = {
        'hspc_iters': 1,
        'hspc_relaxation': 'sequential',
        'hspc_optimizer': 'sgd'
    }
    loss = model.train_step(x, y, config=config)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))

def test_hspc_checkpoint():
    model = build_model("hspc", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu", use_fp16=False)
    path = "tests/unit/test_hspc_cp.pt"
    model.save_checkpoint(path, step=10)
    
    model2 = build_model("hspc", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu", use_fp16=False)
    step = model2.load_checkpoint(path)
    assert step == 10
    
    if torch.os.path.exists(path):
        torch.os.remove(path)
