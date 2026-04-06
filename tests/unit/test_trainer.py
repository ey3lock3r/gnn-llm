import pytest
import torch
import sys
sys.path.insert(0, '/home/ey3lock3r/Source/gnn-llm')

from gnn_llm import build_model

VOCAB, D_MODEL, DEPTH = 512, 64, 2

def fake_loader(n=10):
    for _ in range(n):
        yield torch.randint(0, VOCAB, (2, 16))

def test_aptp_loop_no_crash():
    model = build_model("aptp", vocab_size=VOCAB, depth=DEPTH, d_model=D_MODEL, device="cpu")
    for batch in fake_loader(5):
        y = torch.roll(batch, -1, dims=1)
        loss = model.train_step(batch, y, lr=1e-4)
        assert not torch.isnan(loss)

def test_noprop_loop_no_crash():
    model = build_model("noprop", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu", use_fp16=False)
    for batch in fake_loader(5):
        y = torch.roll(batch, -1, dims=1)
        loss = model.train_step(batch, y)
        assert isinstance(loss, float)
        assert loss < 100
