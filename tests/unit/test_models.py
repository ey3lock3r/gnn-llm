import pytest
import torch
import sys
sys.path.insert(0, '/home/ey3lock3r/Source/gnn-llm')

from gnn_llm import build_model

VOCAB, D_MODEL, DEPTH = 512, 64, 2

def make_batch():
    return torch.randint(0, VOCAB, (2, 16))

def test_aptp_build():
    model = build_model("aptp", vocab_size=VOCAB, depth=DEPTH, d_model=D_MODEL, device="cpu")
    assert model is not None

def test_aptp_train_step():
    model = build_model("aptp", vocab_size=VOCAB, depth=DEPTH, d_model=D_MODEL, device="cpu")
    x = make_batch()
    y = torch.roll(x, -1, dims=1)
    loss = model.train_step(x, y, lr=1e-4)
    assert torch.is_tensor(loss)
    assert not torch.isnan(loss)
    assert loss.item() < 50

def test_noprop_build():
    model = build_model("noprop", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu")
    assert model is not None

def test_noprop_train_step():
    model = build_model("noprop", vocab_size=VOCAB, d_model=D_MODEL, depth=DEPTH, device="cpu")
    x = make_batch()
    y = torch.roll(x, -1, dims=1)
    loss = model.train_step(x, y)
    assert isinstance(loss, float)
    assert loss < 50

def test_checkpoint_roundtrip(tmp_path):
    model = build_model("aptp", vocab_size=VOCAB, depth=DEPTH, d_model=D_MODEL, device="cpu")
    path = str(tmp_path / "test_cp.pt")
    model.save_checkpoint(path, step=42)
    model2 = build_model("aptp", vocab_size=VOCAB, depth=DEPTH, d_model=D_MODEL, device="cpu")
    step = model2.load_checkpoint(path)
    assert step == 42
