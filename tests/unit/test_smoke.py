"""
Smoke Tests (v10.0)
Verifies the full model→data→train_step pipeline end-to-end
using a tiny synthetic dataset (no HuggingFace / network required).
"""
import pytest
import torch
from gnn_llm import build_model

VOCAB = 512
SEQ_LEN = 16
BATCH = 2


def synthetic_batch():
    """Fake one batch of tokenized text without a real data pipeline."""
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


# ── APTP Smoke ───────────────────────────────────────────────────────────────

class TestAPTPSmoke:
    def setup_method(self):
        self.model = build_model(
            "aptp", vocab_size=VOCAB, depth=2, d_model=64, device="cpu"
        )

    def test_init_no_crash(self):
        assert self.model is not None

    def test_single_train_step(self):
        x = synthetic_batch()
        y = torch.roll(x, -1, dims=1)
        loss = self.model.train_step(x, y, lr=1e-4)
        assert torch.is_tensor(loss), "Loss should be a tensor"
        assert not torch.isnan(loss), "Loss must not be NaN"
        assert not torch.isinf(loss), "Loss must not be Inf"
        assert loss.item() < 100, f"Loss suspiciously high: {loss.item():.4f}"

    def test_two_consecutive_steps_stable(self):
        """Loss should not explode over multiple steps."""
        x = synthetic_batch()
        y = torch.roll(x, -1, dims=1)
        losses = [self.model.train_step(x, y, lr=1e-4).item() for _ in range(3)]
        assert all(l < 200 for l in losses), f"Loss exploded: {losses}"

    def test_checkpoint_and_resume(self, tmp_path):
        x = synthetic_batch()
        y = torch.roll(x, -1, dims=1)
        self.model.train_step(x, y, lr=1e-4)
        path = str(tmp_path / "aptp_smoke.pt")
        self.model.save_checkpoint(path, step=1)

        model2 = build_model("aptp", vocab_size=VOCAB, depth=2, d_model=64, device="cpu")
        step = model2.load_checkpoint(path)
        assert step == 1


# ── NoProp Smoke ─────────────────────────────────────────────────────────────

class TestNoPropSmoke:
    def setup_method(self):
        self.model = build_model(
            "noprop", vocab_size=VOCAB, d_model=64, depth=2, device="cpu"
        )

    def test_init_no_crash(self):
        assert self.model is not None

    def test_single_train_step(self):
        x = synthetic_batch()
        y = torch.roll(x, -1, dims=1)
        loss = self.model.train_step(x, y)
        assert isinstance(loss, float), "NoProp loss should be a float (avg MSE)"
        assert loss == loss, "Loss must not be NaN"
        assert loss < 100, f"Loss suspiciously high: {loss:.4f}"

    def test_two_consecutive_steps_stable(self):
        x = synthetic_batch()
        y = torch.roll(x, -1, dims=1)
        losses = [self.model.train_step(x, y) for _ in range(3)]
        assert all(l < 200 for l in losses), f"Loss exploded: {losses}"

    def test_checkpoint_and_resume(self, tmp_path):
        path = str(tmp_path / "noprop_smoke.pt")
        self.model.save_checkpoint(path, step=5)
        model2 = build_model("noprop", vocab_size=VOCAB, d_model=64, depth=2, device="cpu")
        step = model2.load_checkpoint(path)
        assert step == 5


# ── build_model factory ───────────────────────────────────────────────────────

def test_build_model_unknown_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_model("fakealgo")
