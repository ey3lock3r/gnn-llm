# APTP-GNN: Asynchronous PEPITA-TargetProp Graph Neural Network

A biologically plausible, backprop-free, hardware-asynchronous LLM architecture designed to compete with current SOTA models.

## 🚀 Getting Started with `uv`

We use **uv** for fast and deterministic package management.

### 1. Initialize and Setup
```bash
uv init
uv add torch numpy
```

### 2. Run Convergence Tests
```bash
uv run python test_aptp.py
```

### 3. Kaggle / GCP Deployment
Use the provided `kaggle_training.ipynb` which includes automated `uv` setup for Dual-T4 GPU environments.

## 🧬 Architecture: v7.0 (Expert-Refined)
- **Algorithm:** APTP (PEPITA-modulated Target Propagation).
- **Asynchrony:** 100% GPU utilization via Topological Interleaving.
- **Attention:** Entropy-Balanced Neighborhoods (EBA).

For full engineering details, see [docs/APTP-GNN-SPEC.md](docs/APTP-GNN-SPEC.md).
