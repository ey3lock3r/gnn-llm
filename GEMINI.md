# APTP-GNN: GigaGraph 3.2B → 8B (v10.0)

### 🚀 High-Level Objective
Building a **Biologically Plausible**, backprop-free Large Language Model (LLM) at scale (3.2B → 8B parameters).

---

### 🗂️ Project Structure (v10.0 Standard)
```
gnn-llm/
├── gnn_llm/                  # Main Python package (IMPORT FROM HERE)
│   ├── __init__.py           # build_model() factory + MODEL_REGISTRY
│   ├── models/
│   │   ├── base.py           # GigaModel ABC (all models must implement this)
│   │   ├── aptp.py           # APTP-GNN v8.12 (3.2B, Dual-T4 sharded)
│   │   └── noprop.py         # NoProp v9.0 (100M prototype, BP-free denoising)
│   ├── data/
│   │   └── pipeline.py       # GigaDataPipeline (FineWeb-Edu 80/20 stream)
│   └── training/
│       ├── trainer.py        # run_training() unified loop (algorithm-agnostic)
│       └── utils.py          # init_wandb(), checkpoint helpers
├── tests/
│   └── unit/
│       ├── test_models.py    # Build, forward, checkpoint round-trip
│       └── test_trainer.py   # Training loop regression
├── train.ipynb               # ✅ UNIFIED ENTRY POINT (Kaggle)
├── kernel-metadata.json      # Kaggle deployment config
├── legacy/                   # Archived old scripts (do not edit)
│   ├── aptp_gnn.py
│   ├── noprop_gnn.py
│   ├── data_pipeline.py
│   └── kaggle_training.ipynb
├── scripts/                  # One-off utility scripts
├── docs/                     # Research notes, analysis
├── pyproject.toml
└── GEMINI.md
```

### 🧬 Core Architecture
- **Algorithm:** **APTP** (Activity Prop), **NoProp** (Denoising), and **HS-PC** (Hybrid Stochastic Predictive Coding, 2026).
- **Scaling Phase:** **GigaGraph v11.5** (3.2B parameters, HS-PC).
  - `d_model=3072`, `depth=32`, `max_seq_len=4096`
  - **Learning Rule:** Iterative state relaxation + Stochastic shocks.
  - **Distributed Strategy:** Layer-Sharding (16 blocks on `cuda:0`, 16 blocks on `cuda:1`) + FP16 Mixed Precision.

---

### ⚙️ Workflow Standards (v10.0)

#### Adding a New Algorithm
1. Create `gnn_llm/models/myalgo.py` implementing `GigaModel` (inherit `base.py`).
2. Implement `train_step(x, y, **kwargs)`, `save_checkpoint(path, step)`, `load_checkpoint(path)`.
3. Register it in `gnn_llm/__init__.py` → `MODEL_REGISTRY`.
4. Add unit tests in `tests/unit/test_models.py`.
5. Select it in `train.ipynb` by setting `CONFIG['algorithm'] = 'myalgo'`.

#### Before Any Kaggle Push
```bash
# Always run tests first
uv run pytest tests/ -v
# Only push if all tests pass
uv run kaggle kernels push -p .
```

#### Switching Algorithms in train.ipynb
Change one line at the top of Cell 1:
```python
CONFIG['algorithm'] = 'aptp'    # 3.2B run
CONFIG['algorithm'] = 'noprop'  # 100M prototype
```

#### Optimization & Problem-Solving Principle
> **Rule:** Before proceeding with expensive computational overhead or heavy mathematical fixes, ALWAYS explore ways to fix or change things *without* the overhead. Only if there is absolutely no other mathematically sound way should we proceed with expensive overhead.


---

### 🗂️ Data & Pipeline
- **Dataset Mix:** 80% **FineWeb-Edu-10BT** + 20% Logical Reasoning
- **Tokenizer:** **Meta-Llama-3-8B** (128k Vocabulary)
- **Loader:** `GigaDataPipeline` in `gnn_llm/data/pipeline.py`

---

### 🔑 Environment & Secrets
- **HF_TOKEN:** Hugging Face token (Must have Llama-3 Gated access)
- **WANDB_API_KEY:** Weights & Biases API key
- **Kaggle Setup:** Internet: True, GPU: t4_x2, Secrets attached

---

### 🧪 Testing Protocol (MANDATORY)

> **Rule:** `uv run pytest` MUST be run after **every code change** — models, training logic, data pipeline, or config. ALL tests must pass before committing or pushing to Kaggle. No exceptions.

```bash
# Run full suite (always do this before any commit or Kaggle push)
uv run pytest -v --tb=short

# Run a specific test during development
uv run pytest tests/unit/test_models.py::test_aptp_train_step -v
```

**Test locations:**
- `tests/unit/test_models.py` — Build, forward pass, checkpoint round-trip for all models
- `tests/unit/test_trainer.py` — Training loop regression (no crash, no NaN)

**pytest config** (in `pyproject.toml`):
- Only collects from `tests/` — `legacy/` is excluded automatically.
- Run `uv run pytest` from the project root.

---

### 🔭 Roadmap
| Version | Algorithm | Parameters | Status |
|---------|-----------|-----------|--------|
| v8.12 | APTP-GNN | 3.2B | Archived (Plateau 11.9) |
| v10.1 | NoProp-Proto| 100M | Validated (Breakout) |
| v11.0 | NoProp-Scale| 3.2B | Archived (MSE Baseline) |
| v11.5 | HS-PC-Scale| 3.2B | ✅ Active |
| v12.0 | GigaScale | 8B | Planned |

---

### 🤖 AI Agent Implementation Guidelines (Lessons Learned)
For any AI agents operating on this repository in the future:
1. **Always `view_file` Before Replacements:** Do not rely on assumptions or memory of file contents when planning `multi_replace_file_content` operations. Minor discrepancies (like a moved `__init__` variable) will cause chunk failures. Always verify the exact target snippet first.
2. **Toggles > Duplicate Files (DRY):** When exploring new variants (e.g., Fourier Mixing or Stochastic Depth), DO NOT create redundant algorithm copies (like `stochastic_noprop.py`). Inject modular logic "Toggles" via `CONFIG` into the core blocks. This keeps the codebase highly dense, avoids rot, and allows experiments to be freely mixed!
3. **High-Density Memory Management (3.2B+):** For models exceeding 3GB per GPU on 16GB hardware, always perform "Aggressive Deletion" of activation tensors (e.g. `z_refined`) immediately after gradients are computed and BEFORE `optimizer.step()`. This prevents momentum allocation from triggering OOM.
