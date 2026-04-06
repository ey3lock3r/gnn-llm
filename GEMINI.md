# APTP-GNN: GigaGraph 3.2B Context (v8.1.1)

### 🚀 High-Level Objective
Building a **Biologically Plausible**, backprop-free Large Language Model (LLM) at scale (3.2B -> 8B parameters).

### 🧬 Core Architecture
- **Algorithm:** **APTP (Asynchronous PEPITA-TargetProp)**. A two-pass forward-only update rule for GNNs that eliminates backpropagation.
- **Scaling Phase:** **GigaGraph v8.1.1** (3.2B parameters).
  - `d_model`: 3072.
  - `depth`: 32 blocks.
  - `max_seq_len`: 4096.
  - **EBA Attention:** Entropy-Balanced Attention (competitive inhibition).
- **Distributed Strategy:** **Layer-Sharding** (16 blocks on `cuda:0`, 16 blocks on `cuda:1`). 

### 🗂️ Data & Pipeline
- **Dataset Mix:** 80% **FineWeb-Edu-10BT** (Clean CommonCrawl) + 20% Logical Reasoning (Coding).
- **Tokenizer:** **Meta-Llama-3-8B** (128k Vocabulary).
- **Loader:** Streaming interleaved datasets using Hugging Face `datasets` library.

### 🛠️ Directory Structure
- `aptp_gnn.py`: Core architecture & local update rules.
- `data_pipeline.py`: Llama-3 + FineWeb mixed streaming loader.
- `kaggle_training.ipynb`: Self-extracting training bundle for Kaggle Dual T4.
- `smoke_test.py`: Critical verification script for Llama-3 and distributed sharding.
- `run_tests.sh`: Full test suite (Unit & Integration).

### 🔑 Environment & Secrets
- **HF_TOKEN:** Hugging Face token (Must have "Gated Models" access for Llama-3).
- **WANDB_API_KEY:** Weights & Biases API key.
- **Kaggle Setup:** Internet access: **True**, GPU access: **True**, Secrets attached: **Required**.

### 📉 Convergence Metrics
- **GPT-2 Baseline (50k Vocab):** Initial Loss ~10.8.
- **Llama-3 Baseline (128k Vocab):** Initial Loss ~11.76.
- **Target:** Perplexity (PPL) decline from ~120,000 to <50.

### 🔭 Next Scaling Stage: GigaGraph 8B
Current architecture (v8.1.1) is designed with 8B-compatibility. Shifting to 8B requires `d_model=4096, depth=32, num_heads=32` and multi-node sharding.
