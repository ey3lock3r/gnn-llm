# Engineering Requirements Document: APTP-GNN v7.0 (Expert-Refined Final)

## 1. Overview
The **APTP-GNN v7.0** is the definitive specification for a biologically plausible, 100% asynchronous Graph Neural Network architecture for LLM reasoning. This version incorporates expert feedback to stabilize "Death Valley" scaling risks including over-smoothing and asynchronous semantic drift.

## 2. Core Architecture: The "Lattice-Gated" Topology
- **Blocks:** $64 \times 64$ sparse weight matrices ($\theta$) with **Topological Interleaving**.
- **Dual-T4 Interleaving:** 
  - Graph is split by Depth (Layers $0, 2, 4 \dots$ on GPU 0; Layers $1, 3, 5 \dots$ on GPU 1).
  - Activations and PEPITA residuals are transmitted asynchronously via a **Non-Blocking Warp Switch**.
- **Latent Dimension ($d$):** 3072.

## 3. Attention: Entropy-Balanced Neighborhoods (EBA)
Unlike v6.0's simple Hebbian gating, v7.0 uses **Localized Entropy Normalization**.
1. **Hebbian Alignment:** $\hat{\alpha}_{ij} = \text{Norm}(h_i^T \cdot h_j)$.
2. **Competitive Selection:** Within each $64 \times 64$ cluster, a local "Competitive Inhibitor" $\phi$ is applied:
   $$\alpha_{ij} = \frac{\exp(\hat{\alpha}_{ij} / \tau)}{\sum_{k \in \mathcal{N}(i)} \exp(\hat{\alpha}_{ik} / \tau)}$$
   *Note: This is a purely local Softmax, calculated without global autograd locks.*

## 4. Learning Algorithm: APTP (With Dynamic Residual Modulation)
The training process uses a two-pass forward-only protocol with a **Modulation Buffer**.

### Step 1: Forward Pass ($P_1$)
Nodes compute state updates using the EBA-Gated Message Passing:
$$h_i^{(l+1)} = \text{LN}\left( \text{GeLU}\left( \sum_{j} \alpha_{ij} W_{ij} h_j^{(l)} \right) + h_i^{(l)} \right)$$

### Step 2: Adaptive Error Generation
At the global output, error $E$ is calculated. Instead of a fixed $R$, we apply **Dynamic Residual Modulation (DRM)**:
$$e_l = R_l \cdot \text{Tanh}(E \cdot \Gamma_l)$$
Where $\Gamma_l$ is a layer-wise stability coefficient that prevents signal saturation at depth.

### Step 3: Modulated Forward Pass ($P_2$) & Local Update
Weights are updated directly using the **Zero-Lock Plasticity Rule**:
$$\Delta W_{ij} = \eta \cdot (h_i[P_2] - h_i[P_1]) \cdot h_j[P_1]^T - \lambda \cdot W_{ij}$$
*(The $-\lambda \cdot W_{ij}$ term is a weight-decay factor inspired by Synaptic Scaling in biological neurons).*

## 5. Scaling and Stability
To address **Over-smoothing** at the 1B-3B parameter scale:
- **Graph Re-Seeding:** Every $N$ layers, a portion of the original input embedding $E_{in}$ is re-injected as a residual skip-connection to anchor semantic meaning.
- **Micro-Batch Contrastive Variance:** Local blocks are penalized if their activations across a batch become constant ($Var(h) < \epsilon$), forcing the model to distinguish between different contexts.

## 6. Hardware Implementation (Kaggle Dual-T4)
- **VRAM Sharding:** 16GB per GPU. Maximum model depth is constrained by **Activation Checkpointing Lite** (storing only $P_1$ activations for the update step).
- **Communication:** Uses a background CUDA stream for Layer-Swap. Forward pass on Layer $L$ happens while Layer $L-1$ results are being moved across the PCIe bus to the other GPU.
