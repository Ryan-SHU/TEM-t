# TEM-t: Temporal Episodic Memory Transformer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Unofficial reproduction of:**

> James C.R. Whittington, Joseph Warren, Tim E.J. Behrens.
> *"Relating Transformers to Models and Neural Representations of the Hippocampal Formation."*
> arXiv:2112.04035, 2021.
> [[arXiv](https://arxiv.org/abs/2112.04035)]

This repository reproduces the TEM-t (Temporal Episodic Memory Transformer) model from the paper. No official implementation was found, so all code is built from the paper's mathematical descriptions and appendix.

---

## Overview

TEM-t bridges two historically separate frameworks for understanding hippocampal representations:

| Framework | This Paper's Correspondence |
|---|---|
| **Cognitive maps / Tolman-Eichenbaum machines (TEMs)** | Recurrent positional encoding via action-conditioned transitions `g_{t+1} = σ(g_t W_{a_t})` |
| **Transformers (self-attention)** | Softmax attention over episodic memory slots with positional Q/K and sensory V |

**Key insight:** When a Transformer is structurally constrained so that *queries and keys come from position representations* while *values come from sensory representations*, it becomes formally equivalent to a TEM. The model learns structured spatial representations — grid cells, place cells, and band cells — purely from predicting the next sensory observation during random-walk navigation.

### Core Equation

```
g_{t+1}^{PI} = σ(g_t W_{a_t})                          # path integration
p(x_{t+1})   = softmax[f_pred(Attn(ĝ_{t+1}^{PI}, K_t, V_t^x))]   # sensory prediction
```

where `Attn(q, K, V) = softmax(β · qK^T / √d_k) V`.

---

## Architecture

```mermaid
flowchart TD
    subgraph Input
        X["x_{t+1} (next sensory)"]
        A["a_t (action)"]
        G["g_t (current position)"]
        M["M_t (episodic memory)"]
    end

    subgraph predict_next["predict_next (no x_{t+1})"]
        PI["g_{t+1}^{PI} = σ(g_t W_{a_t})"]
        Q["q = project_position(g_{t+1}^{PI})"]
        ATT["attn = softmax(β · qK^T / √d_k)"]
        READ["r = α · V_x"]
        LOGITS["logits_pi = f_pred(r)"]
    end

    subgraph observe_next["observe_next (with x_{t+1})"]
        LAND["g_retrieved = landmark(x_{t+1}, M_t)"]
        FUSE["g_{t+1} = g_{t+1}^{PI} + η ⊙ (g_retrieved - g_{t+1}^{PI})"]
        WRITE["M_{t+1} = write(g_{t+1}, x_{t+1})"]
    end

    G --> PI
    A --> PI
    PI --> Q
    Q --> ATT
    M --> ATT
    ATT --> READ
    READ --> LOGITS
    X --> LAND
    M --> LAND
    PI --> FUSE
    LAND --> FUSE
    FUSE --> WRITE
    X --> WRITE
    G --> WRITE
```

### Two-Phase Online Protocol (No Information Leakage)

1. **`predict_next(g_t, memory_t, a_t)`** → `PredictionState` — predicts `x_{t+1}` **without accessing `x_{t+1}`**
2. **`observe_next(g_pi, memory_t, x_{t+1})`** → `ObservationState` — corrects position, writes memory

Zero-shot evaluation uses only `logits_pi` (from phase 1), ensuring no sensory leakage.

---

---

## What's NOT in This Repository

The following are excluded via `.gitignore` and must be set up locally:

| Excluded | Reason |
|---|---|
| `venv/` | Python virtual environment — recreate with `python3 -m venv venv` |
| `doc/` | Internal Chinese design documents; all info needed is in this README |
| `experiments/` | Training outputs (checkpoints, logs, figures) — generated at runtime |
| `__pycache__/`, `*.pyc` | Compiled Python bytecode |
| `.pytest_cache/` | Test runner cache |
| `.vscode/`, `.idea/` | IDE configuration |

---

## Project Structure

```
TEM-t/
├── model/                          # Core model components
│   ├── temt.py                     # Main TEM-t model (predict_next / observe_next / forward)
│   ├── recurrent_position.py       # g_{t+1} = σ(g_t W_{a_t})
│   ├── tem_attention.py            # FixedLayerNorm, Projectors, TEMAttention
│   ├── memory.py                   # Episodic memory with deduplication
│   ├── heads.py                    # SensoryPredictionHead, LandmarkStabilizer
│   └── baseline_tem.py             # TEM Hebbian baseline for comparison
│
├── training/                       # Training infrastructure
│   ├── batch.py                    # TrajectoryBatch dataclass
│   ├── envs.py                     # GridWorldSpec, EnvironmentInstance, TrajectorySampler
│   ├── losses.py                   # Composite loss (CE + consistency + L2)
│   ├── trainer.py                  # Training loop with checkpointing
│   ├── evaluator.py                # Standard + zero-shot evaluation
│   └── metrics.py                  # Rate maps, gridness, place scores, remapping
│
├── config/                         # YAML configuration files
│   ├── model_temt.yaml             # Model dimensions and toggles
│   ├── dataset_2d_grid.yaml        # Environment parameters
│   └── train.yaml                  # Training hyperparameters
│
├── script/                         # CLI entry points
│   ├── train_temt.py               # Train a TEM-t model
│   ├── eval_zero_shot.py           # Evaluate zero-shot prediction
│   ├── extract_rate_maps.py        # Extract spatial rate maps
│   └── plot_rate_maps.py           # Visualize rate maps
│
├── tests/                          # 73 unit and integration tests
│   ├── test_recurrent_position.py
│   ├── test_tem_attention.py
│   ├── test_memory.py
│   ├── test_heads.py
│   ├── test_temt.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_envs.py
│   ├── test_zero_shot.py
│   └── test_integration.py
│
├── experiments/                    # Experiment outputs (per-run directories)
└── doc/                            # Design documents (Chinese)
```

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.8+ (CPU-only also works for small-scale tests)
- PyTorch 2.0+, PyYAML, NumPy, Matplotlib (optional, for plots)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ryan-SHU/TEM-t.git
cd TEM-t

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install pyyaml numpy matplotlib pytest
```

### Run Tests

```bash
python -m pytest tests/ -v
```

Expected: **73 passed**.

---

## Experiments

This repository reproduces four experiments from the paper.

### Experiment 1: Entorhinal Representations (Grid Cells, Band Cells)

**Goal:** Demonstrate that TEM-t's recurrent positional encodings develop grid-like and band-like spatial tuning, matching empirical observations in medial entorhinal cortex.

**How it works:** Train TEM-t on a 2D grid world with random sensory assignments, then extract the spatial rate map of each hidden unit `g_i` and compute the gridness score via 2D autocorrelation.

**Run:**

```bash
# 1. Train the model
python script/train_temt.py \
  --model_config config/model_temt.yaml \
  --dataset_config config/dataset_2d_grid.yaml \
  --train_config config/train.yaml \
  --exp_dir experiments/exp01_entorhinal_representations/run001 \
  --seed 0

# 2. Extract rate maps
python script/extract_rate_maps.py \
  --ckpt experiments/exp01_entorhinal_representations/run001/checkpoints/latest.pt \
  --dataset_config config/dataset_2d_grid.yaml \
  --model_config config/model_temt.yaml \
  --output_dir experiments/exp01_entorhinal_representations/run001/rate_maps \
  --n_steps 50000

# 3. Plot
python script/plot_rate_maps.py \
  --rate_map_dir experiments/exp01_entorhinal_representations/run001/rate_maps \
  --output_dir experiments/exp01_entorhinal_representations/run001/figures
```

**Key configs:** Try `activation: identity` for grid-like patterns and `activation: relu` for mixed grid/band patterns.

---

### Experiment 2: Zero-Shot Sensory Prediction

**Goal:** Test whether the model learns abstract spatial structure rather than memorizing stimulus-action pairs. A *zero-shot* transition is one where the edge `(s_t, a_t)` has never been traversed, but the destination `s_{t+1}` has been visited — requiring structural inference.

**Run:**

```bash
python script/eval_zero_shot.py \
  --ckpt experiments/exp01_entorhinal_representations/run001/checkpoints/latest.pt \
  --dataset_config config/dataset_2d_grid.yaml \
  --model_config config/model_temt.yaml \
  --batch_size 64 \
  --n_batches 100
```

**Output:**
```json
{
  "acc_all": 0.82,
  "acc_zero_shot": 0.74,
  "n_zero_shot": 15320
}
```

A high `acc_zero_shot` (well above chance level `1/N_x`) indicates the model has learned the transition structure.

---

### Experiment 3: TEM-t vs TEM Sample Efficiency

**Goal:** Compare TEM-t (softmax attention memory) against the original TEM baseline (Hebbian conjunctive memory) on sample efficiency, training time, and zero-shot accuracy.

**Run:**

```bash
# TEM-t (same as Experiment 1 with extended training)
python script/train_temt.py \
  --model_config config/model_temt.yaml \
  --dataset_config config/dataset_2d_grid.yaml \
  --train_config config/train.yaml \
  --exp_dir experiments/exp02_sample_efficiency_vs_tem/temt_run001

# For TEM baseline, modify model config to use baseline_tem.py
# or write a separate comparison script using model/baseline_tem.py
```

**Expected result:** TEM-t reaches higher zero-shot accuracy with fewer gradient updates and similar per-step wall-clock time.

---

### Experiment 4: Memory Neurons as Place Cells

**Goal:** Show that attention memory slots behave like hippocampal place cells — each slot activates selectively in a restricted spatial region, and these place fields randomly remap across environments.

**How it works:** After training, attention weights `α_{t,j}` (memory neuron activations) are recorded during spatial navigation. The spatial rate map of each memory slot is computed, showing localized place-like firing fields.

**Run:** (Uses the same extraction and plotting pipeline as Experiment 1; memory neuron rate maps are saved as `rate_maps_memory.pt`.)

```bash
# After training, extract rate maps (saves both g-unit and memory-neuron maps)
python script/extract_rate_maps.py \
  --ckpt experiments/exp03_memory_place_cells/run001/checkpoints/latest.pt \
  --dataset_config config/dataset_2d_grid.yaml \
  --model_config config/model_temt.yaml \
  --output_dir experiments/exp03_memory_place_cells/run001/rate_maps \
  --n_steps 50000
```

---

## Key Design Constraints

The implementation enforces several constraints from the paper that are critical for reproducing the results:

1. **Q/K from position, V from sensory** — Attention queries and keys derive from positional encoding `g`; values derive from sensory observation `x`. Violating this reverts to a standard Transformer.
2. **Fixed layer norm on position only** — Applied before the attention projection; does **not** feed back into the recurrent dynamics.
3. **No sensory leakage** — `predict_next(g_t, memory_t, a_t)` must never access `x_{t+1}`. Zero-shot evaluation uses only `logits_pi`.
4. **Memory deduplication** — Prevents repeated writes at frequently visited positions, avoiding attention bias.
5. **Adaptive attention temperature** — `β = β₀ · log(m_t + 1)` sharpens attention as memory grows.

---

## Configuration

All hyperparameters are controlled via YAML files in `config/`:

| File | Purpose |
|---|---|
| `model_temt.yaml` | Model dimensions (`d_g`, `d_k`, `d_v`), activation, memory settings |
| `dataset_2d_grid.yaml` | Grid size, trajectory length, number of environments, boundary behavior |
| `train.yaml` | Batch size, learning rate, loss weights, logging intervals |

---

## Citation

If you use this code in your research, please cite both the original paper and this reproduction:

```bibtex
@article{whittington2021relating,
  title   = {Relating Transformers to Models and Neural Representations of the Hippocampal Formation},
  author  = {Whittington, James C.R. and Warren, Joseph and Behrens, Tim E.J.},
  journal = {arXiv preprint arXiv:2112.04035},
  year    = {2021}
}

@software{TEMt_reproduction,
  title   = {{TEM-t}: Unofficial Reproduction of TEM-t},
  author  = {},
  year    = {2026},
  url     = {https://github.com/Ryan-SHU/TEM-t}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
