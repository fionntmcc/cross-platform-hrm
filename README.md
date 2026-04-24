<!--
Project: Hierarchical Reasoning Model for Puzzle Solving
Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
Supervisor: Dr. John Healy
Institution: Atlantic Technological University
Duration: 2025/2026
-->

# Cross-Platform HRM — Hierarchical Reasoning Model for Puzzle Solving

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2018-005CED.svg)](https://onnx.ai/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](https://github.com/fionntmcc/cross-platform-hrm/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-ff6b35.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A cross-platform, CPU-deployable implementation of the **Hierarchical Reasoning Model (HRM)** — a compact iterative-reasoning neural architecture that learns to solve constraint-satisfaction puzzles (Sudoku, weighted mazes) through latent-space refinement. The project implements the *L-module-only* Simplified HRM variant proposed by Ge et al. (2025), trains it on synthetically generated puzzles, and deploys it to a Raspberry Pi 5 via the ONNX Runtime with zero PyTorch dependency at inference time.

This is the Final Year Project of **Kyrylo Kozlovskyi** (G00425385) and **Fionn McCarthy** (G00414386), supervised by **Dr. John Healy** at **Atlantic Technological University (ATU) Galway**, submitted as part of the B.Sc. (Hons) in Software Development.

> **Project title:** *Design and Implementation of a Hierarchical Reasoning Model (HRM) for Puzzle Solving*
> **Repository:** <https://github.com/fionntmcc/cross-platform-hrm>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Headline Results](#headline-results)
5. [Repository Layout](#repository-layout)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Dataset Generation](#dataset-generation)
9. [Training](#training)
10. [Evaluation, Demo and Visualisation](#evaluation-demo-and-visualisation)
11. [ONNX Export and Edge Inference](#onnx-export-and-edge-inference)
12. [Raspberry Pi 5 Deployment](#raspberry-pi-5-deployment)
13. [Multi-Seed Reproducibility](#multi-seed-reproducibility)
14. [Pre-Trained Model Registry](#pre-trained-model-registry)
15. [Testing](#testing)
16. [Continuous Integration](#continuous-integration)
17. [Development Workflow](#development-workflow)
18. [Command Reference](#command-reference)
19. [References](#references)
20. [Authors and Acknowledgements](#authors-and-acknowledgements)

---

## Overview

Chain-of-Thought (CoT) Large Language Models rely on fixed-depth Transformers that generate reasoning token-by-token in natural language. This approach is computationally expensive, prone to cascading failure (one wrong token derails the whole chain), and — according to the International Energy Agency — consumed roughly 1.5 % of global electricity in 2025. The **Hierarchical Reasoning Model (HRM)** proposed by Wang et al. (2025) takes a completely different route: it reasons in continuous latent space through coupled recurrent modules, achieving state-of-the-art results on constraint-satisfaction benchmarks (Sudoku-Extreme, Maze-Hard, ARC-AGI) with only 27 M parameters and 1 000 training examples per task.

A subsequent study by Ge et al. (2025) showed that the full hierarchical two-module (H + L) design is not strictly necessary — a single L-module iterated to convergence achieves comparable accuracy while training **2.4× faster**. This project implements that **Simplified HRM** and asks a practical question that the HRM literature has not yet answered:

> *Can an iterative-reasoning architecture like HRM be exported to ONNX and deployed on resource-constrained ARM edge hardware (Raspberry Pi 5) while preserving its reasoning capability?*

The answer, demonstrated by this repository, is **yes** — with caveats documented in the evaluation chapter of the accompanying dissertation.

---

## Key Features

- **Simplified HRM (L-module-only) implementation** — 6.9 M parameters, 8 weight-shared transformer blocks, hidden size 256, SwiGLU activation, RMSNorm, Rotary Position Embeddings (RoPE), 16 iterative reasoning steps.
- **Multi-task support** — the same model architecture trains and runs on three puzzle domains:
  - 4 × 4 Sudoku (16 tokens, vocab 5) — architecture validation prototype.
  - 9 × 9 Sudoku (81 tokens, vocab 10) — headline reasoning task.
  - Weighted mazes (grid-size² tokens, input vocab 10, binary path output) — pathfinding task.
- **Synthetic dataset generators** — unique-solution Sudoku generator with difficulty levels (easy / medium / hard / mixed) and a weighted-maze generator that uses Dijkstra's algorithm to label optimal paths over randomised edge weights.
- **Training pipeline** with mixed-precision (AMP), gradient accumulation, cosine-annealed learning rate, deep supervision, one-step gradient approximation, and structured JSON metric logging.
- **Cross-platform ONNX export pipeline** — the `SimplifiedHRMForONNX` wrapper resolves six incompatible PyTorch patterns (dynamic enum dispatch, Python control flow, FlashAttention conditionals, dynamic RoPE, enum-dependent embedding, output-head selection) into a fully static graph that passes 5/5 numerical-equivalence verification against the PyTorch reference on every export.
- **Zero-dependency edge inference** — `solve_onnx.py` runs the exported model on any platform with only `numpy` and `onnxruntime` installed. No PyTorch required on the deployment device.
- **Raspberry Pi 5 deployment** — verified working on Raspberry Pi OS Lite 64-bit (Debian Bookworm, aarch64). Inference latencies around 57 ms/puzzle for Sudoku and 81 ms/puzzle for 11 × 11 mazes at 4 reasoning steps.
- **Visualisation tooling** — terminal-based step-by-step reasoning replay, matplotlib publication-quality maze figures, and training-curve plots.
- **Continuous Integration** — GitHub Actions pipeline enforces Black formatting, Ruff linting, and pytest on every push and pull request. Pre-commit hooks mirror the same checks locally.
- **Multi-seed reproducibility** — `run_seeds.py` runs the same configuration under multiple random seeds and aggregates mean ± standard deviation statistics for dissertation tables.
- **Pre-trained model registry** — `publish_models.py` / `download_model.py` push checkpoints to GitHub Releases so collaborators can fetch the exact same weights.

---

## Architecture

The Simplified HRM (L-module only) consists of three components wired together:

```
  ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
  │  Input Embedding     │ ──▶  │  Reasoning Module    │ ──▶  │  Output Head         │
  │  token + position    │      │  8 × TransformerBlock│      │  linear + softmax    │
  │  (puzzle-type aware) │      │  weight-shared       │      │  (puzzle-type aware) │
  └──────────────────────┘      └──────────┬───────────┘      └──────────────────────┘
                                           │
                                           │  ×16 iterative refinement
                                           │  (self-conditioning: h_{t} fed as
                                           │   detached context for h_{t+1})
                                           ▼
```

Each `TransformerBlock` contains:

- **RMSNorm** pre-normalisation (identical to LLaMA).
- **Multi-head attention** with Rotary Position Embeddings (RoPE, pre-computed at export time).
- **SwiGLU** feed-forward network (3× hidden expansion, same gating as LLaMA 2).
- **Post-normalisation** before the residual connection.

The key departure from conventional feedforward Transformers is that **the same 8-block stack is applied 16 times** during a forward pass. The output of one iteration becomes the context for the next, enabling the effective computational depth of `8 × 16 = 128` layers while only storing 8 layers' worth of parameters. During training, the **one-step gradient approximation** from Wang et al. (2025) computes gradients only through the final iteration, keeping memory usage O(1) with respect to the number of iterations.

### Puzzle-Type Configuration

| Puzzle | Grid | Tokens | Input Vocab | Output Vocab | Notes |
|---|---|---|---|---|---|
| `sudoku_4x4` | 4 × 4 | 16 | 5 | 5 | 0 = empty; 1–4 = digits. Prototype for architecture validation. |
| `sudoku_9x9` | 9 × 9 | 81 | 10 | 10 | 0 = empty; 1–9 = digits. Headline task. |
| `maze` | N × N | N² | 10 | 2 | Input: 0 = wall, 1 = floor, 2 = start, 3 = goal. Output: binary path mask. |

---

## Headline Results

Full evaluation tables, multi-seed variance analysis, ablation studies, and training-curve plots are provided in the dissertation. The summary is:

| Task | Model | Training Set Puzzle Acc. | Token Acc. | Notes |
|---|---|---|---|---|
| Sudoku 4 × 4 (mixed) | 6.9 M Simplified HRM | ~ 59 % @ epoch 50 | — | Architecture validation. |
| Sudoku 9 × 9 (mixed) | 6.9 M Simplified HRM | ~ 61 % @ epoch 200 | ~ 87 % | 10 000 samples, RTX 3060 6 GB. |
| Weighted Maze 11 × 11 | 6.9 M Simplified HRM | ~ 87.5 % (train dist.) | — | Generalisation gap observed. |
| Weighted Maze 11 × 11 | 6.9 M Simplified HRM | ~ 8 % (unseen mazes) | — | See evaluation chapter for discussion. |

**ONNX export & CPU inference (x86, 4 reasoning steps):**

| Model | File Size | Verification | Mean Latency | Throughput |
|---|---|---|---|---|
| `simplified_hrm_sudoku_4x4.onnx` | 2.0 MB | 5/5 pass | 13 ms | 77 puzzles/s |
| `simplified_hrm_sudoku_9x9.onnx` | 2.0 MB | 5/5 pass | ~57 ms | ~17 puzzles/s |
| `simplified_hrm_maze_11x11.onnx` | 2.0 MB | 5/5 pass | ~81 ms | ~12 puzzles/s |

The Raspberry Pi 5 benchmark table is produced by running `scripts/solve_onnx.py --benchmark` on the device; exact numbers appear in the dissertation.

---

## Repository Layout

```
cross-platform-hrm/
├── hrm/                        # Core Python package
│   ├── __init__.py             # Top-level exports (SimplifiedHRM, PuzzleType, ...)
│   ├── model_simplified.py     # Simplified HRM (L-module only, 6.9M params)
│   ├── layers/                 # Neural network building blocks
│   │   ├── norm.py             # RMSNorm, RMSNormWithBias
│   │   ├── transformer.py      # Attention, TransformerBlock, SwiGLU, RoPE, ReasoningModule
│   │   ├── input_simplified.py # SimplifiedInputEmbedding (puzzle-aware)
│   │   ├── output_simplified.py# SimplifiedOutputHead (puzzle-aware)
│   │   └── __init__.py
│   ├── data/                   # Dataset generation
│   │   ├── sudoku_generator.py # Unique-solution Sudoku generator (4×4, 9×9)
│   │   ├── weighted_maze_generator.py  # Dijkstra-labelled weighted mazes
│   │   ├── validator.py        # Sudoku / maze solution validators
│   │   ├── io.py               # JSON/CSV/npz serialisation
│   │   └── generate_dataset.py # CLI entry point
│   └── training/               # Training utilities
│       ├── metrics.py          # MetricsTracker (loss, token acc, puzzle acc)
│       ├── logger.py           # TrainingLogger (JSON output)
│       └── seed_analysis.py    # Multi-seed aggregation
│   └── prototype/              # Archived full-HRM research code
├── scripts/                    # User-facing entry points
│   ├── train_simplified.py     # Main training script
│   ├── run_simplified.py       # Dataset evaluation
│   ├── demo_simplified.py      # Hardcoded + interactive demo
│   ├── visualise_simplified.py # Terminal step-by-step reasoning
│   ├── visualise_maze.py       # Matplotlib publication figures
│   ├── export_onnx.py          # PyTorch → ONNX export pipeline
│   ├── solve_onnx.py           # Zero-dependency ONNX inference
│   ├── run_seeds.py            # Multi-seed experiment runner
│   ├── download_model.py       # Fetch checkpoints from GitHub Releases
│   └── publish_models.py       # Upload checkpoints to GitHub Releases
├── test/                       # pytest test suite
│   ├── hrm_test/               # Active package tests
│   └── other/                  # Data and utility tests
├── .github/workflows/          # GitHub Actions
│   └── ci.yml                  # Black, Ruff, pytest matrix
├── model/                      # Saved checkpoints (.pt) and ONNX (.onnx)
├── data/                       # Generated datasets (.npz, .json)
├── figures/                    # Visual outputs
├── requirements.txt            # Core runtime dependencies
├── pyproject.toml              # Black, Ruff, pytest configuration
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .gitignore
├── LICENSE
└── README.md                   # You are here.
```

---

## Installation

The project targets **Python 3.10, 3.11, or 3.12**. Training requires a CUDA-capable GPU for reasonable epoch times (the reference runs use an NVIDIA RTX 3060 6 GB); inference via ONNX Runtime runs on any x86-64 or aarch64 CPU.

### Clone the repository

```bash
git clone https://github.com/fionntmcc/cross-platform-hrm.git
cd cross-platform-hrm
```

### Option A — Training/Development environment (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# For development (tests, linters, pre-commit):
pip install -r requirements-dev.txt
```

### Option B — Training/Development environment (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# For development:
pip install -r requirements-dev.txt
```

### Option C — CPU-only inference (any platform)

If all you want to do is run a pre-trained `.onnx` model, you do **not** need PyTorch:

```bash
python3 -m venv .venv
source .venv/bin/activate   # (or .\.venv\Scripts\Activate.ps1 on Windows)
pip install numpy onnxruntime
```

That is enough to run `scripts/solve_onnx.py` against any exported model.

### Option D — Raspberry Pi 5 (aarch64, Debian Bookworm)

Flash **Raspberry Pi OS Lite (64-bit)** using the Raspberry Pi Imager, enable SSH during flashing, then:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git
git clone https://github.com/fionntmcc/cross-platform-hrm.git
cd cross-platform-hrm
python3 -m venv ~/hrm-env
source ~/hrm-env/bin/activate
pip install --upgrade pip
pip install numpy onnxruntime
python scripts/download_model.py --puzzle sudoku_9x9   # fetches pre-exported .onnx
python scripts/solve_onnx.py --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --benchmark
```

---

## Quick Start

**Five-minute demo** — train a 4 × 4 Sudoku model (CPU or GPU) and watch it reason step-by-step:

```bash
# 1. Train (≈ 5 min on RTX 3060, ≈ 20 min on CPU)
python scripts/train_simplified.py \
    --puzzle sudoku_4x4 \
    --generate-data \
    --num-samples 5000 \
    --difficulty mixed \
    --epochs 100 \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --amp

# 2. Run the demo
python scripts/demo_simplified.py \
    --model model/simplified_hrm_sudoku_4x4_best.pt \
    --puzzle sudoku_4x4

# 3. Watch the iterative reasoning trajectory
python scripts/visualise_simplified.py \
    --model model/simplified_hrm_sudoku_4x4_best.pt \
    --puzzle sudoku_4x4
```

---

## Dataset Generation

All datasets are synthetic; no external downloads are required.

### Sudoku

```bash
# 9×9 with mixed difficulty — 10 000 puzzles to data/sudoku_9x9_train.npz
python -m hrm.data.generate_dataset \
    --puzzle sudoku_9x9 \
    --num-samples 10000 \
    --difficulty mixed \
    --seed 42 \
    --output data/sudoku_9x9_train

# 4×4 easy only — 5 000 puzzles
python -m hrm.data.generate_dataset \
    --puzzle sudoku_4x4 \
    --num-samples 5000 \
    --difficulty easy \
    --output data/sudoku_4x4_train
```

Difficulty controls the number of cells removed from the fully solved board (easy ≈ 30 empty cells on 9 × 9, hard ≈ 50). Every generated puzzle is validated to have a *unique* solution via constraint propagation + backtracking.

### Weighted Maze

```bash
# 5 000 mazes of size 11×11 with random edge weights
python -m hrm.data.generate_dataset \
    --puzzle maze \
    --num-samples 5000 \
    --maze-size 11 \
    --seed 42 \
    --output data/maze_11x11_train
```

Each maze is paired with its shortest weighted path, computed using Dijkstra's algorithm on the randomised grid. The path is encoded as a binary mask over the grid — the model learns to predict which cells belong to the optimal path.

You can also let `train_simplified.py` generate the data inline by passing `--generate-data --num-samples N`.

---

## Training

The primary training script is `scripts/train_simplified.py`. It handles data loading (from `.npz` or inline generation), model initialisation, optimiser setup, the training loop with deep supervision, validation, checkpointing, and metric logging.

### Recommended commands

```bash
# 9×9 Sudoku — best-performing configuration (RTX 3060 6 GB, ≈ 4 h for 200 epochs)
python scripts/train_simplified.py \
    --puzzle sudoku_9x9 \
    --generate-data \
    --num-samples 10000 \
    --difficulty mixed \
    --epochs 200 \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --amp

# 4×4 Sudoku — prototype (≈ 5 min on GPU)
python scripts/train_simplified.py \
    --puzzle sudoku_4x4 \
    --generate-data \
    --num-samples 5000 \
    --difficulty mixed \
    --epochs 100 \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --amp

# Weighted Maze 15×15 (≈ 15 min on GPU)
python scripts/train_simplified.py \
    --puzzle maze \
    --generate-data \
    --num-samples 5000 \
    --maze-size 15 \
    --epochs 200 \
    --batch-size 128 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --amp

# Train from a pre-generated dataset
python scripts/train_simplified.py \
    --puzzle sudoku_9x9 \
    --data-path data/sudoku_9x9_train.npz \
    --epochs 500 \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --amp
```

### Key flags

| Flag | Purpose |
|---|---|
| `--puzzle {sudoku_4x4, sudoku_9x9, maze}` | Selects the task. |
| `--generate-data` | Generate the training set in-process instead of loading `.npz`. |
| `--data-path PATH` | Load a pre-generated `.npz` dataset. |
| `--num-samples N` | Training-set size (used with `--generate-data`). |
| `--difficulty {easy, medium, hard, mixed}` | Sudoku puzzle difficulty. |
| `--maze-size N` | Grid size for maze task (odd numbers recommended). |
| `--epochs N` | Number of training epochs. |
| `--batch-size N` | Per-step batch size. Reduce if you see CUDA out-of-memory. |
| `--lr RATE` | Initial Adam learning rate (cosine-annealed to 1e-6). |
| `--weight-decay D` | AdamW weight decay (default `0.1` matches the paper). |
| `--amp` | Enable PyTorch mixed-precision autocast (recommended on GPU). |
| `--reasoning-steps N` | Inner iteration count at training time (default `16`). |
| `--seed N` | Random seed for reproducibility. |

### What gets saved

Running a training job writes three artefacts into `model/`:

- `simplified_hrm_<puzzle>_best.pt` — the checkpoint with the highest validation puzzle-accuracy.
- `simplified_hrm_<puzzle>_final.pt` — the checkpoint at the last epoch.
- `training_history_simplified_<puzzle>.json` — structured per-epoch metrics (loss, token accuracy, puzzle accuracy, learning rate, wall time).

Use the `_best.pt` checkpoint for evaluation and ONNX export — with long training runs the final checkpoint may have overfit past the validation peak.

### Training on free cloud GPUs

If you do not have a local CUDA GPU, two Jupyter notebooks are provided under `notebooks/`:

- `simplified_hrm_kaggle_training.ipynb` — Kaggle Notebooks (T4, 15 GB VRAM, 30 GPU-hours/week free).
- `simplified_hrm_colab_training.ipynb` — Google Colab (T4, 12-hour sessions, no verification needed).

Both clone the repository, verify the GPU, train, plot training curves, and package the checkpoint for download.

---

## Evaluation, Demo and Visualisation

Three user-facing scripts consume trained `.pt` checkpoints. None of them modify the model — they are read-only evaluators.

### Dataset evaluation — accuracy statistics

```bash
python scripts/run_simplified.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 \
    --data data/sudoku_9x9_train.npz
```

Prints token accuracy, puzzle accuracy, and a handful of colour-coded worked examples showing the input puzzle, the model prediction, and the ground-truth solution side by side (correct cells in green, incorrect in red).

### Demo mode — hardcoded and interactive puzzles

```bash
# Demo on built-in example puzzles
python scripts/demo_simplified.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9

# Type in your own puzzle
python scripts/demo_simplified.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 \
    --interactive

# Maze demo with fresh generation
python scripts/demo_simplified.py \
    --puzzle maze \
    --model model/simplified_hrm_maze_best.pt
```

### Step-by-step reasoning visualisation — terminal replay

```bash
python scripts/visualise_simplified.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 \
    --delay 0.3
```

Replays each of the 16 internal reasoning steps with a configurable delay, showing how the latent-state prediction evolves from step 1 (essentially random) through convergence.

### Publication-quality maze figures

```bash
python scripts/visualise_maze.py \
    --model model/simplified_hrm_maze_best.pt \
    --data data/maze_11x11_train.npz \
    --index 42 \
    --output figures/maze_trajectory_42.pdf
```

Renders a matplotlib figure showing the input maze, the ground-truth optimal path, and the model's predicted path overlaid on the grid. Suitable for LaTeX inclusion.

---

## ONNX Export and Edge Inference

### Exporting a trained checkpoint

The export pipeline converts a PyTorch checkpoint into a static ONNX graph. The `SimplifiedHRMForONNX` wrapper (in `scripts/export_onnx.py`) handles six ONNX-incompatible patterns in the PyTorch model:

1. Baking the `PuzzleType` enum at construction time (no runtime enum dispatch).
2. Unrolling the 16-step reasoning loop as a Python `int` rather than a tracked tensor loop.
3. Forcing standard attention (no FlashAttention branch).
4. Pre-computing RoPE cosine/sine tables as registered buffers.
5. Inlining `InputEmbedding` to avoid enum-dependent forward paths.
6. Selecting the correct `OutputHead` at construction time instead of at forward time.

```bash
# Export the 9×9 Sudoku model (default 4 reasoning steps for fast inference)
python scripts/export_onnx.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9

# Export a maze model (grid size must match training)
python scripts/export_onnx.py \
    --model model/simplified_hrm_maze_best.pt \
    --puzzle maze \
    --maze-size 11

# Multiple reasoning-step variants for latency comparison
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 --reasoning-steps 16 \
    --output model/simplified_hrm_9x9_s16.onnx
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 --reasoning-steps 8 \
    --output model/simplified_hrm_9x9_s8.onnx
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 --reasoning-steps 4 \
    --output model/simplified_hrm_9x9_s4.onnx
```

Every export is followed by an **automatic verification step** that runs five random inputs through both PyTorch and ONNX Runtime and asserts exact logit-level agreement (`5/5 pass`). If verification fails the `.onnx` file is discarded.

**Note:** the default ONNX opset is **18** (not 17) because PyTorch's dynamo-based exporter requires at least opset 18. Older opsets will fail with a down-conversion error.

### Running inference with `solve_onnx.py`

`solve_onnx.py` has **zero PyTorch dependency** — only `numpy` and `onnxruntime`. It supports four modes:

```bash
# Demo mode — solves randomly generated puzzles with colour-coded output
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9

# Dataset evaluation
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9 \
    --data data/sudoku_9x9_train.npz

# Latency benchmark (mean / std / min / max / throughput)
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9 \
    --benchmark

# Interactive mode — type in puzzles as comma-separated integers
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9 \
    --interactive
```

For mazes, the demo mode displays the input grid, the optimal ground-truth path, and the model's predicted path overlaid, plus path-precision / path-recall / path-F1 metrics.

---

## Raspberry Pi 5 Deployment

The complete edge-deployment workflow:

```bash
# ── On your dev machine ──────────────────────────────────────────────
# Export the trained model to ONNX
python scripts/export_onnx.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 \
    --reasoning-steps 4

# Transfer only the .onnx file to the Pi (≈ 2 MB)
scp model/simplified_hrm_sudoku_9x9.onnx pi@<pi-ip>:~/cross-platform-hrm/model/

# ── On the Raspberry Pi 5 ────────────────────────────────────────────
ssh pi@<pi-ip>
cd ~/cross-platform-hrm
source ~/hrm-env/bin/activate
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9 \
    --benchmark
```

Typical Pi 5 benchmarks (4 reasoning steps, 8 GB model, Raspberry Pi OS Lite 64-bit):

| Model | Mean latency | Throughput |
|---|---|---|
| Sudoku 9 × 9 | ~ X ms | ~ Y puzzles/s |
| Maze 11 × 11 | ~ X ms | ~ Y puzzles/s |

Exact figures appear in Chapter 5 (Evaluation) of the dissertation.

---

## Multi-Seed Reproducibility

To report honest mean ± standard-deviation results, training runs are repeated across multiple random seeds. The `run_seeds.py` script orchestrates this:

```bash
# Run the same configuration under seeds 0, 1, 2
python scripts/run_seeds.py \
    --puzzle sudoku_9x9 \
    --seeds 0 1 2 \
    --epochs 200 \
    --batch-size 512 \
    --num-samples 10000

# Aggregate the results
python -m hrm.training.seed_analysis \
    --log-dir results/seed_runs \
    --output results/seed_summary.json
```

This produces a JSON summary suitable for dissertation tables (per-seed final accuracy, cross-seed mean / std / min / max, convergence epoch).

---

## Pre-Trained Model Registry

To avoid committing multi-megabyte checkpoints to Git, trained models are hosted on **GitHub Releases**. Two helper scripts manage the registry.

### Fetching a pre-trained model

```bash
# Download the latest 9×9 Sudoku PyTorch checkpoint
python scripts/download_model.py --puzzle sudoku_9x9

# Download a specific tag
python scripts/download_model.py --puzzle sudoku_9x9 --tag v1.0.0

# Download the matching ONNX file
python scripts/download_model.py --puzzle sudoku_9x9 --format onnx
```

Models are placed in `model/` and are ready to use immediately.

### Publishing a new checkpoint (maintainers only)

```bash
# Tag + push + upload the best checkpoint and its ONNX export
python scripts/publish_models.py \
    --puzzle sudoku_9x9 \
    --tag v1.1.0 \
    --notes "Improved training run — 61% puzzle accuracy" \
    --checkpoint model/simplified_hrm_sudoku_9x9_best.pt \
    --onnx model/simplified_hrm_sudoku_9x9.onnx
```

Requires a `GITHUB_TOKEN` environment variable with `repo` scope.

---

## Testing

The test suite uses **pytest**. All tests live under `test/` and cover the core package, data generators, and layer primitives.

```bash
# Run the full suite
pytest

# Run with coverage
pytest --cov=hrm --cov-report=term-missing

# Parallel execution (requires pytest-xdist)
pytest -n auto

# Single test file
pytest test/other/test_weighted_maze_generator.py -v

# Single test
pytest test/hrm_test/test_simplified_hrm.py -v
```

New contributions must include tests for any non-trivial changes and must not lower overall coverage.

---

## Continuous Integration

Every push and pull request triggers the `.github/workflows/ci.yml` pipeline, which runs on a Python 3.10 / 3.11 / 3.12 matrix:

1. **Black** — enforces consistent formatting (line length 100).
2. **Ruff** — static linter (pycodestyle, pyflakes, isort, bugbear rules).
3. **pytest** — runs the full test suite against installed dependencies.
4. **Architecture smoke test** — instantiates the model, runs a forward pass, checks output shapes.

A pull request cannot be merged until all three checks pass and at least one human reviewer approves. GitHub Copilot is configured to automatically review every pull request for additional feedback on code quality.

---

## Development Workflow

### Set up pre-commit hooks

Pre-commit runs the same checks as CI, locally, before every commit:

```bash
pip install pre-commit
pre-commit install
# Run once against the whole tree to clean up existing files
pre-commit run --all-files
```

### Formatting and linting manually

```bash
# Format everything
black .

# Lint and auto-fix what Ruff can fix
ruff check --fix .

# Check formatting without modifying files (matches CI)
black --check .
ruff check .
```

### Branch-per-issue workflow

The project follows a Kanban board on GitHub Projects with work tracked as issues tagged `WP1`–`WP6`. For every issue:

1. Create a branch off `main` named `<issue-number>-<short-description>`.
2. Commit in logical atomic chunks with messages prefixed `[WP<n>]`.
3. Open a pull request referencing the issue.
4. Address review comments from the other team member and from Copilot.
5. Merge via squash-and-merge once CI is green.

### Project management

- **Kanban board:** tracks work-package progress.
- **Issues:** one per unit of work, labelled by work package.
- **Pull requests:** squash-merged to keep `main` linear.
- **Releases:** tagged at dissertation milestones (`v0.1-prototype`, `v1.0-dissertation`).

---

## Command Reference

A consolidated cheat sheet of the most commonly used commands:

### End-to-end pipelines

```bash
# ── Sudoku 9×9: full pipeline ──
python scripts/train_simplified.py --puzzle sudoku_9x9 --generate-data --num-samples 10000 --difficulty mixed --epochs 200 --batch-size 512 --lr 3e-4 --weight-decay 0.1 --amp
python scripts/run_simplified.py --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9 --data data/sudoku_9x9_train.npz
python scripts/demo_simplified.py --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9
python scripts/visualise_simplified.py --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9
python scripts/solve_onnx.py --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --benchmark

# ── Sudoku 4×4: full pipeline ──
python scripts/train_simplified.py --puzzle sudoku_4x4 --generate-data --num-samples 5000 --difficulty mixed --epochs 100 --batch-size 512 --lr 3e-4 --weight-decay 0.1 --amp
python scripts/run_simplified.py --model model/simplified_hrm_sudoku_4x4_best.pt --puzzle sudoku_4x4 --data data/sudoku_4x4_train.npz
python scripts/demo_simplified.py --model model/simplified_hrm_sudoku_4x4_best.pt --puzzle sudoku_4x4
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_4x4_best.pt --puzzle sudoku_4x4

# ── Weighted maze 15×15: full pipeline ──
python scripts/train_simplified.py --puzzle maze --generate-data --num-samples 5000 --maze-size 15 --epochs 200 --batch-size 128 --lr 3e-4 --weight-decay 0.1 --amp
python scripts/run_simplified.py --model model/simplified_hrm_maze_best.pt --puzzle maze --data data/maze_15x15_train.npz
python scripts/visualise_maze.py --model model/simplified_hrm_maze_best.pt --data data/maze_15x15_train.npz --index 0
python scripts/export_onnx.py --model model/simplified_hrm_maze_best.pt --puzzle maze --maze-size 15
python scripts/solve_onnx.py --model model/simplified_hrm_maze_15x15.onnx --puzzle maze --benchmark
```

### Common troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` during training | Batch size too large for GPU VRAM | Halve `--batch-size` or remove `--amp` if already small |
| `ONNX export fails with opset 17` | Using modern PyTorch dynamo exporter | Use default opset 18 (don't pass `--opset`) |
| `RuntimeError: No such file or directory: 'model/...pt'` | Training did not complete or checkpoint not downloaded | Either retrain or run `scripts/download_model.py` |
| Slow Pi 5 inference (> 500 ms/puzzle) | Running full 16-step model | Export with `--reasoning-steps 4` |
| `ImportError: cannot import name 'SimplifiedHRM'` | Stale install | Reinstall: `pip install -e .` or reactivate venv |
| Black / Ruff failing in pre-commit | Code not formatted | Run `black . && ruff check --fix .` |

---

## References

The model and training procedure are based on the following publications:

- **Wang, G. et al. (2025).** *Hierarchical Reasoning Model.* Sapient Intelligence Technical Report. The original HRM paper introducing the two-module H + L architecture.
- **Ge, B. et al. (2025).** *Critical Analysis of the Hierarchical Reasoning Model.* arXiv:2510.00355v2. Shows that the L-module-only variant achieves comparable accuracy while training 2.4× faster — the basis for this project's Simplified HRM implementation.
- **Pochelu, P. (2022).** *Deep Learning Inference Frameworks Benchmark.* arXiv:2210.04323. Cross-platform deployment and ONNX Runtime performance at small batch sizes.

Full citations, plus references to supporting work on RoPE, RMSNorm, SwiGLU, Adaptive Computation Time, and the Deep Equilibrium Model, are listed in the dissertation bibliography.

---

## Authors and Acknowledgements

**Kyrylo Kozlovskyi** (G00425385) — L-module-only Simplified HRM implementation, GitHub Actions CI/CD pipeline, ONNX export pipeline (`scripts/export_onnx.py`), zero-dependency inference script (`scripts/solve_onnx.py`), Raspberry Pi 5 deployment, model registry (`download_model.py`, `publish_models.py`), evaluation and conclusion chapters of the dissertation.

**Fionn McCarthy** (G00414386) — Full HRM implementation, Weighted Maze generator, training pipeline design, metrics and logger modules, multi-seed analysis framework, Sudoku generators, introduction and methodology chapters of the dissertation.

**Supervisor:** Dr. John Healy — Atlantic Technological University Galway.

**Institution:** B.Sc. (Hons) in Software Development, Department of Computer Science & Applied Physics, Atlantic Technological University, Galway, Ireland.

We also acknowledge Sapient Intelligence for releasing the original HRM reference implementation and Ge et al. for the critical analysis that motivated the simplified architecture used here.

---

## License

Released under the [MIT License](LICENSE). The model architecture draws on ideas from Wang et al. (2025) and Ge et al. (2025); please cite those works in addition to this repository if you build on this project academically.

---

*Repository:* <https://github.com/fionntmcc/cross-platform-hrm>
