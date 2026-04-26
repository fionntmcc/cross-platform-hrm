# Simplified Cross-Platform HRM for Puzzle-Solving

[![Python 3.10+](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2018-005CED.svg)](https://onnx.ai/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](https://github.com/fionntmcc/cross-platform-hrm/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of the **Hierarchical Reasoning Model (HRM)**, a recurrent architecture that solves constraint-satisfaction puzzles by iteratively refining a hidden state rather than producing chain-of-thought text. The original HRM paper [[1]](#references) uses two coupled recurrent modules (a slow planner and a fast worker); a follow-up analysis by Ge, Liao and Poggio [[2]](#references) argues that the planner isn't actually doing much, and that a single L-module iterated to convergence reaches comparable accuracy while training ~2.4x faster. This project implements that simplified L-only variant, trains it on Sudoku and weighted mazes, and deploys the result to a Raspberry Pi 5 via ONNX Runtime.

This is our final year project for the B.Sc. (Hons) Computing in Software Development at **Atlantic Technological University**, supervised by **Dr. John Healy**.

- **Authors:** Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)

---

## Deliverables

| | |
|---|---|
| Source code | (`.github/workflows/`, `docs/`, `hrm/`, `scripts/`, `test/`) |
| Dissertation — Kyrylo Kozlovskyi | [`docs/KyryloKozlovskyi_G00425385_Dissertation.pdf`](docs/KyryloKozlovskyi_G00425385_Dissertation.pdf) |
| Dissertation — Fionn McCarthy | [`docs/FionnMcCarthy_G00414386_Dissertation.pdf`](docs/FionnMcCarthy_G00414386_Dissertation.pdf) |
| Screencast | [`docs/screencast_link.txt`](docs/screencast_link.txt) |
| Poster | [`docs/poster.pdf`](docs/poster.pdf) |
| Command reference | [`docs/commands.md`](docs/commands.md) |
| Pre-trained models and datasets | [GitHub Releases](https://github.com/fionntmcc/cross-platform-hrm/releases) |

---

## What it does

The model is a 6.9 M parameter transformer stack (8 blocks, hidden size 256, SwiGLU MLPs, RMSNorm, rotary position embeddings) that gets applied to the input 16 times in a row. The same weights are reused every step, so the effective computational depth is 128 layers for only 8 layers' worth of parameters. Gradients flow only through the last step (the "one-step approximation" from the Wang paper [[1]](#references)), which keeps memory constant regardless of how many reasoning steps you use.

One architecture handles three tasks:

- **4×4 Sudoku** — tiny, fast to train, used for sanity-checking the architecture.
- **9×9 Sudoku** — the headline task.
- **Weighted mazes** — a new benchmark we built that labels shortest paths over randomised edge weights using Dijkstra algorithm.

### Features

Beyond the model itself, the project includes:

- **Two generators** — unique-solution Sudoku (easy / medium / hard / mixed, difficulty is defined by number of backtracks required) and the weighted-maze generator.
- **Training pipeline** with mixed-precision, cosine LR schedule, deep supervision, and structured JSONL / CSV / TensorBoard logging.
- **ONNX export** that produces a static graph the dynamo exporter is happy with. The wrapper (`SimplifiedHRMForONNX`) hard-codes the puzzle type at construction, unrolls the 16-step loop into Python, swaps FlashAttention for plain attention, and pre-computes the RoPE tables. Every export runs a 5-input check against the PyTorch reference before the `.onnx` file is written.
- **`solve_onnx.py`** — pure numpy + onnxruntime, no PyTorch, no CUDA. Runs on anything including a Raspberry Pi.
- **Multi-seed runner** that trains the same config under several seeds and aggregates mean / std for dissertation tables.
- **Step-by-step visualisation** that replays the 16 reasoning iterations in the terminal so you can watch the grid converge.
- **Maze figures** via matplotlib, with predicted vs ground-truth path overlays.
- **GitHub Actions CI** (Black, Ruff, pytest on Python 3.10–3.13) and a release workflow for publishing trained checkpoints.
- **Model release scripts** for publishing to and downloading from GitHub Releases.

---

## Results

---

## Setup

Needs Python 3.10, 3.11, 3.12, or 3.13. Training is much faster with a CUDA GPU. Our reference runs used an NVIDIA GPU, but everything in the repo also runs on CPU. Run all commands from the repository root.

### General setup

```bash
git clone https://github.com/fionntmcc/cross-platform-hrm.git
cd cross-platform-hrm

python -m venv .venv
source .venv/bin/activate              # Linux / macOS
# .\.venv\Scripts\Activate.ps1         # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

This installs the runtime requirements: PyTorch, numpy, ONNX export dependencies, and the `hrm` package in editable mode.

### Dev environment setup

For training, tests, linting, TensorBoard logging, and figure generation, install the dev extras:

```bash
pip install -e ".[dev]"
pre-commit install
```

That adds:

- `pytest` and `pytest-cov` for tests
- `black` and `ruff` for formatting and linting
- `matplotlib` for maze figures
- `tensorboard` for training logs
- `pre-commit` hooks matching CI

Useful validation commands after setup:

```bash
pytest
ruff check hrm test
black --check hrm test/hrm_test test/other
```

### ONNX-only inference setup (Raspberry Pi 5 / edge devices)

If you only want to run exported ONNX models and do not need PyTorch for training or export:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy onnxruntime
```

No PyTorch, no CUDA, no build tools. ONNX Runtime has prebuilt aarch64 wheels so this installs in seconds on a Pi.

---

## Quick start

The quickest way to navigate the repo is:

1. Generate a dataset.
2. Train or download a model.
3. Evaluate it on a dataset.
4. Visualise predictions or export to ONNX.

### Generate datasets

#### Sudoku 9×9

```bash
python -m hrm.data.generate_dataset sudoku \
    --size 9 --num 1000 --difficulty mixed --seed 123 \
    --output data/sudoku_9x9_eval.json
```

#### Maze 15×15

```bash
python -m hrm.data.generate_dataset maze \
    --size 15 --num 500 --seed 123 \
    --output data/maze_15x15_eval.json
```

Use this generator when you want portable `.json` or `.csv` datasets for evaluation, demos, or figure generation. The training script can also generate `.npz` training data inline with `--generate-data`.

### Train models

#### Sudoku 9×9

```bash
python scripts/train_simplified.py \
    --puzzle sudoku_9x9 --generate-data \
    --num-samples 10000 --difficulty mixed \
    --epochs 150 \
    --lr 3e-4 --weight-decay 0.1 --amp \
    --run-name sudoku_9x9 \
    --data-output-path data/sudoku_9x9_train.npz
```

#### Maze 15×15

```bash
python scripts/train_simplified.py \
    --puzzle maze --generate-data \
    --maze-size 15 --num-samples 5000 \
    --epochs 200 --batch-size 128 \
    --lr 3e-4 --weight-decay 0.1 --amp \
    --run-name maze_15x15 \
    --data-output-path data/maze_15x15_train.npz
```

Each run writes `..._best.pt`, `..._final.pt`, and `training_history_*.json` into `model/`. If you already have `.npz` training data, replace `--generate-data` with `--data-path path/to/file.npz`.

### Or use a released model on a generated dataset

Download a released checkpoint, generate a fresh dataset, then run evaluation.

```bash
python scripts/download_model.py --puzzle sudoku_9x9

python -m hrm.data.generate_dataset sudoku \
    --size 9 --num 100 --difficulty mixed --seed 123 \
    --output data/sudoku_9x9_release_eval.json

python scripts/run_simplified.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 \
    --data data/sudoku_9x9_release_eval.json \
    --eval-count 100 --num-examples 5
```

`download_model.py` can filter Sudoku release assets directly. For maze release checkpoints, run `python scripts/download_model.py` without `--puzzle`, or download the maze checkpoint from the Releases page and point the commands below at that `.pt` file.

### Evaluate a maze 15×15 model on a fresh generated dataset

This keeps evaluation separate from training data by generating a new file with a different seed.

```bash
python -m hrm.data.generate_dataset maze \
    --size 15 --num 500 --seed 314 \
    --output data/maze_15x15_eval.json

python scripts/run_simplified.py \
    --model model/simplified_hrm_maze_15x15_best.pt \
    --puzzle maze \
    --data data/maze_15x15_eval.json \
    --eval-count 500 --num-examples 5
```

If you used the default maze run name instead of `--run-name maze_15x15`, replace the model path with `model/simplified_hrm_maze_best.pt`.

### Maze visualiser

Generate comparison figures for predicted vs ground-truth paths:

```bash
python scripts/visualise_maze.py \
    --model model/simplified_hrm_maze_15x15_best.pt \
    --data data/maze_15x15_eval.json \
    --num 5 --save-dir figures/maze_15x15
```

Add `--steps --index 0` to also save a per-step reasoning figure for one selected maze.

### Sudoku solution across reasoning steps

Replay the iterative refinement process in the terminal:

```bash
python scripts/visualise_simplified.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9 \
    --delay 0.3
```

This uses the built-in sample Sudoku. To inspect a specific puzzle, pass `--input` and optionally `--target` as row-major comma-separated values.

---

## ONNX export and inference

Export a trained checkpoint to ONNX and then benchmark or evaluate it with ONNX Runtime.

```bash
# Export (requires PyTorch). Verifies against PyTorch automatically (5/5 must pass).
python scripts/export_onnx.py \
    --model model/simplified_hrm_sudoku_9x9_best.pt \
    --puzzle sudoku_9x9

python scripts/export_onnx.py \
    --model model/simplified_hrm_maze_15x15_best.pt \
    --puzzle maze --maze-size 15

# Run (numpy + onnxruntime only, no PyTorch)
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9

# Evaluate on a dataset
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9 \
    --data data/sudoku_9x9_eval.npz

# Latency benchmark
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9 --benchmark

# Interactive mode
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx \
    --puzzle sudoku_9x9 --interactive
```

The default opset is 18 (not 17). PyTorch's dynamo-based exporter needs it; opset 17 fails with a down-conversion error.

`export_onnx.py` automatically verifies the exported graph against the PyTorch checkpoint before writing the `.onnx` file. The `SimplifiedHRMForONNX` wrapper resolves six ONNX-incompatible patterns: baking the puzzle type enum, unrolling the reasoning loop, forcing standard attention, pre-computing RoPE buffers, selecting the correct output head, and resolving prediction feedback (on for Sudoku, off for maze).

---

## Raspberry Pi 5 deployment

The cross-platform bit: train on a GPU, export once, run the same `.onnx` file on any CPU.

```bash
# On the Pi
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv ~/hrm-env && source ~/hrm-env/bin/activate && pip install numpy onnxruntime
git clone https://github.com/fionntmcc/cross-platform-hrm.git && cd cross-platform-hrm

# Copy models and data from your laptop
# scp model/*.onnx pi@<pi-ip>:~/cross-platform-hrm/model/
# scp data/*.npz pi@<pi-ip>:~/cross-platform-hrm/data/

# Or download from GitHub Releases
python scripts/download_model.py --puzzle sudoku_9x9

# Benchmark
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --benchmark

# Evaluate
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 \
    --data data/sudoku_9x9_eval.npz

# Interactive demo
python scripts/solve_onnx.py \
    --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --interactive
```

No PyTorch, no CUDA, no cross-compilation. ONNX Runtime picks up NEON SIMD on ARM automatically. Same weights, same outputs (verified to 5 decimal places at export time).

---

## Dataset generation

Everything is generated inside the environment, no downloads necessary.

```bash
# Top-level CLI (JSON / CSV output)
python -m hrm.data.generate_dataset sudoku \
    --size 9 --num 10000 --difficulty mixed --seed 123 \
    --output data/sudoku_9x9_train.json

python -m hrm.data.generate_dataset maze \
    --size 15 --num 5000 --seed 123 \
    --output data/maze_15x15_train.json
```

If you'd rather have the `.npz` format that `run_simplified.py` consumes directly, use the standalone generator:

```bash
python hrm/data/sudoku_generator.py \
    --size 9 --num 10000 --difficulty mixed --seed 123 \
    --output data/sudoku_9x9_train
```

Every Sudoku puzzle is validated to have a unique solution. Mazes are paired with their shortest weighted path (Dijkstra) as a binary mask.

---

## Multi-seed runs

For honest error bars in the dissertation:

```bash
# Defaults to seeds 123 / 456 / 789
python scripts/run_seeds.py \
    --puzzle sudoku_9x9 --generate-data --num-samples 10000 --difficulty mixed \
    --epochs 200 --batch-size 512 --lr 3e-4 --weight-decay 0.1 --amp \
    --seed-list 123 456 789 --output-dir model/seed_experiment

# Just re-aggregate without re-training
python scripts/run_seeds.py --puzzle sudoku_9x9 \
    --aggregate-only --output-dir model/seed_experiment
```

Output: `seed_summary_<puzzle>.json` and `seed_comparison_<puzzle>.png` with per-seed curves, mean ± std bands, and a variance flag that trips if any accuracy metric goes over 1.5 %.

---

## Tests

```bash
pytest                                             # full suite, runs in seconds
pytest --cov=hrm --cov-report=term-missing         # with coverage (80 % floor)
pytest test/hrm_test/test_simplified_hrm.py -v     # one file
```

Tests live under `test/hrm_test/` (core package) and `test/other/` (data layer). CI runs the whole thing on Python 3.10 through 3.13.

---

## Dev

Pre-commit runs Black and Ruff before every commit, matching what CI does:

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files   # first time, to clean up existing files
```

Manual:

```bash
black hrm/ test/hrm_test/ test/other/
ruff check --fix hrm/ test/hrm_test/ test/other/
```

---

## Command reference

A full CLI reference for every script (flags, defaults, examples) is available at [`docs/commands.md`](docs/commands.md).

---

## References

The architecture and training procedure follow these two papers. Full bibliographies (RoPE, SwiGLU, RMSNorm, ONNX Runtime, and supporting work) are in each dissertation.

[1] G. Wang, J. Li, Y. Sun, X. Chen, C. Liu, Y. Wu, M. Lu, S. Song, and Y. Abbasi Yadkori, *"Hierarchical Reasoning Model,"* arXiv preprint [arXiv:2506.21734](https://arxiv.org/abs/2506.21734), 2025. Reference code: <https://github.com/sapientinc/HRM>.

[2] R. Ge, Q. Liao, and T. Poggio, *"Hierarchical Reasoning Models: Perspectives and Misconceptions,"* arXiv preprint [arXiv:2510.00355](https://arxiv.org/abs/2510.00355), 2025. Motivates the L-module-only simplification we implement here.

---

## Authors

**Kyrylo Kozlovskyi** (G00425385) — Simplified HRM (L-module only), ONNX export pipeline (`export_onnx.py`), zero-dependency inference (`solve_onnx.py`), Raspberry Pi 5 deployment, GitHub Actions CI/CD, model release scripts (`download_model.py`, `publish_models.py`).

**Fionn McCarthy** (G00414386) — Full HRM reference implementation, weighted-maze generator and task design, training pipeline, metrics / logger modules, multi-seed analysis, Sudoku generator.

Supervised by Dr. John Healy (Department of Computer Science & Applied Physics, ATU Galway). Thanks to Sapient Intelligence for open-sourcing the original HRM code, and to Ge, Liao and Poggio for the follow-up analysis that pointed us at the L-module-only variant.

---

## License

MIT — see [LICENSE](LICENSE). If you build on this for academic work, please also cite Wang et al. [[1]](#references) and Ge, Liao and Poggio [[2]](#references).
