# Cross-Platform HRM

A transformer-based **Hierarchical Reasoning Machine (HRM)** that learns to solve constraint-satisfaction puzzles (Sudoku 4×4, 9×9 and mazes) through iterative reasoning.

The model uses two nested loops of computation inspired by Adaptive Computation Time (ACT):

- **Inner loop**: a Worker (L-level transformer) refines the hidden state over multiple cycles, guided by a Planner (H-level transformer).
- **Outer loop**: an ACT halting mechanism repeats the inner loop, producing intermediate predictions that feed back into the next step, allowing the model to iteratively backtrack and correct itself.

Puzzle-type embeddings enable multi-task learning from a single set of weights.

---

## Setup

```bash
git clone https://github.com/fionntmcc/cross-platform-hrm
cd cross-platform-hrm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Requirements:** Python 3.10+, PyTorch ≥ 2.0, NumPy ≥ 1.24, tqdm ≥ 4.65.

---

## Training

Training is done with `scripts/train_unified.py`. The script can generate training data on the fly or load pre-existing `.npz` files from the `data/` directory.

### Train on 9×9 Sudoku

```bash
python scripts/train_unified.py --puzzle sudoku_9x9 --epochs 100
```

If no data file exists yet, pass `--generate-data` to create one:

```bash
python scripts/train_unified.py --puzzle sudoku_9x9 --generate-data --num-samples 10000 --epochs 100
```

### Train on 4×4 Sudoku

```bash
python scripts/train_unified.py --puzzle sudoku_4x4 --generate-data --num-samples 5000 --epochs 50
```

### Script Parameters

#### Data

| Flag | Default | Description |
|---|---|---|
| `--puzzle` | `sudoku_9x9` | Puzzle type to train on. Choices: `sudoku_4x4`, `sudoku_9x9`, `maze`, `all`. |
| `--data-path` | *auto* | Path to a `.npz` training data file. If omitted, the script looks in `data/` for a default file matching the puzzle type. |
| `--generate-data` | off | Generate new training data before training. Required the first time if no `.npz` file exists. |
| `--num-samples` | `10000` | Number of puzzles to generate when `--generate-data` is set. |
| `--difficulty` | `medium` | Puzzle difficulty. Choices: `easy`, `medium`, `hard`. Controls how many cells are left empty. |

#### Model Architecture

| Flag | Default | Description |
|---|---|---|
| `--hidden-size` | `256` | Hidden dimension of the transformer. |
| `--num-heads` | `4` | Number of attention heads. |
| `--num-layers` | `4` | Number of transformer blocks in both the Worker and Planner. |
| `--halt-max-steps` | *from config* | Override the maximum outer ACT halting steps (default is puzzle-dependent: 4 for 4×4, 8 for 9×9). |
| `--l-cycles` | *from config* | Override L-cycles (Worker iterations) per H-cycle (default: 4 for 4×4, 8 for 9×9). |

#### Training

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `100` | Number of training epochs. |
| `--batch-size` | `32` | Batch size. |
| `--lr` | `1e-4` | Learning rate (Adam). |
| `--val-split` | `0.1` | Fraction of data reserved for validation. |
| `--multistep-loss` | off | Compute loss on **all** outer ACT steps (weighted so later steps count more), not just the final prediction. Encourages useful intermediate predictions. |
| `--constraint-weight` | `0.0` | Weight for an auxiliary Sudoku constraint penalty that penalises duplicate digits in rows/columns/boxes. Try `0.1` to enable. |
| `--label-smoothing` | `0.0` | Label smoothing factor for cross-entropy loss. |

#### System

| Flag | Default | Description |
|---|---|---|
| `--save-dir` | `model` | Directory where model checkpoints and training history are saved. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--device` | `auto` | Compute device. `auto` picks CUDA → MPS → CPU in order of availability. |
| `--amp` | off | Enable automatic mixed-precision training (CUDA only). |
| `--compile` | off | Enable `torch.compile` for faster training (PyTorch 2.0+). |

### Output

Training saves the following to `--save-dir` (default `model/`):

- `unified_hrm_<puzzle>_best.pt` : best model checkpoint (by validation accuracy).
- `training_history_<puzzle>.json` : per-epoch metrics (loss, accuracy, etc.).

---

## Inference

Use `scripts/demo_unified.py` to load a trained checkpoint and solve puzzles.

### Solve a 9×9 Sudoku

```bash
python scripts/demo_unified.py --puzzle sudoku_9x9 --model model/unified_hrm_sudoku_9x9_best.pt
```

### Solve a 4×4 Sudoku

```bash
python scripts/demo_unified.py --puzzle sudoku_4x4 --model model/unified_hrm_sudoku_4x4_best.pt
```

### Interactive Mode

Enter your own puzzles at the command line:

```bash
python scripts/demo_unified.py --interactive --model model/unified_hrm_sudoku_9x9_best.pt
```

You will be prompted to choose a puzzle type and enter cell values as comma-separated integers (use `0` for empty cells).

### Script Parameters

| Flag | Default | Description |
|---|---|---|
| `--model` | *none* | Path to a trained model checkpoint (`.pt`). If omitted, a randomly initialised model is used (predictions will be meaningless). |
| `--puzzle` | `sudoku_9x9` | Puzzle type to demo. Choices: `sudoku_4x4`, `sudoku_9x9`, `all`. |
| `--interactive` | off | Launch interactive mode where you can input your own puzzles. |
| `--device` | `auto` | Compute device. `auto` picks CUDA → MPS → CPU. |

---

## Project Structure

```
cross-platform-hrm/
├── hrm/                    # Core model code
│   ├── model_unified.py    # Unified HRM model & config
│   ├── core/               # Iteration loops & ACT halting
│   └── layers/             # Transformer, Worker, Planner, I/O networks
├── scripts/
│   ├── train_unified.py    # Training script
│   └── demo_unified.py     # Inference / demo script
├── generators/             # Puzzle generators (Sudoku, maze)
├── data/                   # Training data (.npz files)
├── model/                  # Saved checkpoints & training history
└── test/                   # Unit tests
```
