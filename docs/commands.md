# Command Reference

Complete CLI reference for the cross-platform-hrm project.

---

## Training

### `scripts/train_simplified.py`

Train the SimplifiedHRM (L-Module Only) model on Sudoku or maze puzzles.

```bash
python scripts/train_simplified.py [OPTIONS]
```

**Data options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--puzzle` | `sudoku_9x9` | Puzzle type: `sudoku_4x4`, `sudoku_9x9`, `maze` |
| `--data-path` | auto | Path to training data (.npz) |
| `--data-output-path` | auto | Custom output path for generated data (.npz) |
| `--generate-data` | off | Generate new training data |
| `--num-samples` | `1000` | Number of samples to generate |
| `--difficulty` | `hard` | Puzzle difficulty: `easy`, `medium`, `hard`, `mixed` |
| `--maze-size` | `15` | Maze grid size (>= 7, forced odd) |

**Model options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden-size` | `256` | Model hidden dimension |
| `--num-heads` | `4` | Number of attention heads |
| `--num-layers` | `8` | Number of transformer layers |
| `--reasoning-steps` | `16` | Number of iterative reasoning steps |
| `--no-feedback` | off | Disable prediction feedback / self-conditioning |
| `--small` | off | Small config (128d, 4L, 8 steps) for quick experiments |

**Training options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `64` | Batch size (physical, fits in VRAM) |
| `--accum-steps` | `1` | Gradient accumulation steps |
| `--lr` | `3e-4` | Learning rate |
| `--weight-decay` | `0.1` | Weight decay |
| `--val-split` | `0.1` | Validation split ratio |
| `--amp` | off | Enable mixed-precision training |
| `--compile` | off | Enable torch.compile (PyTorch 2.0+) |
| `--constraint-weight` | `0.0` | Sudoku constraint penalty weight (0 = off) |
| `--label-smoothing` | `0.0` | Label smoothing factor (0 = off) |

**Output options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--save-dir` | `model` | Directory to save models |
| `--run-name` | auto | Unique run name for saved filenames |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |

**Examples:**

```bash
# Sudoku 9x9 — mixed difficulty, full training
python scripts/train_simplified.py --puzzle sudoku_9x9 --generate-data --num-samples 10000 --difficulty mixed --epochs 200 --batch-size 512 --lr 3e-4 --weight-decay 0.1 --amp

# Sudoku 4x4 — quick training
python scripts/train_simplified.py --puzzle sudoku_4x4 --generate-data --num-samples 5000 --difficulty mixed --epochs 100 --batch-size 512 --lr 3e-4 --weight-decay 0.1 --amp

# Maze 11x11
python scripts/train_simplified.py --puzzle maze --generate-data --num-samples 5000 --maze-size 11 --epochs 200 --batch-size 128 --lr 3e-4 --weight-decay 0.1 --amp

# Maze 15x15
python scripts/train_simplified.py --puzzle maze --generate-data --num-samples 5000 --maze-size 15 --epochs 200 --batch-size 128 --lr 3e-4 --weight-decay 0.1 --amp

# Small model for quick experiments
python scripts/train_simplified.py --puzzle sudoku_9x9 --generate-data --small --epochs 20 --batch-size 256
```

---

### `scripts/run_seeds.py`

Automated multi-seed training for measuring performance variance.

```bash
python scripts/run_seeds.py [OPTIONS] [-- TRAIN_SIMPLIFIED_ARGS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed-list` | `123 456 789` | Seeds to train with |
| `--output-dir` | `model/seed_experiment` | Root output directory |
| `--aggregate-only` | off | Skip training, only aggregate existing results |
| `--puzzle` | `sudoku_9x9` | Puzzle type: `sudoku_4x4`, `sudoku_9x9`, `maze` |

All other flags are forwarded to `train_simplified.py`.

**Examples:**

```bash
# 3-seed experiment on maze
python scripts/run_seeds.py --puzzle maze --generate-data --num-samples 500 --maze-size 11 --epochs 100

# Custom seeds
python scripts/run_seeds.py --puzzle sudoku_9x9 --epochs 50 --seed-list 1 2 3

# Aggregate only (recompute stats from existing runs)
python scripts/run_seeds.py --puzzle maze --aggregate-only --output-dir model/seed_experiment
```

---

## Evaluation

### `scripts/run_simplified.py`

Evaluate a trained PyTorch model on a dataset. Requires PyTorch.

```bash
python scripts/run_simplified.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to checkpoint (.pt file) |
| `--puzzle` | `sudoku_4x4` | Puzzle type: `sudoku_4x4`, `sudoku_9x9`, `maze` |
| `--data` | auto | Path to .npz data file |
| `--num-examples` | `5` | Number of example puzzles to display |
| `--eval-count` | `200` | Number of puzzles to evaluate |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |

**Examples:**

```bash
# Evaluate 9x9 Sudoku
python scripts/run_simplified.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9 --data data/sudoku_9x9_eval.npz

# Evaluate maze
python scripts/run_simplified.py --model model/simplified_hrm_maze_11x11.pt --puzzle maze --data data/maze_11x11_unseen.npz
```

---

## Demo & Visualisation

### `scripts/demo_simplified.py`

Interactive demo with hardcoded examples, dataset examples, and interactive mode.

```bash
python scripts/demo_simplified.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | auto | Path to trained model checkpoint |
| `--puzzle` | `sudoku_4x4` | Puzzle type: `sudoku_4x4`, `sudoku_9x9`, `maze`, `all` |
| `--data` | none | Path to .npz data file (for maze demo) |
| `--num-examples` | `3` | Number of examples to display |
| `--interactive` | off | Run in interactive mode |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |

**Examples:**

```bash
# Demo all puzzle types
python scripts/demo_simplified.py --puzzle all

# Interactive 9x9 Sudoku
python scripts/demo_simplified.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9 --interactive

# Maze demo from dataset
python scripts/demo_simplified.py --model model/simplified_hrm_maze_11x11.pt --puzzle maze --data data/maze_11x11_train.npz
```

---

### `scripts/visualise_simplified.py`

Terminal visualisation of reasoning steps — watch the model refine its solution.

```bash
python scripts/visualise_simplified.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to trained model checkpoint |
| `--puzzle` | required | Puzzle type: `sudoku_4x4`, `sudoku_9x9`, `maze` |
| `--input` | none | Puzzle as comma-separated integers (row-major, 0 = empty) |
| `--target` | none | Target solution as comma-separated integers |
| `--data` | none | Path to .npz data file (picks random example) |
| `--index` | random | Index of puzzle to pick from `--data` file |
| `--steps` | model default | Override number of reasoning steps |
| `--delay` | `0.3` | Seconds to pause between steps |
| `--device` | `cpu` | Device: `cpu`, `cuda`, `mps` |

**Examples:**

```bash
# Visualise 9x9 Sudoku step by step
python scripts/visualise_simplified.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9

# Visualise maze from dataset
python scripts/visualise_simplified.py --model model/simplified_hrm_maze_11x11.pt --puzzle maze --data data/maze_11x11_train.npz

# Custom puzzle input
python scripts/visualise_simplified.py --model model/simplified_hrm_sudoku_4x4.pt --puzzle sudoku_4x4 --input "1,0,0,4,0,0,1,0,0,1,0,0,4,0,0,1"

# Slow playback
python scripts/visualise_simplified.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9 --delay 1.0
```

---

### `scripts/visualise_maze.py`

Publication-quality maze visualisation using matplotlib.

```bash
python scripts/visualise_maze.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to trained maze checkpoint (.pt) |
| `--data` | none | Path to maze .npz dataset |
| `--generate` | off | Generate fresh mazes instead of loading from `--data` |
| `--maze-size` | `11` | Grid size for generated mazes |
| `--num` | `5` | Number of mazes to visualise |
| `--index` | none | Specific puzzle index from dataset |
| `--steps` | off | Show per-step reasoning trajectory |
| `--save-dir` | none | Directory to save figures (default: display only) |
| `--show` | off | Display figures interactively |
| `--dpi` | `200` | Figure DPI |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |

**Examples:**

```bash
# Save maze comparison figures
python scripts/visualise_maze.py --model model/simplified_hrm_maze_11x11.pt --data data/maze_11x11_train.npz --num 10 --save-dir figures/

# Show step-by-step reasoning
python scripts/visualise_maze.py --model model/simplified_hrm_maze_11x11.pt --data data/maze_11x11_train.npz --steps --index 0 --show

# Generate fresh mazes and visualise
python scripts/visualise_maze.py --model model/simplified_hrm_maze_11x11.pt --generate --maze-size 11 --num 5 --show
```

---

## ONNX Export & Inference

### `scripts/export_onnx.py`

Export trained SimplifiedHRM checkpoint to ONNX format. Requires PyTorch.

```bash
python scripts/export_onnx.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to trained .pt checkpoint |
| `--puzzle` | required | Puzzle type: `sudoku_4x4`, `sudoku_9x9`, `maze` |
| `--output`, `-o` | auto | Output .onnx path |
| `--reasoning-steps` | model default | Override number of reasoning steps |
| `--maze-size` | `11` | Maze grid size for seq_len calculation |
| `--opset` | `18` | ONNX opset version |
| `--skip-verify` | off | Skip ONNX verification step |

**Examples:**

```bash
# Sudoku 9x9
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9

# Sudoku 4x4
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_4x4.pt --puzzle sudoku_4x4

# Maze 11x11
python scripts/export_onnx.py --model model/simplified_hrm_maze_11x11.pt --puzzle maze --maze-size 11

# Maze 15x15
python scripts/export_onnx.py --model model/simplified_hrm_maze_15x15.pt --puzzle maze --maze-size 15

# Step variants for latency comparison
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9 --reasoning-steps 16 --output model/simplified_hrm_9x9_s16.onnx
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9 --reasoning-steps 8 --output model/simplified_hrm_9x9_s8.onnx
python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9.pt --puzzle sudoku_9x9 --reasoning-steps 4 --output model/simplified_hrm_9x9_s4.onnx
```

---

### `scripts/solve_onnx.py`

ONNX inference and benchmarking. **No PyTorch required** — only `numpy` + `onnxruntime`.

```bash
python scripts/solve_onnx.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to .onnx model file |
| `--puzzle` | required | Puzzle type: `sudoku_4x4`, `sudoku_9x9`, `maze` |
| `--maze-size` | `11` | Maze grid size |
| `--data` | none | Path to .npz dataset for evaluation |
| `--benchmark` | off | Run latency benchmark |
| `--interactive` | off | Interactive puzzle solving mode |
| `--num-puzzles` | `3` | Number of random puzzles to demo |

**Modes:**

- **Demo** (default): solve random puzzles, display with colour
- **Dataset eval** (`--data`): compute token/puzzle accuracy on .npz file
- **Benchmark** (`--benchmark`): warmup + timed runs, latency stats
- **Interactive** (`--interactive`): type puzzle values, get instant solution

**Examples:**

```bash
# Demo — solve random puzzles
python scripts/solve_onnx.py --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9

# Evaluate on dataset
python scripts/solve_onnx.py --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --data data/sudoku_9x9_eval.npz

# Benchmark latency
python scripts/solve_onnx.py --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --benchmark

# Interactive mode
python scripts/solve_onnx.py --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --interactive

# Maze evaluation
python scripts/solve_onnx.py --model model/simplified_hrm_maze_11x11.onnx --puzzle maze --maze-size 11 --data data/maze_11x11_eval.npz

# Maze benchmark
python scripts/solve_onnx.py --model model/simplified_hrm_maze_15x15.onnx --puzzle maze --maze-size 15 --benchmark
```

---

## Data Generation

### `python -m hrm.data.generate_dataset`

Generate Sudoku or maze datasets as JSON or CSV.

```bash
python -m hrm.data.generate_dataset {sudoku,maze} [OPTIONS]
```

**Sudoku options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--size` | `9` | Grid size: `4` or `9` |
| `--num` | `100` | Number of puzzles |
| `--difficulty` | `medium` | Difficulty: `easy`, `medium`, `hard`, `mixed` |
| `--seed` | none | Random seed |
| `-o`, `--output` | required | Output file path (.json or .csv) |
| `--format` | auto | Output format |

**Maze options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--size` | `15` | Grid size (>= 7) |
| `--num` | `100` | Number of puzzles |
| `--seed` | none | Random seed |
| `-o`, `--output` | required | Output file path (.json or .csv) |
| `--format` | auto | Output format |

**Examples:**

```bash
# Generate 500 unseen mazes
python -m hrm.data.generate_dataset maze --size 11 --num 500 --seed 99 -o data/maze_11x11_unseen.json

# Generate 1000 hard 9x9 Sudoku puzzles
python -m hrm.data.generate_dataset sudoku --size 9 --num 1000 --difficulty hard -o data/sudoku_hard.json
```

**Quick .npz generation (inline):**

```bash
# Sudoku 4x4 mixed
python -c "import numpy as np; from hrm.data import generate_sudoku_dataset; d=generate_sudoku_dataset(500, grid_size=4, difficulty='mixed', seed=99); np.savez('data/sudoku_4x4_eval.npz', problems=d['problems'], solutions=d['solutions']); print('Saved', len(d['problems']))"

# Sudoku 9x9 mixed
python -c "import numpy as np; from hrm.data import generate_sudoku_dataset; d=generate_sudoku_dataset(500, grid_size=9, difficulty='mixed', seed=99); np.savez('data/sudoku_9x9_eval.npz', problems=d['problems'], solutions=d['solutions']); print('Saved', len(d['problems']))"

# Maze 11x11 unseen
python -c "import numpy as np; from hrm.data import generate_weighted_maze_dataset; d=generate_weighted_maze_dataset(500, grid_size=11, seed=99); np.savez('data/maze_11x11_unseen.npz', problems=d['problems'], solutions=d['solutions']); print('Saved', len(d['problems']))"

# Maze 15x15 unseen
python -c "import numpy as np; from hrm.data import generate_weighted_maze_dataset; d=generate_weighted_maze_dataset(500, grid_size=15, seed=99); np.savez('data/maze_15x15_unseen.npz', problems=d['problems'], solutions=d['solutions']); print('Saved', len(d['problems']))"
```

---

## Model Release

### `scripts/publish_models.py`

Tag, upload, and publish trained models to a GitHub Release.

```bash
python scripts/publish_models.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tag` | required | Release tag (e.g. `v1.0.0`) |
| `--message` | auto | Annotated tag message / release description |
| `--token` | `GITHUB_TOKEN` env | GitHub PAT with `repo` scope |
| `--puzzle` | all | Upload only models for this puzzle type |
| `--skip-tag` | off | Skip creating/pushing git tag |
| `--dry-run` | off | Create and push tag only, skip upload |

**Examples:**

```bash
# Full release
python scripts/publish_models.py --tag v1.0.0 --token ghp_xxxxx

# Dry run (tag only)
python scripts/publish_models.py --tag v1.0.0 --dry-run

# Upload only Sudoku models
python scripts/publish_models.py --tag v1.0.0 --puzzle sudoku_9x9 --skip-tag
```

---

### `scripts/download_model.py`

Download trained models from a GitHub Release.

```bash
python scripts/download_model.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tag` | `latest` | Release tag to download from |
| `--puzzle` | all | Only download models for this puzzle type |
| `--output-dir` | `model/` | Directory to save models |
| `--list` | off | List available releases and exit |
| `--token` | `GITHUB_TOKEN` env | GitHub PAT |

**Examples:**

```bash
# Download latest release
python scripts/download_model.py

# Download specific tag
python scripts/download_model.py --tag v1.0.0

# Download only maze models
python scripts/download_model.py --puzzle maze

# List available releases
python scripts/download_model.py --list
```

---

## Raspberry Pi 5 Setup

```bash
# Install
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv ~/hrm-env && source ~/hrm-env/bin/activate && pip install numpy onnxruntime
git clone https://github.com/fionntmcc/cross-platform-hrm.git && cd cross-platform-hrm

# Copy models + data from laptop
scp model/*.onnx pi@<pi-ip>:~/cross-platform-hrm/model/
scp data/*.npz pi@<pi-ip>:~/cross-platform-hrm/data/

# Run
python scripts/solve_onnx.py --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 --benchmark
python scripts/solve_onnx.py --model model/simplified_hrm_maze_11x11.onnx --puzzle maze --maze-size 11 --data data/maze_11x11_eval.npz
```
