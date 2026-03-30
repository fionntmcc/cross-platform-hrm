"""
ONNX Export for Simplified HRM (L-Module Only)

Exports a trained SimplifiedHRM checkpoint to ONNX format for
cross-platform CPU inference (Raspberry Pi 5, edge devices).

ONNX export requires removing Python-level control flow that the
ONNX tracer cannot handle.  The SimplifiedHRMForONNX wrapper:
    - Bakes the PuzzleType at export time (no enum dispatch)
    - Unrolls the reasoning loop (fixed iteration count)
    - Forces standard attention (no FlashAttention conditional)
    - Pre-computes RoPE cos/sin as registered buffers
    - Inlines InputEmbedding to avoid PuzzleType in forward()
    - Selects the correct OutputHead at construction time
    - Disables prediction feedback for maze (different token domains)

Supports all three puzzle types:
    - sudoku_4x4: 16 tokens, vocab 5
    - sudoku_9x9: 81 tokens, vocab 10
    - maze:       grid_size² tokens, input vocab 10, output vocab 2

Usage:
    # Export 9x9 Sudoku model
    python scripts/export_onnx.py \\
        --model model/simplified_hrm_sudoku_9x9_best.pt \\
        --puzzle sudoku_9x9

    # Export 4x4 Sudoku model
    python scripts/export_onnx.py \\
        --model model/simplified_hrm_sudoku_4x4_best.pt \\
        --puzzle sudoku_4x4

    # Export maze model (11x11)
    python scripts/export_onnx.py \\
        --model model/simplified_hrm_maze_best.pt \\
        --puzzle maze --maze-size 11

    # Export with fewer reasoning steps (faster inference)
    python scripts/export_onnx.py \\
        --model model/simplified_hrm_sudoku_9x9_best.pt \\
        --puzzle sudoku_9x9 --reasoning-steps 8

    # Multiple step variants for latency comparison
    python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9_best.pt \\
        --puzzle sudoku_9x9 --reasoning-steps 16 --output model/simplified_hrm_9x9_s16.onnx
    python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9_best.pt \\
        --puzzle sudoku_9x9 --reasoning-steps 8  --output model/simplified_hrm_9x9_s8.onnx
    python scripts/export_onnx.py --model model/simplified_hrm_sudoku_9x9_best.pt \\
        --puzzle sudoku_9x9 --reasoning-steps 4  --output model/simplified_hrm_9x9_s4.onnx

Authors:
    - Kyrylo Kozlovskyi (G00425385)
    - Fionn McCarthy (G00414386)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.model_simplified import (
    PUZZLE_DEFAULTS,
    PuzzleType,
    SimplifiedHRM,
    SimplifiedHRMConfig,
)


def _check_onnx_export_dependencies() -> None:
    """Fail fast with a clear message when ONNX export deps are missing."""
    try:
        import onnxscript  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'onnxscript'. "
            "Install ONNX export dependencies with: "
            "pip install onnx onnxscript"
        ) from exc


# -----------------------------------------------------------------------
# ONNX-compatible wrapper
# -----------------------------------------------------------------------
class SimplifiedHRMForONNX(nn.Module):
    """
    ONNX-exportable wrapper around SimplifiedHRM.

    Removes all Python-level control flow that the ONNX tracer cannot
    handle:
        - PuzzleType enum -> baked into the wrapper at export time
        - Reasoning loop  -> unrolled (fixed iteration count)
        - Flash attention -> forced off
        - RoPE cos/sin    -> pre-computed and stored as buffers
        - Prediction feedback -> resolved at construction (on for Sudoku,
          off for maze since input/output are different token domains)
        - InputEmbedding  -> inlined to avoid PuzzleType in forward()
        - OutputHead      -> correct head selected at construction time

    The wrapper shares all weights with the original model (no
    duplication in memory).

    Args:
        model: Trained SimplifiedHRM model.
        puzzle_type: Which puzzle type to bake in.
        num_reasoning_steps: Override reasoning steps (None = model default).
    """

    def __init__(
        self,
        model: SimplifiedHRM,
        puzzle_type: PuzzleType,
        num_reasoning_steps: int | None = None,
    ):
        super().__init__()

        self.num_reasoning_steps = (
            num_reasoning_steps or model.config.num_reasoning_steps
        )

        # Prediction feedback: on for Sudoku, off for maze.
        # Maze input tokens (0-9: wall/path/start/goal/weighted) and
        # output tokens (0-1: not-on-path/on-path) are different domains,
        # so re-embedding predictions as input doesn't work.
        is_maze = puzzle_type == PuzzleType.MAZE
        self.use_feedback = model.config.use_prediction_feedback and not is_maze

        # ---- Share model sub-modules (no weight duplication) ----
        # Reference the sub-layers of InputEmbedding directly so that
        # _embed() can inline the forward without PuzzleType dispatch.
        self.tok_emb = model.input_net.tok_emb
        self.puzzle_emb = model.input_net.puzzle_emb
        self.input_proj = model.input_net.input_proj
        self.reasoning = model.reasoning

        # Select the correct output head (bake puzzle type)
        if puzzle_type == PuzzleType.SUDOKU_4X4:
            self.output_proj = model.output_head.heads["sudoku_4x4"]
        elif puzzle_type == PuzzleType.SUDOKU_9X9:
            self.output_proj = model.output_head.heads["sudoku_9x9"]
        else:
            self.output_proj = model.output_head.heads["maze"]

        # ---- Bake puzzle type index as buffer ----
        self.register_buffer(
            "puzzle_type_idx",
            torch.tensor([puzzle_type.value - 1], dtype=torch.long),
        )

        # ---- Copy z_L_init buffer ----
        self.register_buffer("z_L_init", model.z_L_init.clone())

        # ---- Embedding scale factor (Python constant, OK for tracing) ----
        self._scale = math.sqrt(model.config.hidden_size)

        # ---- Force standard attention (no FlashAttention for ONNX) ----
        for layer in self.reasoning.layers:
            layer.self_attn._has_flash_attn = False

        # ---- Pre-compute RoPE and register as buffers ----
        with torch.no_grad():
            cos, sin = model.rotary_emb()
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed tokens -- inlined InputEmbedding.forward() to avoid enum.

        Steps:
            1. Token embedding lookup
            2. Add puzzle-type embedding (baked index)
            3. Scale by sqrt(hidden_size)
            4. Linear projection (no bias)
            5. No dropout (inference mode)
        """
        h = self.tok_emb(x)
        h = h + self.puzzle_emb(self.puzzle_type_idx).unsqueeze(0)
        h = h * self._scale
        h = self.input_proj(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ONNX-compatible forward pass.

        Args:
            x: Integer tokens (batch, seq_len) in [0, vocab_size).

        Returns:
            Predictions (batch, seq_len) as integer token IDs.
        """
        batch_size, seq_len = x.shape

        # Embed input
        z_input = self._embed(x)

        # Initialise latent state
        z_L = self.z_L_init.expand(batch_size, seq_len, -1).clone()

        # Given mask for feedback (True where puzzle has clues)
        given_mask = x != 0

        # RoPE cache (pre-computed, stored as buffers)
        cos_sin = (self.rope_cos, self.rope_sin)

        # Unrolled reasoning loop.
        # The tracer unrolls this since self.num_reasoning_steps is a
        # Python int.  Each iteration traces through the same transformer
        # weights (weight sharing / recurrence).
        logits = None
        for step in range(self.num_reasoning_steps):
            # Apply reasoning module (input injection + transformer)
            z_L = self.reasoning(z_L, z_input, cos_sin=cos_sin)

            # Compute logits via baked output head
            logits = self.output_proj(z_L)

            # Prediction feedback (all steps except last, Sudoku only).
            # self.use_feedback is a Python bool set at construction,
            # so the tracer resolves this branch at trace time.
            if self.use_feedback and step < self.num_reasoning_steps - 1:
                preds = logits.argmax(dim=-1)
                refined = torch.where(given_mask, x, preds)
                z_input = self._embed(refined)

            # Detach for next step (no BPTT -- no-op at inference)
            z_L = z_L.detach()

        # Return integer predictions
        assert logits is not None
        return logits.argmax(dim=-1)


# -----------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------
def load_model(checkpoint_path: str, device: str = "cpu") -> SimplifiedHRM:
    """Load a trained SimplifiedHRM from checkpoint.

    Handles both current format (config as SimplifiedHRMConfig dataclass)
    and legacy format (hyperparams dict).
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    if "config" in ckpt and isinstance(ckpt["config"], SimplifiedHRMConfig):
        config = ckpt["config"]
    elif "hyperparams" in ckpt:
        hp = ckpt["hyperparams"]
        config = SimplifiedHRMConfig(
            hidden_size=hp.get("hidden_size", 256),
            num_heads=hp.get("num_heads", 4),
            num_layers=hp.get("num_layers", 8),
            num_reasoning_steps=hp.get("reasoning_steps", 16),
            use_prediction_feedback=hp.get("prediction_feedback", True),
        )
    else:
        print("  Warning: No config found in checkpoint, using defaults")
        config = SimplifiedHRMConfig()

    model = SimplifiedHRM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Parameters:  {model.num_parameters:,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers:      {config.num_layers}")
    print(f"  Steps:       {config.num_reasoning_steps}")
    print(f"  Feedback:    {config.use_prediction_feedback}")
    return model


# -----------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------
def export_to_onnx(
    model: SimplifiedHRM,
    puzzle_type: PuzzleType,
    output_path: str,
    num_reasoning_steps: int | None = None,
    seq_len: int | None = None,
    opset_version: int = 18,
) -> str:
    """
    Export SimplifiedHRM to ONNX format.

    Args:
        model: Trained SimplifiedHRM.
        puzzle_type: Puzzle type to bake into the export.
        output_path: Path for the .onnx file.
        num_reasoning_steps: Override reasoning steps (None = model default).
        seq_len: Override sequence length for dummy input.
        opset_version: ONNX opset version (default: 18).

    Returns:
        Path to the exported .onnx file.
    """
    defaults = PUZZLE_DEFAULTS[puzzle_type]
    if seq_len is None:
        seq_len = defaults["seq_len"]
    vocab_size = defaults["vocab_size"]
    is_maze = puzzle_type == PuzzleType.MAZE

    steps = num_reasoning_steps or model.config.num_reasoning_steps

    print(f"\nExport config:")
    print(f"  Puzzle type:     {puzzle_type.name}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input vocab:     {vocab_size}")
    print(f"  Output classes:  {'2 (binary)' if is_maze else vocab_size}")
    print(f"  Reasoning steps: {steps}")
    print(f"  Pred. feedback:  {'Off (maze)' if is_maze else 'On'}")

    # Newer PyTorch ONNX exporters depend on onnxscript.
    _check_onnx_export_dependencies()

    # Create ONNX wrapper (shares weights, no duplication)
    wrapper = SimplifiedHRMForONNX(
        model, puzzle_type, num_reasoning_steps=num_reasoning_steps
    )
    wrapper.eval()

    # Create dummy input
    dummy = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    print(f"  Dummy input:     shape={dummy.shape}, dtype={dummy.dtype}")

    # Verify wrapper produces valid output before export
    print("\nVerifying wrapper...")
    with torch.no_grad():
        test_out = wrapper(dummy)
    print(f"  Output shape: {test_out.shape}, dtype: {test_out.dtype}")
    assert test_out.shape == (1, seq_len), f"Bad output shape: {test_out.shape}"
    print("  \u2713 Wrapper verification passed")

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        opset_version=opset_version,
        input_names=["puzzle"],
        output_names=["predictions"],
        dynamic_axes={
            "puzzle": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
    )
    export_time = time.time() - t0

    file_size = os.path.getsize(output_path) / 1024**2
    print(f"  \u2713 Exported in {export_time:.1f}s")
    print(f"  File: {output_path} ({file_size:.1f} MB)")

    return output_path


# -----------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------
def verify_onnx(
    onnx_path: str,
    pytorch_model: SimplifiedHRM,
    puzzle_type: PuzzleType,
    num_reasoning_steps: int | None = None,
    seq_len: int | None = None,
    num_tests: int = 5,
) -> bool:
    """
    Verify ONNX model matches PyTorch model outputs.

    Args:
        onnx_path: Path to exported .onnx file.
        pytorch_model: Original PyTorch model for comparison.
        puzzle_type: Puzzle type used for export.
        num_reasoning_steps: Override reasoning steps.
        seq_len: Sequence length (auto-detected if None).
        num_tests: Number of random inputs to test.

    Returns:
        True if all tests pass.
    """
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError:
        print("\n\u26a0\ufe0f  onnxruntime not installed. Skipping verification.")
        print("  Install with: pip install onnxruntime")
        return False

    # Optional: validate ONNX graph structure
    try:
        import onnx  # type: ignore[import-not-found]

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("\n\u2713 ONNX model passes onnx.checker validation")
    except ImportError:
        print("\n\u26a0\ufe0f  onnx package not installed. Skipping graph validation.")
    except Exception as e:
        print(f"\n\u26a0\ufe0f  ONNX validation warning: {e}")

    defaults = PUZZLE_DEFAULTS[puzzle_type]
    if seq_len is None:
        seq_len = defaults["seq_len"]
    vocab_size = defaults["vocab_size"]

    # Create ONNX runtime session
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    # Create PyTorch wrapper for comparison (same wrapper used for export)
    wrapper = SimplifiedHRMForONNX(
        pytorch_model, puzzle_type, num_reasoning_steps=num_reasoning_steps
    )
    wrapper.eval()

    print(f"\nVerifying ONNX vs PyTorch ({num_tests} random inputs)...")
    all_match = True

    for i in range(num_tests):
        inp = np.random.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

        # ONNX inference
        ort_out = sess.run(None, {"puzzle": inp})[0]

        # PyTorch inference
        with torch.no_grad():
            pt_out = wrapper(torch.from_numpy(inp)).numpy()

        match = np.array_equal(ort_out, pt_out)
        status = "\u2713" if match else "\u2717"
        print(f"  Test {i + 1}: {status}")

        if not match:
            all_match = False
            diff = int((ort_out != pt_out).sum())
            print(f"    Mismatched cells: {diff}/{seq_len}")

    if all_match:
        print("\u2713 All tests passed \u2014 ONNX matches PyTorch exactly")
    else:
        print(
            "\u26a0\ufe0f  Some mismatches detected "
            "(may be minor floating-point differences in argmax ties)"
        )

    # Benchmark ONNX inference speed
    print(f"\nBenchmarking ONNX inference (CPU)...")
    inp = np.random.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

    # Warmup
    for _ in range(3):
        sess.run(None, {"puzzle": inp})

    # Timed runs
    times = []
    for _ in range(20):
        t0 = time.time()
        sess.run(None, {"puzzle": inp})
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  Latency: {avg_ms:.1f} \u00b1 {std_ms:.1f} ms per puzzle")
    print(f"  Throughput: {1000 / avg_ms:.0f} puzzles/sec")

    return all_match


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SimplifiedHRM to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Sudoku 9x9
  python scripts/export_onnx.py \\
      --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9

  # Sudoku 9x9 with 8 reasoning steps (faster inference)
  python scripts/export_onnx.py \\
      --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9 \\
      --reasoning-steps 8

  # Sudoku 4x4
  python scripts/export_onnx.py \\
      --model model/simplified_hrm_sudoku_4x4_best.pt --puzzle sudoku_4x4

  # Maze 11x11
  python scripts/export_onnx.py \\
      --model model/simplified_hrm_maze_best.pt --puzzle maze --maze-size 11

  # Maze 15x15
  python scripts/export_onnx.py \\
      --model model/simplified_hrm_maze_best.pt --puzzle maze --maze-size 15
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained .pt checkpoint",
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        required=True,
        choices=["sudoku_4x4", "sudoku_9x9", "maze"],
        help="Puzzle type to export",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output .onnx path (default: auto-generated from model name)",
    )
    parser.add_argument(
        "--reasoning-steps",
        type=int,
        default=None,
        help="Override number of reasoning steps (default: from model config)",
    )
    parser.add_argument(
        "--maze-size",
        type=int,
        default=11,
        help="Maze grid size for seq_len calculation (default: 11)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18, minimum for modern PyTorch)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX verification step",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Map puzzle string to PuzzleType enum
    puzzle_map = {
        "sudoku_4x4": PuzzleType.SUDOKU_4X4,
        "sudoku_9x9": PuzzleType.SUDOKU_9X9,
        "maze": PuzzleType.MAZE,
    }
    puzzle_type = puzzle_map[args.puzzle]

    # Determine sequence length
    if args.puzzle == "maze":
        seq_len = args.maze_size * args.maze_size
    else:
        seq_len = PUZZLE_DEFAULTS[puzzle_type]["seq_len"]

    # Auto-generate output path if not specified
    if args.output is None:
        stem = Path(args.model).stem.replace("_best", "").replace("_final", "")
        steps_suffix = (
            f"_s{args.reasoning_steps}" if args.reasoning_steps else ""
        )
        if args.puzzle == "maze":
            size_tag = f"_{args.maze_size}x{args.maze_size}"
            if f"{args.maze_size}x{args.maze_size}" not in stem:
                stem += size_tag
        args.output = f"model/{stem}{steps_suffix}.onnx"

    # Load model
    model = load_model(args.model)

    # Export
    onnx_path = export_to_onnx(
        model=model,
        puzzle_type=puzzle_type,
        output_path=args.output,
        num_reasoning_steps=args.reasoning_steps,
        seq_len=seq_len,
        opset_version=args.opset,
    )

    # Verify
    if not args.skip_verify:
        verify_onnx(
            onnx_path=onnx_path,
            pytorch_model=model,
            puzzle_type=puzzle_type,
            num_reasoning_steps=args.reasoning_steps,
            seq_len=seq_len,
        )

    # Print summary
    steps = args.reasoning_steps or model.config.num_reasoning_steps
    file_size = os.path.getsize(onnx_path) / 1024**2

    print(f"\n{'=' * 55}")
    print(f"  EXPORT COMPLETE")
    print(f"{'=' * 55}")
    print(f"  ONNX file:   {onnx_path}")
    print(f"  File size:   {file_size:.1f} MB")
    print(f"  Puzzle:      {args.puzzle}")
    if args.puzzle == "maze":
        print(f"  Maze size:   {args.maze_size}x{args.maze_size}")
    print(f"  Seq length:  {seq_len}")
    print(f"  Steps:       {steps}")
    print(f"  Feedback:    {'Off (maze)' if puzzle_type == PuzzleType.MAZE else 'On'}")
    print(f"{'=' * 55}")
    print(f"\nRun on Raspberry Pi 5:")
    print(f"  pip install onnxruntime")
    print(f"  python scripts/solve_onnx.py \\")
    print(f"      --model {onnx_path} --puzzle {args.puzzle}")
    if args.puzzle == "maze":
        print(f"      --maze-size {args.maze_size}")


if __name__ == "__main__":
    main()
