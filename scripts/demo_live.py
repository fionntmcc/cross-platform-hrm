# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Realistic Demo — Generate fresh Sudoku puzzles and solve them with the trained HRM.

Generates brand-new puzzles (not from the training set), feeds them to the model,
then validates and displays results with a compact progress table and rich summary.

Usage:
    python scripts/demo_live.py
    python scripts/demo_live.py --difficulty medium --count 100
    python scripts/demo_live.py --difficulty easy --count 50 --show-grids 3
    python scripts/demo_live.py --model model/unified_hrm_sudoku_9x9_best.pt --difficulty hard
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.data.sudoku_generator import generate_sudoku_dataset
from hrm.prototype.models.model_unified import PuzzleType, UnifiedHRM

# ── ANSI helpers ──────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
RESET  = "\033[0m"
WHITE  = "\033[97m"


def cell(val, is_given, is_correct):
    v = str(int(val))
    if is_given:
        return f"{BOLD}{v}{RESET}"
    return f"{GREEN}{v}{RESET}" if is_correct else f"{RED}{v}{RESET}"


def print_9x9(grid, given_mask, solution=None, title=""):
    if title:
        print(f"    {title}")
    print("    ┌───────┬───────┬───────┐")
    for i in range(9):
        c = [cell(grid[i,j], given_mask[i,j],
                   solution is None or grid[i,j]==solution[i,j]) for j in range(9)]
        print(f"    │ {c[0]} {c[1]} {c[2]} │ {c[3]} {c[4]} {c[5]} │ {c[6]} {c[7]} {c[8]} │")
        if i in (2, 5):
            print("    ├───────┼───────┼───────┤")
    print("    └───────┴───────┴───────┘")


def print_side_by_side(pred, truth, given_mask, solution):
    """Print model prediction and ground truth side by side."""
    def row_str(grid, r):
        c = [cell(grid[r,j], given_mask[r,j],
                   solution is None or grid[r,j]==solution[r,j]) for j in range(9)]
        return f"│ {c[0]} {c[1]} {c[2]} │ {c[3]} {c[4]} {c[5]} │ {c[6]} {c[7]} {c[8]} │"

    def truth_row(grid, r):
        c = [cell(grid[r,j], given_mask[r,j], True) for j in range(9)]
        return f"│ {c[0]} {c[1]} {c[2]} │ {c[3]} {c[4]} {c[5]} │ {c[6]} {c[7]} {c[8]} │"

    top  = "┌───────┬───────┬───────┐"
    mid  = "├───────┼───────┼───────┤"
    bot  = "└───────┴───────┴───────┘"
    gap  = "      "

    print(f"    {'Model prediction':<27}{gap}{'Ground truth'}")
    print(f"    {top}{gap}{top}")
    for i in range(9):
        print(f"    {row_str(pred, i)}{gap}{truth_row(truth, i)}")
        if i in (2, 5):
            print(f"    {mid}{gap}{mid}")
    print(f"    {bot}{gap}{bot}")


def progress_bar(fraction, width=30):
    """Render an inline ASCII progress bar."""
    filled = int(fraction * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fraction:5.1%}"


# ── Validation ────────────────────────────────────────────────────────────────

def check_sudoku(sol):
    """Return (is_valid, row_viol, col_viol, box_viol) for a 9x9 grid."""
    expected = set(range(1, 10))
    row_v = col_v = box_v = 0
    for i in range(9):
        if set(int(x) for x in sol[i]) != expected:
            row_v += 1
        if set(int(x) for x in sol[:, i]) != expected:
            col_v += 1
    for bi in range(3):
        for bj in range(3):
            block = sol[bi*3:(bi+1)*3, bj*3:(bj+1)*3]
            if set(int(x) for x in block.flatten()) != expected:
                box_v += 1
    total = row_v + col_v + box_v
    return total == 0, row_v, col_v, box_v


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Live Sudoku demo — generate fresh puzzles and solve with trained HRM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="model/unified_hrm_sudoku_9x9_best.pt",
                    help="Path to trained model checkpoint")
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"],
                    help="Difficulty of generated puzzles")
    p.add_argument("--count", type=int, default=10,
                    help="Number of puzzles to generate and solve")
    p.add_argument("--show-grids", type=int, default=3,
                    help="Max number of grids to display (0=none). Best/worst always shown.")
    p.add_argument("--show-all", action="store_true",
                    help="Print grids for every puzzle (overrides --show-grids)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible puzzle generation")
    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{CYAN}{'═'*62}")
    print("  Hierarchical Reasoning Model (HRM) — Sudoku Solver Demo")
    print(f"{'═'*62}{RESET}")

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = PROJECT_ROOT / args.model
    if not ckpt_path.exists():
        print(f"\n  {RED}Error: checkpoint not found: {ckpt_path}{RESET}")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = ckpt.get("config")
    if config is None:
        from hrm.prototype.models.model_unified import UnifiedHRMConfig
        config = UnifiedHRMConfig()

    model = UnifiedHRM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_accuracy")

    print(f"\n  {BOLD}Model{RESET}")
    print(f"  ├─ Checkpoint  : {ckpt_path.name}")
    print(f"  ├─ Architecture: hidden={config.hidden_size}, heads={config.num_heads}, "
          f"layers_L={config.num_layers_L}, layers_H={config.num_layers_H}")
    print(f"  ├─ Parameters  : {model.num_parameters:,}")
    print(f"  ├─ Trained     : {epoch} epochs" + (f"  (best val acc {val_acc:.1%})" if val_acc else ""))
    print(f"  └─ Device      : {device}")

    # ── Generate fresh puzzles ────────────────────────────────────────────
    gen_seed = args.seed if args.seed is not None else int(time.time()) % 100000

    print(f"\n  {BOLD}Puzzle generation{RESET}")
    print(f"  ├─ Count       : {args.count}")
    print(f"  ├─ Difficulty  : {args.difficulty}")
    print("  ├─ Grid size   : 9×9")
    print(f"  └─ Seed        : {gen_seed}")

    t_gen = time.perf_counter()
    dataset = generate_sudoku_dataset(
        num_puzzles=args.count,
        grid_size=9,
        difficulty=args.difficulty,
        seed=gen_seed,
        verbose=False,
    )
    t_gen = time.perf_counter() - t_gen

    puzzles = np.array(dataset["problems"])
    solutions = np.array(dataset["solutions"])
    metadata = dataset["metadata"]

    empty_counts = [m["empty_cells"] for m in metadata]
    bt_counts = [m["backtracks"] for m in metadata]
    print(f"\n  Generated {args.count} puzzles in {t_gen:.2f}s")
    print(f"  Empty cells : min={min(empty_counts)}, max={max(empty_counts)}, "
          f"avg={np.mean(empty_counts):.1f}")
    print(f"  Backtracks  : min={min(bt_counts)}, max={max(bt_counts)}, "
          f"avg={np.mean(bt_counts):.1f}")

    # ── Inference ─────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Running inference …{RESET}", end=" ", flush=True)
    t0 = time.perf_counter()
    batch_size = 128
    all_preds = []
    with torch.no_grad():
        for start in range(0, args.count, batch_size):
            end = min(start + batch_size, args.count)
            batch = torch.from_numpy(puzzles[start:end]).long().view(end - start, -1).to(device)
            out = model.forward(batch, PuzzleType.SUDOKU_9X9)
            all_preds.append(out["predictions"].cpu().numpy().reshape(-1, 9, 9))
    preds = np.concatenate(all_preds, axis=0)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.3f}s  ({elapsed/args.count*1000:.1f} ms/puzzle)")

    # ── Analyse per-puzzle ────────────────────────────────────────────────
    results = []  # list of dicts per puzzle
    for i in range(args.count):
        puz, sol, pred = puzzles[i], solutions[i], preds[i]
        given = (puz != 0)
        empty = ~given

        display = pred.copy()
        display[given] = puz[given]

        n_empty = int(empty.sum())
        n_correct = int(((display == sol) & empty).sum())
        n_wrong = n_empty - n_correct
        cell_acc = n_correct / n_empty if n_empty > 0 else 1.0
        perfect = (n_correct == n_empty)
        is_valid, rv, cv, bv = check_sudoku(display)

        results.append(dict(
            idx=i, puzzle=puz, solution=sol, prediction=display,
            given_mask=given, empty_mask=empty,
            n_empty=n_empty, n_correct=n_correct, n_wrong=n_wrong,
            cell_acc=cell_acc, perfect=perfect, is_valid=is_valid,
            row_viol=rv, col_viol=cv, box_viol=bv,
            empties=metadata[i]["empty_cells"],
            backtracks=metadata[i]["backtracks"],
        ))

    n_solved = sum(r["perfect"] for r in results)
    n_valid  = sum(r["is_valid"] for r in results)
    cell_accs = [r["cell_acc"] for r in results]
    wrong_counts = [r["n_wrong"] for r in results]
    solved = [r for r in results if r["perfect"]]
    failed = [r for r in results if not r["perfect"]]

    # ── Compact results table ─────────────────────────────────────────────
    print(f"\n  {BOLD}Results{RESET}")
    print(f"  {'─'*58}")

    # Decide how many lines to show
    if args.count <= 30:
        # Show all puzzles in a compact table
        print(f"  {'#':>3}  {'Empty':>5}  {'Correct':>7}  {'Wrong':>5}  {'Cell%':>6}  Status")
        print(f"  {'─'*3}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*22}")
        for r in results:
            if r["perfect"] and r["is_valid"]:
                status = f"{GREEN}SOLVED ✓{RESET}"
            elif r["perfect"]:
                status = f"{YELLOW}correct, invalid grid{RESET}"
            elif r["cell_acc"] >= 0.9:
                status = f"{YELLOW}✗ {r['n_wrong']} wrong{RESET}"
            else:
                status = f"{RED}✗ {r['n_wrong']} wrong{RESET}"
            print(f"  {r['idx']+1:>3}  {r['n_empty']:>5}  {r['n_correct']:>7}  "
                  f"{r['n_wrong']:>5}  {r['cell_acc']:>5.1%}  {status}")
    else:
        # Large batch — show a progress-style summary instead of per-puzzle
        # Group by accuracy buckets
        buckets = {"100% (solved)": 0, "90-99%": 0, "80-89%": 0,
                   "50-79%": 0, "<50%": 0}
        for r in results:
            a = r["cell_acc"]
            if a >= 1.0 - 1e-9:
                buckets["100% (solved)"] += 1
            elif a >= 0.9:
                buckets["90-99%"] += 1
            elif a >= 0.8:
                buckets["80-89%"] += 1
            elif a >= 0.5:
                buckets["50-79%"] += 1
            else:
                buckets["<50%"] += 1

        max_bar = max(buckets.values()) if max(buckets.values()) > 0 else 1
        print(f"\n  {'Accuracy bucket':<18} {'Count':>6}  Distribution")
        print(f"  {'─'*18} {'─'*6}  {'─'*32}")
        colours = [GREEN, GREEN, YELLOW, YELLOW, RED]
        for (label, cnt), col in zip(buckets.items(), colours):
            bar_len = int(cnt / max_bar * 25)
            bar = "█" * bar_len
            pct = cnt / args.count * 100
            print(f"  {label:<18} {cnt:>5}   {col}{bar}{RESET} {pct:.0f}%")

    # ── Show all grids if requested ────────────────────────────────────
    if args.show_all:
        print(f"\n  {BOLD}All puzzles{RESET}  "
              f"(legend: {BOLD}bold{RESET}=given  {GREEN}green{RESET}=correct  {RED}red{RESET}=wrong)")
        for r in results:
            tag = f"{GREEN}SOLVED ✓{RESET}" if r['perfect'] else \
                  f"{RED}{r['n_correct']}/{r['n_empty']} cells correct ({r['cell_acc']:.0%}){RESET}"
            print(f"\n  {CYAN}── Puzzle {r['idx']+1}{RESET}  "
                  f"({r['n_empty']} empty, {r['backtracks']} backtracks) — {tag}")
            if r['perfect']:
                print_9x9(r['prediction'], r['given_mask'], r['solution'])
            else:
                print_side_by_side(r['prediction'], r['solution'],
                                   r['given_mask'], r['solution'])

    # ── Show selected grids ───────────────────────────────────────────
    n_grids = args.show_grids
    if not args.show_all and n_grids > 0:
        # Always include: best solved, worst failure
        sorted_by_acc = sorted(results, key=lambda r: r["cell_acc"])
        to_show = []
        show_indices = set()

        # Best solved puzzle (fewest givens among perfect)
        if solved:
            hardest_solved = max(solved, key=lambda r: r["n_empty"])
            if hardest_solved["idx"] not in show_indices:
                to_show.append(("Hardest puzzle solved correctly", hardest_solved))
                show_indices.add(hardest_solved["idx"])

        # Closest miss (highest accuracy among failures)
        if failed:
            closest = max(failed, key=lambda r: r["cell_acc"])
            if closest["idx"] not in show_indices:
                to_show.append(("Closest miss", closest))
                show_indices.add(closest["idx"])

        # Worst failure
        if failed:
            worst = min(failed, key=lambda r: r["cell_acc"])
            if worst["idx"] not in show_indices:
                to_show.append(("Worst failure", worst))
                show_indices.add(worst["idx"])

        # Fill remaining slots with random solved examples
        remaining = n_grids - len(to_show)
        if remaining > 0 and solved:
            extras = [r for r in solved if r["idx"] not in show_indices]
            np.random.shuffle(extras)
            for r in extras[:remaining]:
                to_show.append(("Solved example", r))
                show_indices.add(r["idx"])

        print(f"\n  {BOLD}Selected grids{RESET}  "
              f"(legend: {BOLD}bold{RESET}=given  {GREEN}green{RESET}=correct  {RED}red{RESET}=wrong)")
        for label, r in to_show:
            tag = f"{GREEN}SOLVED ✓{RESET}" if r["perfect"] else \
                  f"{RED}{r['n_correct']}/{r['n_empty']} cells correct ({r['cell_acc']:.0%}){RESET}"
            print(f"\n  {CYAN}── Puzzle {r['idx']+1}: {label}{RESET}  "
                  f"({r['n_empty']} empty, {r['backtracks']} backtracks) — {tag}")

            if r["perfect"]:
                print_9x9(r["prediction"], r["given_mask"], r["solution"])
            else:
                print_side_by_side(r["prediction"], r["solution"],
                                   r["given_mask"], r["solution"])

    # ── Summary statistics ────────────────────────────────────────────────
    avg_cell = np.mean(cell_accs)
    median_cell = np.median(cell_accs)
    std_cell = np.std(cell_accs)
    total_wrong = sum(wrong_counts)
    total_empty = sum(r["n_empty"] for r in results)
    total_violations = sum(r["row_viol"] + r["col_viol"] + r["box_viol"] for r in results)

    # Accuracy vs empty-cell count correlation
    accs_arr = np.array(cell_accs)
    empties_arr = np.array([r["n_empty"] for r in results])

    print(f"\n{CYAN}{'═'*62}")
    print(f"  SUMMARY — {args.count} {args.difficulty} 9×9 Sudoku puzzles (unseen)")
    print(f"{'═'*62}{RESET}")

    print(f"\n  {BOLD}Solve rate{RESET}")
    print(f"  Puzzles solved perfectly : {BOLD}{n_solved}/{args.count}{RESET}  "
          f"{progress_bar(n_solved/args.count)}")
    print(f"  Valid Sudoku grids       : {n_valid}/{args.count}  "
          f"{progress_bar(n_valid/args.count)}")

    print(f"\n  {BOLD}Cell accuracy{RESET} (empty cells only)")
    print(f"  Mean   : {avg_cell:.2%}")
    print(f"  Median : {median_cell:.2%}")
    print(f"  Std    : {std_cell:.2%}")
    print(f"  Total  : {total_empty - total_wrong}/{total_empty} cells correct")

    if failed:
        wrong_on_fail = [r["n_wrong"] for r in failed]
        print(f"\n  {BOLD}Error analysis{RESET} (on {len(failed)} unsolved puzzles)")
        print(f"  Avg wrong cells   : {np.mean(wrong_on_fail):.1f}")
        print(f"  Median wrong      : {np.median(wrong_on_fail):.0f}")
        print(f"  Max wrong         : {max(wrong_on_fail)}")
        print(f"  Total constraint  : {total_violations} violations "
              f"(row/col/box across all puzzles)")

    # Performance by puzzle difficulty (empty cell count)
    if args.count >= 10:
        lo = np.percentile(empties_arr, 33)
        hi = np.percentile(empties_arr, 66)
        groups = {
            f"Fewer empties (≤{int(lo)})": [r for r in results if r["n_empty"] <= lo],
            f"Medium empties ({int(lo)+1}-{int(hi)})": [r for r in results if lo < r["n_empty"] <= hi],
            f"More empties (>{int(hi)})": [r for r in results if r["n_empty"] > hi],
        }
        print(f"\n  {BOLD}Accuracy by puzzle complexity{RESET}")
        print(f"  {'Group':<30} {'Cnt':>4}  {'Solve%':>6}  {'Cell Acc':>8}")
        print(f"  {'─'*30} {'─'*4}  {'─'*6}  {'─'*8}")
        for label, grp in groups.items():
            if not grp:
                continue
            gc = len(grp)
            gs = sum(r["perfect"] for r in grp)
            ga = np.mean([r["cell_acc"] for r in grp])
            print(f"  {label:<30} {gc:>4}  {gs/gc:>5.0%}  {ga:>7.1%}")

    print(f"\n  {BOLD}Performance{RESET}")
    print(f"  Generation : {t_gen:.2f}s  ({t_gen/args.count*1000:.0f} ms/puzzle)")
    print(f"  Inference  : {elapsed:.3f}s  ({elapsed/args.count*1000:.1f} ms/puzzle)")
    print(f"  Throughput : {args.count/elapsed:.0f} puzzles/sec")
    print()


if __name__ == "__main__":
    main()
