#!/usr/bin/env python
"""Aggregate the single-skill baseline over every available seed (yLdq w3).

`paper_runs/table2/aggregate_table2.py` is fixed to seeds 0-2; this reads
whatever seeds exist so the baseline row can be reported at n = 10 for all four
skills, matching the seed count the rotation row already had.

It also prints the paper's Table 2 values alongside, so any cell that does not
reproduce is visible rather than silently replaced.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_runs.rebuttal.manifest import EVAL_ROOT, TABLE2_ROOT

# What the submitted Table 2 prints for the DiT-S/2 baseline row.
PAPER_TABLE2_BASELINE = {
    "size": {"train": 100.0, "interp": 96.8, "extra": 47.3},
    "position": {"train": 100.0, "interp": 97.4, "extra": 31.0},
    "rotation": {"train": 100.0, "interp": 100.0, "extra": 68.0},
    # 91.4 per the submitted PDF. The local Experimental_Setup.tex reads 51.4, so the
    # two disagree; the submission is authoritative. Our 10-seed exact-match measurement
    # is 51.8 (watershed) / 56.7 (Hough), and watershed within +-1 is 92.8, so the
    # published cell is likely a tolerance-based metric rather than exact match.
    "count": {"train": 91.4, "interp": 0.0, "extra": 15.4},
}

# Each paper cell was produced by a specific evaluation run, and reading the
# wrong one manufactures differences that are not there: size and position use
# the extended extrapolation grids, rotation uses the 5-degree run, and rotation
# uses the learnable-null variant that carries ten seeds.
SOURCES = {
    "size": ("baseline", "samples20_steps250_cfg1_extended"),
    "position": ("baseline", "samples20_steps250_cfg1_extended"),
    "rotation": ("baseline_nz", "samples20_steps250_cfg1_rot5deg"),
    "count": ("baseline", "samples20_steps250_cfg1"),
}


def collect(skill, variant, eval_run):
    base = TABLE2_ROOT / "eval" / eval_run / skill / variant
    out = {}
    for seed_dir in sorted(base.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        path = seed_dir / "results.json"
        if not path.exists():
            continue
        splits = json.loads(path.read_text())["paper_metrics"]["splits"]
        out[int(seed_dir.name.split("_")[1])] = {
            k: float(v["accuracy"]) for k, v in splits.items()
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline over all seeds.")
    parser.add_argument("--eval-run", default=None,
                        help="Override the per-skill source run (default: the run that "
                             "produced each paper cell).")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "baseline_all_seeds.json")
    args = parser.parse_args()

    payload = {}
    print(f"{'skill':>9} {'n':>3} {'split':>7} {'measured':>16} {'paper Table 2':>14} {'delta':>8}")
    print("-" * 62)
    for skill, (variant, default_run) in SOURCES.items():
        run = args.eval_run or default_run
        per_seed = collect(skill, variant, run)
        if not per_seed:
            print(f"{skill:>9}   -  (no results yet)")
            continue
        payload[skill] = {"variant": variant, "eval_run": run, "seeds": sorted(per_seed),
                          "per_seed": per_seed, "summary": {}}
        for split in ("train", "interp", "extra"):
            values = [v[split] for v in per_seed.values() if split in v]
            if not values:
                continue
            mean, std = float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            paper = PAPER_TABLE2_BASELINE[skill][split]
            payload[skill]["summary"][split] = {"mean": mean, "std": std, "n": len(values),
                                                "paper_table2": paper, "delta": mean - paper}
            flag = "  <-- differs" if abs(mean - paper) > 2.0 else ""
            print(f"{skill:>9} {len(values):>3} {split:>7} {mean:>9.1f} ± {std:<4.1f} "
                  f"{paper:>14.1f} {mean - paper:>+8.1f}{flag}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
