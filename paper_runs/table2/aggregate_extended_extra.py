#!/usr/bin/env python
from __future__ import annotations

import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")

"""Aggregate extended-OOD extra metrics across variants and seeds.

For position: re-aggregates from v1 results.json by filtering to L-infinity distance >= min_margin.
              No new compute required (the new extras are a subset of the existing ones).
For size: reads v2 results.json (samples20_steps250_cfg1_extended) when present, falls back to v1.
"""

import argparse
import json
import math
import statistics
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.manifest import EVAL_ROOT


SIZE_VARIANTS = ["baseline", "sinusoidal", "rotary", "adaln", "vae", "flow", "unet"]
POSITION_VARIANTS = ["baseline", "sinusoidal", "rotary", "adaln", "vae", "flow", "unet"]
SEEDS = [0, 1, 2]

POSITION_TRAIN_BOUND = 18.0  # |x| <= 18 and |y| <= 18 is training
POSITION_MIN_MARGIN = 4.0    # extra_far = positions with L-inf distance >= 4 px outside training


def linf_dist(x: float, y: float, bound: float = POSITION_TRAIN_BOUND) -> float:
    return max(abs(x) - bound, abs(y) - bound, 0.0)


def aggregate_position(eval_root: Path, min_margin: float = POSITION_MIN_MARGIN):
    out = {}
    for variant in POSITION_VARIANTS:
        per_seed = []
        for seed in SEEDS:
            path = eval_root / "position" / variant / f"seed_{seed}" / "results.json"
            if not path.exists():
                continue
            with open(path) as f:
                r = json.load(f)
            extra = r["paper_metrics"]["splits"]["extra"]
            cond = extra["condition_accuracies"]
            far_accs = []
            far_errs = []
            for key, data in cond.items():
                xs, ys = key.split(",")
                x, y = float(xs), float(ys)
                if linf_dist(x, y) + 1e-6 >= min_margin:
                    far_accs.append(data["accuracy"])
                    if data.get("detection_rate", 0) > 0 and not math.isinf(data.get("mean_position_error", float("inf"))):
                        far_errs.append(data["mean_position_error"])
            if not far_accs:
                continue
            per_seed.append({
                "seed": seed,
                "accuracy": sum(far_accs) / len(far_accs),
                "mean_position_error": sum(far_errs) / len(far_errs) if far_errs else float("inf"),
                "num_conditions": len(far_accs),
            })
        if per_seed:
            accs = [s["accuracy"] for s in per_seed]
            out[variant] = {
                "per_seed": per_seed,
                "mean_accuracy": statistics.mean(accs),
                "std_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "n_seeds": len(per_seed),
            }
    return out


def aggregate_size(extended_root: Path, fallback_root: Path = None):
    out = {}
    for variant in SIZE_VARIANTS:
        per_seed = []
        for seed in SEEDS:
            path = extended_root / "size" / variant / f"seed_{seed}" / "results.json"
            source = "extended"
            if not path.exists() and fallback_root is not None:
                path = fallback_root / "size" / variant / f"seed_{seed}" / "results.json"
                source = "v1_fallback"
            if not path.exists():
                continue
            with open(path) as f:
                r = json.load(f)
            extra = r["paper_metrics"]["splits"]["extra"]
            per_seed.append({
                "seed": seed,
                "accuracy": extra["accuracy"],
                "mean_iou": extra["mean_iou"],
                "num_conditions": extra["num_conditions"],
                "source": source,
                "size_extrap_max": r.get("size_extrap_max", 25),
            })
        if per_seed:
            accs = [s["accuracy"] for s in per_seed]
            out[variant] = {
                "per_seed": per_seed,
                "mean_accuracy": statistics.mean(accs),
                "std_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "n_seeds": len(per_seed),
                "all_extended": all(s["source"] == "extended" for s in per_seed),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, default=EVAL_ROOT / "samples20_steps250_cfg1")
    parser.add_argument("--extended-root", type=Path, default=EVAL_ROOT / "samples20_steps250_cfg1_extended")
    parser.add_argument("--output", type=Path, default=Path(VGL_ROOT) / "paper_runs/table2/extended_extra_aggregate.json")
    args = parser.parse_args()

    position = aggregate_position(args.v1_root, min_margin=POSITION_MIN_MARGIN)
    size = aggregate_size(args.extended_root, fallback_root=args.v1_root)

    print(f"\n=== Position extra_far (>={POSITION_MIN_MARGIN} px outside [-{POSITION_TRAIN_BOUND}, {POSITION_TRAIN_BOUND}]^2) ===")
    print(f"{'variant':<12} {'mean':>8} {'std':>8} {'seeds':>6} {'per-seed acc':>30}")
    for variant, agg in position.items():
        per_seed_str = " ".join(f"{s['accuracy']:.1f}" for s in agg["per_seed"])
        print(f"{variant:<12} {agg['mean_accuracy']:>8.1f} {agg['std_accuracy']:>8.1f} {agg['n_seeds']:>6} {per_seed_str:>30}")

    print(f"\n=== Size extra (extended: r in [1..4, 21..30]) ===")
    print(f"{'variant':<12} {'mean':>8} {'std':>8} {'seeds':>6} {'all_ext':>8} {'per-seed acc':>30}")
    for variant, agg in size.items():
        per_seed_str = " ".join(f"{s['accuracy']:.1f}" for s in agg["per_seed"])
        all_ext = "yes" if agg["all_extended"] else "no"
        print(f"{variant:<12} {agg['mean_accuracy']:>8.1f} {agg['std_accuracy']:>8.1f} {agg['n_seeds']:>6} {all_ext:>8} {per_seed_str:>30}")

    with open(args.output, "w") as f:
        json.dump({
            "position_min_margin": POSITION_MIN_MARGIN,
            "size_extrap_range": [1, 4, 21, 30],
            "position": position,
            "size": size,
        }, f, indent=2, default=str)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
