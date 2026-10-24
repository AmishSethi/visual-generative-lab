#!/usr/bin/env python
"""Aggregate the single-skill baseline over every available seed.

Unlike `paper_runs/table2/aggregate_table2.py`, which is fixed to the Table 2 seeds,
this reads whatever seeds exist. Size and position read the Table 2 baseline and
rotation the Table 2 rotation baseline (baseline_nz). Count reads the 1000-epoch
random-layout comparator of Appendices D.3 and D.4 (radius-8 circles at random
positions; eval_family.py --family count_random), not the Table 2 count runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_runs.appendix.manifest import EVAL_ROOT, TABLE2_ROOT

# skill -> (variant, directory holding seed_*/results.json)
SOURCES = {
    "size": ("baseline", TABLE2_ROOT / "eval" / "samples20_steps250_cfg1" / "size" / "baseline"),
    "position": ("baseline", TABLE2_ROOT / "eval" / "samples20_steps250_cfg1" / "position" / "baseline"),
    "rotation": ("baseline_nz", TABLE2_ROOT / "eval" / "samples20_steps250_cfg1" / "rotation" / "baseline_nz"),
    "count": ("count_random", EVAL_ROOT / "count_random" / "count"),
}


def collect(base):
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
                        help="Read size, position and rotation from this Table 2 evaluation run "
                             "(a directory under MORE_SEEDS/table2/eval) instead of samples20_steps250_cfg1.")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "baseline_all_seeds.json")
    args = parser.parse_args()

    payload = {}
    print(f"{'skill':>9} {'n':>3} {'split':>7} {'measured':>16}")
    print("-" * 38)
    for skill, (variant, base) in SOURCES.items():
        if args.eval_run and skill != "count":
            base = TABLE2_ROOT / "eval" / args.eval_run / skill / variant
        per_seed = collect(base)
        if not per_seed:
            print(f"{skill:>9}   -  (no results)")
            continue
        payload[skill] = {"variant": variant, "source": str(base), "seeds": sorted(per_seed),
                          "per_seed": per_seed, "summary": {}}
        for split in ("train", "interp", "extra"):
            if skill == "count" and split == "interp":
                continue  # count has no interpolation cell: its interpolation queries are non-integer midpoints
            values = [v[split] for v in per_seed.values() if split in v]
            if not values:
                continue
            mean, std = float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            payload[skill]["summary"][split] = {"mean": mean, "std": std, "n": len(values)}
            print(f"{skill:>9} {len(values):>3} {split:>7} {mean:>9.1f} ± {std:<4.1f}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
