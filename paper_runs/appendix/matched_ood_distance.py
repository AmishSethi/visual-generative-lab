#!/usr/bin/env python
"""Re-score extrapolation at matched normalised out-of-distribution distance.

Question: is rotation still easier to extrapolate once the numerical scale
and extrapolation distance of every continuous condition are controlled for?

The paper's per-skill extrapolation sets are not matched in normalised terms.
For a query q we define

    delta(q) = (distance from q to the nearest training boundary) / (training span)

with spans  size 15 px, position 36 px per axis, rotation 270 deg, count 5.
Because rotation's held-out band is only 90 deg wide, no rotation query is more
than 45 deg -- delta = 0.167 -- outside training, whereas size is evaluated out
to delta = 0.67.  Comparing raw column averages therefore compares different
extrapolation distances.  This script re-bins every skill onto a common delta
grid using the per-condition accuracies already stored by the paper evaluator,
so no GPU work is required.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.manifest import EVAL_ROOT, TABLE2_ROOT

# (variant, eval-run directory) chosen per skill: the extended runs carry the
# wider size/position extrapolation grids; rotation's 10-seed baseline is the
# null-embedding=learnable variant used for the paper's Table 1 rotation row.
SKILL_SOURCES = {
    "size": ("baseline", "samples20_steps250_cfg1_extended"),
    "position": ("baseline", "samples20_steps250_cfg1_extended"),
    # Must be the 5-degree run: that is the threshold the paper's rotation cell uses.
    # The 2-degree run scores rotation more strictly than every other skill in this
    # table and is not comparable to the published rotation number.
    "rotation": ("baseline_nz", "samples20_steps250_cfg1_rot5deg"),
    "count": ("baseline", "samples20_steps250_cfg1"),
}

SPANS = {"size": 15.0, "position": 36.0, "rotation": 270.0, "count": 5.0}

DELTA_BINS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
              (0.20, 0.30), (0.30, 0.50), (0.50, 1.01)]


def parse_condition(skill, key):
    if skill == "position":
        x, y = key.split(",")
        return (float(x), float(y))
    return float(key)


def normalised_distance(skill, value):
    """Distance outside the training support, divided by the training span."""
    if skill == "size":
        excess = max(5.0 - value, value - 20.0, 0.0)
    elif skill == "count":
        excess = max(2.0 - value, value - 7.0, 0.0)
    elif skill == "position":
        x, y = value
        excess = max(max(abs(x), abs(y)) - 18.0, 0.0)
    elif skill == "rotation":
        angle = value % 360.0
        if 45.0 <= angle <= 315.0:
            excess = 0.0
        else:
            # circular distance to the [45, 315] arc
            excess = min((45.0 - angle) % 360.0, (angle - 315.0) % 360.0)
    else:
        raise ValueError(skill)
    return excess / SPANS[skill]


def load_skill(skill, root):
    variant, run = SKILL_SOURCES[skill]
    base = Path(root) / "eval" / run / skill / variant
    per_seed = []
    for seed_dir in sorted(base.glob("seed_*")):
        results = seed_dir / "results.json"
        if not results.exists():
            continue
        splits = json.loads(results.read_text())["paper_metrics"]["splits"]
        merged = {}
        for split_name in ("extra", "interp"):
            for key, acc in splits.get(split_name, {}).get("condition_accuracies", {}).items():
                # Evaluators store either a bare accuracy or a per-condition dict.
                value = acc["accuracy"] if isinstance(acc, dict) else acc
                merged[parse_condition(skill, key)] = float(value)
        if merged:
            per_seed.append((seed_dir.name, merged))
    return variant, run, per_seed


def bin_accuracies(skill, per_seed):
    """Mean accuracy per delta bin, averaged within seed then across seeds."""
    binned = defaultdict(list)
    for _, conditions in per_seed:
        per_bin = defaultdict(list)
        for value, acc in conditions.items():
            delta = normalised_distance(skill, value)
            if delta <= 0:
                continue
            for lo, hi in DELTA_BINS:
                if lo < delta <= hi:
                    per_bin[(lo, hi)].append(acc)
                    break
        for bin_key, accs in per_bin.items():
            binned[bin_key].append(float(np.mean(accs)))
    return binned


def main():
    parser = argparse.ArgumentParser(description="Matched normalised OOD-distance rescoring.")
    parser.add_argument("--root", default=str(TABLE2_ROOT))
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "matched_ood_distance.json")
    args = parser.parse_args()

    payload = {"spans": SPANS, "definition": "delta = distance outside training support / training span",
               "skills": {}}

    all_binned = {}
    for skill in ("size", "position", "rotation", "count"):
        variant, run, per_seed = load_skill(skill, args.root)
        if not per_seed:
            print(f"[warn] no results for {skill} ({variant}, {run})")
            continue
        binned = bin_accuracies(skill, per_seed)
        all_binned[skill] = binned
        max_delta = max(
            normalised_distance(skill, v)
            for _, conds in per_seed for v in conds
        )
        payload["skills"][skill] = {
            "variant": variant, "eval_run": run, "num_seeds": len(per_seed),
            "max_delta_evaluated": max_delta,
            "bins": {f"{lo:.2f}-{hi:.2f}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
                     for (lo, hi), v in sorted(binned.items())},
        }

    for label, reduce_fn in (("mean +- std over seeds", None), ("best seed", max)):
        header = f"{'delta bin':>12} " + "".join(f"{s:>17}" for s in all_binned)
        print(f"\nAccuracy (%) at matched normalised OOD distance -- {label}")
        print(header)
        print("-" * len(header))
        for lo, hi in DELTA_BINS:
            row = f"{lo:.2f}-{hi:.2f}".rjust(12) + " "
            for skill in all_binned:
                vals = all_binned[skill].get((lo, hi))
                if not vals:
                    row += f"{'--':>17}"
                elif reduce_fn is None:
                    row += f"{np.mean(vals):8.1f}+-{np.std(vals):<7.1f}"
                else:
                    row += f"{reduce_fn(vals):>17.1f}"
            print(row)

    # Rotation's spread is the documented seed trimodality, so record the
    # per-seed curve too rather than hiding it inside a standard deviation.
    for skill in all_binned:
        payload["skills"][skill]["per_seed_bins"] = {
            f"{lo:.2f}-{hi:.2f}": vals for (lo, hi), vals in sorted(all_binned[skill].items())
        }

    print("\nLargest normalised distance each skill is actually evaluated at:")
    for skill, info in payload["skills"].items():
        print(f"    {skill:>9}: delta_max = {info['max_delta_evaluated']:.3f}  ({info['num_seeds']} seeds)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
