#!/usr/bin/env python
"""
Aggregate Table 3 evaluation results across 3 seeds.
Outputs mean ± std for each (pair, coverage, category, skill) entry.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table3.manifest import (
    CANONICAL_SEEDS,
    COVERAGES,
    EVAL_ROOT,
    SKILL_PAIRS,
)


def load_results(eval_root: Path, pair_name: str, coverage: int, seed: int) -> dict | None:
    path = eval_root / pair_name / f"cov{coverage}" / f"seed_{seed}" / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate Table 3 results across seeds.")
    parser.add_argument("--eval-root", type=Path,
                        default=EVAL_ROOT / "samples20_steps250_cfg1")
    parser.add_argument("--pairs", nargs="+", default=list(SKILL_PAIRS.keys()))
    parser.add_argument("--coverages", nargs="+", type=int, default=list(COVERAGES))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Table 3 column order from the paper
    all_skills = ["shape", "color", "position", "size", "count"]

    rows = []
    missing = []

    for pair_name in args.pairs:
        pair_spec = SKILL_PAIRS.get(pair_name)
        if pair_spec is None:
            print(f"Warning: unknown pair {pair_name}, skipping")
            continue
        properties = pair_spec["properties"]
        # Map property names to table columns
        prop_to_col = {}
        for p in properties:
            if p == "radius":
                prop_to_col[p] = "size"
            else:
                prop_to_col[p] = p

        for coverage in args.coverages:
            seed_results = []
            for seed in CANONICAL_SEEDS:
                result = load_results(args.eval_root, pair_name, coverage, seed)
                if result is None:
                    missing.append(f"{pair_name}/cov{coverage}/seed_{seed}")
                    continue
                seed_results.append(result)

            if not seed_results:
                continue

            # Collect per-split accuracies across seeds
            # Identify the "B" split (metadata_unseen or test_unseen)
            split_accs = defaultdict(lambda: defaultdict(list))

            for sr in seed_results:
                for split_name, split_data in sr.get("splits", {}).items():
                    accs = split_data.get("accuracies", {})
                    for prop, acc in accs.items():
                        if acc is not None:
                            col = prop_to_col.get(prop, prop)
                            split_accs[split_name][col].append(acc)

            # Build row for this (pair, coverage)
            # Map split names to paper categories
            for split_name, skill_accs in split_accs.items():
                category = split_name
                if "unseen" in split_name.lower() or split_name == "metadata_unseen":
                    category = "B"
                elif split_name == "train" or split_name == "metadata_train":
                    category = "A"
                elif "test_" in split_name:
                    category = split_name.replace("test_", "")

                row = {
                    "pair": pair_name,
                    "coverage": coverage,
                    "category": category,
                    "n_seeds": len(seed_results),
                }
                for skill in all_skills:
                    vals = skill_accs.get(skill, [])
                    if vals:
                        row[f"{skill}_mean"] = float(np.mean(vals))
                        row[f"{skill}_std"] = float(np.std(vals))
                    else:
                        row[f"{skill}_mean"] = None
                        row[f"{skill}_std"] = None
                rows.append(row)

    # Print table
    print("\n" + "=" * 120)
    print("TABLE 3: Compositional generalization (mean ± std across seeds)")
    print("=" * 120)
    header = f"{'Pair':<20} {'Cov':>4} {'Cat':<15}"
    for skill in all_skills:
        header += f" {skill:>15}"
    print(header)
    print("-" * 120)

    current_pair = None
    for row in rows:
        if row["pair"] != current_pair:
            if current_pair is not None:
                print()
            current_pair = row["pair"]

        line = f"{row['pair']:<20} {row['coverage']:>3}% {row['category']:<15}"
        for skill in all_skills:
            mean = row.get(f"{skill}_mean")
            std = row.get(f"{skill}_std")
            if mean is not None:
                if row["n_seeds"] > 1 and std is not None:
                    line += f" {mean:>6.1f}±{std:>4.1f}"
                else:
                    line += f" {mean:>6.1f}     "
            else:
                line += f" {'–':>11}"
        print(line)

    if missing:
        print(f"\n--- Missing evaluations ({len(missing)}): ---")
        for m in missing:
            print(f"  {m}")

    # Optionally save as JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"rows": rows, "missing": missing}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
