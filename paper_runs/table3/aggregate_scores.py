#!/usr/bin/env python
"""Aggregate score_table3.py results into the cells of Table 3.

Reads <scores>/<pair>_cov<c>_seed<s>.json for every seed in TABLE3_SEEDS. A is the mean over the
pair's two skills of the train accuracy (each a mean over seeds). B is test_compositional; at 75%
coverage, pairs with out-of-range values also get C(<skill>) (test_prop_a_ood, test_prop_b_ood:
that skill out of range) and D (test_both_ood). Each cell is the mean and sample standard
deviation over seeds of the per-sample accuracy, printed rounded half up to one decimal as in
Table 3; the JSON written by --output holds the unrounded values.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table3.manifest import ARTIFACT_ROOT, COVERAGES, SKILL_PAIRS, TABLE3_SEEDS

SKILL_NAMES = {"radius": "size"}


def mean_std(values):
    """Exact decimal mean and sample standard deviation of the per-seed values."""
    exact = [Decimal(repr(float(v))) for v in values]
    return statistics.mean(exact), (statistics.stdev(exact) if len(exact) > 1 else Decimal(0))


def half_up(value):
    """One decimal, rounded half up (0.25 -> 0.3)."""
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description="Aggregate score_table3.py results over seeds.")
    parser.add_argument("--scores", type=Path, default=ARTIFACT_ROOT / "scores",
                        help="directory of <pair>_cov<c>_seed<s>.json files")
    parser.add_argument("--pairs", nargs="+", default=list(SKILL_PAIRS), choices=list(SKILL_PAIRS))
    parser.add_argument("--coverages", nargs="+", type=int, default=list(COVERAGES), choices=list(COVERAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TABLE3_SEEDS))
    parser.add_argument("--output", type=Path, default=None, help="also write the cells as JSON")
    args = parser.parse_args()

    cells, missing = {}, []
    for pair in args.pairs:
        props = SKILL_PAIRS[pair]["properties"]
        for cov in args.coverages:
            paths = [args.scores / f"{pair}_cov{cov}_seed{seed}.json" for seed in args.seeds]
            missing += [p.name for p in paths if not p.exists()]
            if not all(p.exists() for p in paths):
                continue
            runs = [json.loads(p.read_text())["splits"] for p in paths]
            a_col = statistics.mean(mean_std([r["train"]["accuracy"][p] for r in runs])[0] for p in props)
            rows = [("test_compositional", "B")]
            if cov == 75 and "test_both_ood" in runs[0]:
                name_a, name_b = (SKILL_NAMES.get(p, p) for p in props)
                rows += [("test_prop_a_ood", f"C({name_a})"), ("test_prop_b_ood", f"C({name_b})"),
                         ("test_both_ood", "D")]
            for split, category in rows:
                skills, line = {}, []
                for p in props:
                    per_seed = [r[split]["accuracy"][p] for r in runs]
                    mean, std = mean_std(per_seed)
                    name = SKILL_NAMES.get(p, p)
                    skills[name] = {"mean": float(mean), "std": float(std), "per_seed": per_seed}
                    line.append(f"{name} {half_up(mean):>5} ± {half_up(std):>4}")
                key = f"{pair}/cov{cov}" + ("" if split == "test_compositional" else f"/{split}")
                cells[key] = {"category": category, "seeds": list(args.seeds), "A": float(a_col), "skills": skills}
                print(f"{pair:<18} {cov:>3}%  {category:<14} A {half_up(a_col):>5}  {'  '.join(line)}")

    if missing:
        print(f"missing {len(missing)} score files, e.g. {missing[:3]}")
    if args.output:
        args.output.write_text(json.dumps(cells, indent=1))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
