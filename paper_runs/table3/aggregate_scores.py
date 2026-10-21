#!/usr/bin/env python
"""Aggregate score_table3.py results into the cells of Table 3.

Reads <scores>/<pair>_cov<c>_seed<s>.json for every seed in TABLE3_SEEDS. A is the mean over the
pair's two skills of the train accuracy (each a mean over seeds). B is test_compositional; at 75%
coverage, pairs with out-of-range values also get C(<skill>) (test_prop_a_ood, test_prop_b_ood:
that skill out of range) and D (test_both_ood). Each cell is the mean and sample standard
deviation over seeds of the per-sample accuracy.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table3.manifest import ARTIFACT_ROOT, COVERAGES, SKILL_PAIRS, TABLE3_SEEDS

SKILL_NAMES = {"radius": "size"}


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


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
            a_col = statistics.mean(statistics.mean(r["train"]["accuracy"][p] for r in runs) for p in props)
            rows = [("test_compositional", "B")]
            if cov == 75 and "test_both_ood" in runs[0]:
                name_a, name_b = (SKILL_NAMES.get(p, p) for p in props)
                rows += [("test_prop_a_ood", f"C({name_a})"), ("test_prop_b_ood", f"C({name_b})"),
                         ("test_both_ood", "D")]
            for split, category in rows:
                skills = {}
                for p in props:
                    per_seed = [r[split]["accuracy"][p] for r in runs]
                    mean, std = mean_std(per_seed)
                    skills[SKILL_NAMES.get(p, p)] = {"mean": mean, "std": std, "per_seed": per_seed}
                key = f"{pair}/cov{cov}" + ("" if split == "test_compositional" else f"/{split}")
                cells[key] = {"category": category, "seeds": list(args.seeds), "A": a_col, "skills": skills}
                line = "  ".join(f"{k} {v['mean']:5.1f} ± {v['std']:4.1f}" for k, v in skills.items())
                print(f"{pair:<18} {cov:>3}%  {category:<14} A {a_col:5.1f}  {line}")

    if missing:
        print(f"missing {len(missing)} score files, e.g. {missing[:3]}")
    if args.output:
        args.output.write_text(json.dumps(cells, indent=1))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
