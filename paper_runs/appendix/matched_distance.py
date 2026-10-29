#!/usr/bin/env python
"""Extrapolation accuracy at matched distance from the training range (Appendix C.1, Table 8).

Reads the baseline extrapolation results that paper_runs.table2.evaluate_table2 writes (results.json per
skill and seed; rotation is baseline_nz over seeds 0-9, as in Table 2) and bins each out-of-range value by

    delta = d_out / width,

the distance outside the training range divided by the width of that range: size radii 5-20 (15 px),
position [-18, 18]^2 (36 px, d_out = L_inf distance to the training square), rotation 45-315 deg (270 deg),
count 2-7 (5). Per seed, a bin is the mean condition accuracy over the values in it (values weighted
equally); cells are mean +- std (ddof=1) over seeds.

    python -m paper_runs.appendix.matched_distance
    python -m paper_runs.appendix.matched_distance --eval-root $VGL_ROOT/MORE_SEEDS/table2/eval/samples20_steps250_cfg1
"""
import argparse
import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.manifest import CANONICAL_SEEDS, CELL_OVERRIDES, EVAL_ROOT

SKILLS = ("size", "position", "rotation", "count")
# training range (low, high) of the Table 2 baseline datasets
RANGE = {"size": (5, 20), "position": (-18, 18), "rotation": (45, 315), "count": (2, 7)}
# (low, high, low closed, high closed, label) as in Table 8
BINS = [
    (Fraction(0), Fraction(7, 100), True, False, "[0, 0.07)"),
    (Fraction(7, 100), Fraction(14, 100), True, False, "[0.07, 0.14)"),
    (Fraction(14, 100), Fraction(1, 6), True, True, "[0.14, 1/6]"),
    (Fraction(1, 6), Fraction(1, 4), False, False, "(1/6, 0.25)"),
    (Fraction(1, 4), Fraction(45, 100), True, False, "[0.25, 0.45)"),
    (Fraction(45, 100), Fraction(2, 3), True, True, "[0.45, 2/3]"),
]
BAND = Fraction(1, 6)  # every out-of-range angle lies within delta 1/6


def seeds_and_variant(skill):
    return CELL_OVERRIDES.get((skill, "baseline"), ("baseline", CANONICAL_SEEDS))


def d_out(skill, key):
    """Distance outside the training range for a condition key of results.json."""
    lo, hi = RANGE[skill]
    if skill == "position":
        x, y = (Fraction(v).limit_denominator(100) for v in key.split(","))
        return max(abs(x), abs(y)) - hi
    v = Fraction(key).limit_denominator(100)
    return lo - v if v < lo else v - hi


def load_extra(eval_root, skill):
    """{seed: {condition key: accuracy}} for the extrapolation split."""
    variant, seeds = seeds_and_variant(skill)
    out = {}
    for seed in seeds:
        path = eval_root / skill / variant / f"seed_{seed}" / "results.json"
        split = json.loads(path.read_text())["paper_metrics"]["splits"]["extra"]
        out[seed] = {k: v["accuracy"] for k, v in split["condition_accuracies"].items()}
    return out


def in_bin(x, b):
    lo, hi, lo_closed, hi_closed, _ = b
    return (x >= lo if lo_closed else x > lo) and (x <= hi if hi_closed else x < hi)


def cell(acc, keys):
    """Mean +- std over seeds of the per-seed mean over keys, or None for an empty bin."""
    if not keys:
        return None
    per_seed = [statistics.mean(a[k] for k in keys) for a in acc.values()]
    return statistics.mean(per_seed), statistics.stdev(per_seed)


def fmt(c):
    return "-" if c is None else f"{c[0]:.1f} +- {c[1]:.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT / "samples20_steps250_cfg1")
    parser.add_argument("--out", type=Path, help="also write the table as JSON")
    args = parser.parse_args()

    acc = {skill: load_extra(args.eval_root, skill) for skill in SKILLS}
    delta = {}
    for skill in SKILLS:
        width = RANGE[skill][1] - RANGE[skill][0]
        keys = next(iter(acc[skill].values()))
        assert all(set(a) == set(keys) for a in acc[skill].values()), f"{skill}: seeds evaluated on different values"
        delta[skill] = {k: d_out(skill, k) / width for k in keys}
        assert all(d > 0 for d in delta[skill].values()), f"{skill}: a value inside the training range"

    rows = []
    for b in BINS:
        rows.append((b[4], {s: cell(acc[s], [k for k, d in delta[s].items() if in_bin(d, b)]) for s in SKILLS}))
    band_keys = {s: [k for k, d in delta[s].items() if d <= BAND] for s in SKILLS}
    rows.append(("delta <= 1/6, mean", {s: cell(acc[s], band_keys[s]) for s in SKILLS}))
    ge90 = {s: (f"{sum(all(a[k] >= 90.0 for k in band_keys[s]) for a in acc[s].values())}/{len(acc[s])}"
                if band_keys[s] else "-") for s in SKILLS}

    print("| delta | " + " | ".join(SKILLS) + " |")
    print("|---|" + "---|" * len(SKILLS))
    for label, r in rows:
        print(f"| {label} | " + " | ".join(fmt(r[s]) for s in SKILLS) + " |")
    print("| delta <= 1/6, seeds >= 90% | " + " | ".join(ge90[s] for s in SKILLS) + " |")
    if args.out:
        table = {label: {s: (None if r[s] is None else {"mean": r[s][0], "std": r[s][1]}) for s in SKILLS}
                 for label, r in rows}
        table["delta <= 1/6, seeds >= 90%"] = ge90
        args.out.write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
