#!/usr/bin/env python
"""Recompute every simple-64 baseline comparator on the *same* condition set as
the experiment it is compared against.

The paper's Extrap. column is not one protocol. Size uses a 14-condition extended
grid ({1..4} u {21..30}); the visually complex runs score 9 conditions
({1..4} u {21..25}). Position's published cell uses a 320-condition extended grid;
the complex runs score 100 evenly sampled conditions. Pasting those into one row
compares different denominators and manufactures differences that are not there.

This restricts the baseline's per-condition results to the exact conditions the
comparison experiment scored, then averages. Output is persisted so the paper
quotes a file rather than a shell one-liner.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import os as _os
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
REB = Path(f"{VGL_ROOT}/REBUTTAL")
T2 = Path(f"{VGL_ROOT}/MORE_SEEDS/table2")


def baseline_per_condition(skill, split, run):
    """Per-condition accuracies for the simple-64 baseline, one dict per seed."""
    out = []
    for f in sorted((T2 / "eval" / run / skill / "baseline").glob("seed_*/results.json")):
        d = json.loads(f.read_text())["paper_metrics"]["splits"].get(split, {})
        ca = d.get("condition_accuracies")
        if ca:
            out.append({k: v["accuracy"] for k, v in ca.items()})
    return out


def restricted_mean(per_seed, keys):
    """Mean over the given conditions, per seed, then across seeds."""
    vals = []
    for cond in per_seed:
        sel = [cond[k] for k in keys if k in cond]
        if len(sel) == len(keys):
            vals.append(st.mean(sel))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REB / "eval" / "matched_comparators.json")
    a = ap.parse_args()

    # experiment file -> (skill, split, which baseline eval run to draw from)
    EXPERIMENTS = {
        "realistic": [("size", "extra", "samples20_steps250_cfg1_extended"),
                      ("position", "extra", "samples20_steps250_cfg1_extended"),
                      ("rotation", "extra", "samples20_steps250_cfg1_rot5deg"),
                      ("count", "extra", "samples20_steps250_cfg1")],
    }

    report = {}
    for family, specs in EXPERIMENTS.items():
        for skill, split, run in specs:
            # conditions the experiment actually scored
            exp_files = sorted(REB.glob(f"eval/{family}_{skill}.json")) + \
                        sorted(f for f in REB.glob(f"eval/{family}_s[0-9]*.json"))   # seed files only; never the per-skill file twice
            keys, exp_vals = None, []
            for f in exp_files:
                try:
                    blob = json.loads(f.read_text())
                except Exception:
                    continue
                sk = blob.get("skills", {}).get(skill)
                if not sk:
                    continue
                sp = sk["splits"].get(split, {})
                ca = sp.get("condition_accuracies")
                if ca:
                    keys = keys or sorted(ca)
                    exp_vals.append(sp["accuracy"])
            if not keys:
                print(f"[skip] {family}/{skill}: no per-condition data"); continue

            base_seeds = baseline_per_condition(skill, split, run)
            matched = restricted_mean(base_seeds, keys)
            published = restricted_mean(base_seeds, sorted(base_seeds[0])) if base_seeds else []

            report[f"{family}:{skill}:{split}"] = {
                "n_conditions_experiment": len(keys),
                "n_conditions_published": len(base_seeds[0]) if base_seeds else 0,
                "baseline_matched_mean": round(st.mean(matched), 2) if matched else None,
                "baseline_matched_std": round(st.stdev(matched), 2) if len(matched) > 1 else 0.0,
                "baseline_published_grid_mean": round(st.mean(published), 2) if published else None,
                "experiment_mean": round(st.mean(exp_vals), 2) if exp_vals else None,
                "experiment_n_seeds": len(exp_vals),
                "delta_matched": round(st.mean(exp_vals) - st.mean(matched), 2)
                                 if exp_vals and matched else None,
            }
            r = report[f"{family}:{skill}:{split}"]
            f = lambda v: "  n/a" if v is None else f"{v:6.1f}"
            note = "" if r["baseline_matched_mean"] is not None else \
                   "   <-- no seed covers all experiment conditions; grids disjoint"
            print(f"{skill:9s} exp {f(r['experiment_mean'])} (n={r['experiment_n_seeds']})  "
                  f"baseline matched {f(r['baseline_matched_mean'])} on {r['n_conditions_experiment']} conds "
                  f"(published: {r['baseline_published_grid_mean']} on {r['n_conditions_published']})  "
                  f"delta {f(r['delta_matched'])}{note}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
