#!/usr/bin/env python
"""Separate coverage fraction from number of combinations (Reviewer XDWw / AC).

XDWw's objection is that our "coverage matters" result may really be "more
unique combinations matter", since in the paper's 4-shape x 8-colour grid the
two move together.  Varying both vocabularies breaks that:

    S in {4, 8} x C in {8, 16} x coverage in {25%, 50%, 75%}

gives twelve cells in which the same K appears at different coverage fractions
and the same coverage fraction appears at different K.  Two analyses:

  matched contrasts  cells sharing K, compared across coverage, and cells
                     sharing coverage, compared across K.  No modelling.
  regression         accuracy ~ coverage + log2(K), reporting each term's
                     partial contribution, so the two hypotheses are weighed
                     on the same twelve points.

The total image budget is identical in every cell, so samples-per-combination
falls as K rises -- exactly the tradeoff the paper's Finding 2 is about.
"""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_runs.rebuttal.manifest import EVAL_ROOT

METRIC = "joint_accuracy"


def load_cells(path_or_dir):
    """Merge every per-seed result file, averaging each cell across seeds."""
    path = Path(path_or_dir)
    search_dir = path.parent if path.suffix == ".json" else path
    files = sorted(search_dir.glob("shape_vocab_results_seed*.json"))
    if not files and path.is_file():
        files = [path]

    # The four shapes the paper uses; S=8 adds pentagon/hexagon/star/cross, which
    # may simply be harder to draw.  Restricting to the original four separates
    # "more combinations" from "a harder vocabulary" when comparing S=4 to S=8.
    ORIGINAL_SHAPES = {"circle", "square", "triangle", "diamond"}

    per_name = defaultdict(list)
    meta = {}
    seen = set()
    for f in files:
        # Filenames are shape_vocab_results_seed<N><suffix>.json; the suffix lets
        # one seed be split across parallel jobs, so key on the seed number and
        # drop repeats rather than counting them as extra seeds.
        digits = "".join(ch for ch in f.stem.split("seed")[-1] if ch.isdigit())
        seed_id = int(digits) if digits else 0
        data = json.loads(f.read_text())
        for name, res in data.items():
            if (seed_id, name) in seen:
                continue
            seen.add((seed_id, name))
            b = res["categories"]["B_heldout"]
            per_combo = b.get("per_combination", {})
            orig = [v["joint"] for combo, v in per_combo.items()
                    if combo.rsplit("_", 1)[0] in ORIGINAL_SHAPES]
            per_name[name].append((b["shape_accuracy"], b["color_accuracy"],
                                   b["joint_accuracy"],
                                   float(np.mean(orig)) if orig else float("nan")))
            meta[name] = res

    cells = []
    for name, values in per_name.items():
        res = meta[name]
        arr = np.array(values, dtype=float)
        cells.append({
            "name": name,
            "S": res["num_shapes"],
            "C": len(res.get("colors", [])) or (16 if "_c16_" in name else 8),
            "coverage": float(res["coverage"]),
            "K": int(res["K_train_combinations"]),
            "samples_per_combo": int(res["samples_per_combination"]),
            "n_seeds": len(values),
            "shape": float(arr[:, 0].mean()),
            "color": float(arr[:, 1].mean()),
            "joint": float(arr[:, 2].mean()),
            "joint_std": float(arr[:, 2].std(ddof=1)) if len(values) > 1 else 0.0,
            "joint_original4": float(np.nanmean(arr[:, 3])),
        })
    return cells


def matched_contrasts(cells):
    print("\n--- cells sharing K, varying coverage "
          "(if K is what matters these are equal) ---")
    by_k = defaultdict(list)
    for c in cells:
        by_k[c["K"]].append(c)
    any_shown = False
    for k, group in sorted(by_k.items()):
        coverages = {c["coverage"] for c in group}
        if len(group) < 2 or len(coverages) < 2:
            continue
        any_shown = True
        print(f"  K={k}:")
        for c in sorted(group, key=lambda x: -x["coverage"]):
            print(f"     {c['name']:>16}  cov {c['coverage']:>4.0%}  "
                  f"joint {c['joint']:6.1f}%  shape {c['shape']:6.1f}%  colour {c['color']:6.1f}%")
        hi = max(group, key=lambda x: x["coverage"])
        lo = min(group, key=lambda x: x["coverage"])
        print(f"     -> coverage {lo['coverage']:.0%} to {hi['coverage']:.0%} at fixed K: "
              f"{hi['joint'] - lo['joint']:+.1f} pp")
    if not any_shown:
        print("  (no K value yet has two different coverages evaluated)")

    print("\n--- cells sharing coverage, varying K "
          "(if coverage is what matters these are equal) ---")
    by_cov = defaultdict(list)
    for c in cells:
        by_cov[round(c["coverage"], 3)].append(c)
    for cov, group in sorted(by_cov.items()):
        ks = {c["K"] for c in group}
        if len(group) < 2 or len(ks) < 2:
            continue
        print(f"  coverage={cov:.0%}:")
        for c in sorted(group, key=lambda x: x["K"]):
            print(f"     {c['name']:>16}  K {c['K']:>3}  "
                  f"joint {c['joint']:6.1f}%  shape {c['shape']:6.1f}%  colour {c['color']:6.1f}%")
        hi = max(group, key=lambda x: x["K"])
        lo = min(group, key=lambda x: x["K"])
        print(f"     -> K {lo['K']} to {hi['K']} at fixed coverage: "
              f"{hi['joint'] - lo['joint']:+.1f} pp")


def regression(cells, metric=METRIC):
    """OLS of accuracy on coverage and log2(K), with per-term partial R^2."""
    y = np.array([c[metric.replace("_accuracy", "")] for c in cells], dtype=float)
    coverage = np.array([c["coverage"] for c in cells], dtype=float)
    logk = np.array([math.log2(c["K"]) for c in cells], dtype=float)

    def fit(columns):
        X = np.column_stack([np.ones(len(y))] + columns)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return beta, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

    beta_full, r2_full = fit([coverage, logk])
    _, r2_cov_only = fit([coverage])
    _, r2_k_only = fit([logk])

    print(f"\n--- regression on {len(cells)} cells, target = B-split {metric} ---")
    print(f"  accuracy = {beta_full[0]:.1f} + {beta_full[1]:.1f}*coverage "
          f"+ {beta_full[2]:.1f}*log2(K)")
    print(f"  R^2 full model            : {r2_full:.3f}")
    print(f"  R^2 coverage alone        : {r2_cov_only:.3f}")
    print(f"  R^2 log2(K) alone         : {r2_k_only:.3f}")
    print(f"  unique to coverage        : {r2_full - r2_k_only:.3f}")
    print(f"  unique to log2(K)         : {r2_full - r2_cov_only:.3f}")
    return {
        "intercept": beta_full[0], "beta_coverage": beta_full[1], "beta_log2K": beta_full[2],
        "r2_full": r2_full, "r2_coverage_only": r2_cov_only, "r2_logK_only": r2_k_only,
        "unique_to_coverage": r2_full - r2_k_only, "unique_to_logK": r2_full - r2_cov_only,
        "n_cells": len(cells),
    }


def main():
    parser = argparse.ArgumentParser(description="Coverage vs number-of-combinations analysis.")
    parser.add_argument("--results", type=Path, default=EVAL_ROOT,
                        help="Directory holding shape_vocab_results_seed*.json.")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "coverage_vs_k_analysis.json")
    args = parser.parse_args()

    cells = load_cells(args.results)
    if not cells:
        print("no cells")
        return

    print(f"{'cell':>16} {'S':>2} {'C':>3} {'cov':>5} {'K':>4} {'smp/K':>6} {'n':>2} "
          f"{'shape':>7} {'colour':>7} {'joint':>14} {'joint(4sh)':>10}")
    for c in sorted(cells, key=lambda x: (x["S"], x["C"], x["coverage"])):
        print(f"{c['name']:>16} {c['S']:>2} {c['C']:>3} {c['coverage']:>5.0%} {c['K']:>4} "
              f"{c['samples_per_combo']:>6} {c['n_seeds']:>2} {c['shape']:>6.1f}% {c['color']:>6.1f}% "
              f"{c['joint']:>7.1f} ± {c['joint_std']:<4.1f} {c['joint_original4']:>7.1f}")

    matched_contrasts(cells)
    stats = regression(cells) if len(cells) >= 5 else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cells": cells, "regression": stats}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
