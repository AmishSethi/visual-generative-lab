#!/usr/bin/env python
"""Does the extrapolation boundary follow the training support? (yLdq w2)

yLdq argues our failures may come from "numerical normalization [or] training
range design" rather than from the model failing to learn the visual rule.  The
two hypotheses make different predictions when the support is moved:

  numeric-scale limit   the model breaks a fixed number of *pixels* past the
                        boundary, wherever that boundary sits
  support-relative      the model breaks a fixed *fraction of the training
                        span* past the boundary

For each run we find the break point: the furthest query, on each side, that
still clears 50% accuracy.  Reporting it both in raw pixels and normalised by
the training span separates the hypotheses directly.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_runs.rebuttal.manifest import EVAL_ROOT, RANGE_SPECS

ACCURACY_FLOOR = 50.0


def support_of(name):
    spec = RANGE_SPECS[name]
    if spec["skill"] == "size":
        radii = spec["train_radii"]
        return float(min(radii)), float(max(radii))
    return float(spec["min_angle"]), float(spec["max_angle"])


def break_points(per_condition, low, high, skill):
    """Furthest value beyond each boundary still at or above the floor."""
    values = sorted((float(k.split(",")[0]), v.get("accuracy", 0.0))
                    for k, v in per_condition.items())
    upper = high
    for value, accuracy in values:
        if value > high and accuracy >= ACCURACY_FLOOR:
            upper = max(upper, value)
    lower = low
    for value, accuracy in sorted(values, reverse=True):
        if value < low and accuracy >= ACCURACY_FLOOR:
            lower = min(lower, value)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description="Training-range break-point analysis.")
    parser.add_argument("--results", type=Path, default=EVAL_ROOT / "ranges_results.json")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "ranges_analysis.json")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"no results yet at {args.results}")
        return
    data = json.loads(args.results.read_text())

    rows = []
    for name, res in data.items():
        low, high = support_of(name)
        span = high - low
        merged = {}
        for split in ("train", "interp", "extra"):
            merged.update(res["per_condition"].get(split, {}))
        if not merged:
            continue
        b_low, b_high = break_points(merged, low, high, res["skill"])
        rows.append({
            "run": name, "skill": res["skill"],
            "support": [low, high], "span": span,
            "break_below_px": low - b_low, "break_above_px": b_high - high,
            "break_below_norm": (low - b_low) / span, "break_above_norm": (b_high - high) / span,
            "splits": res["splits"],
        })

    if not rows:
        print("no per-condition data")
        return

    print(f"{'run':>18} {'skill':>9} {'support':>14} {'span':>6} "
          f"{'break+ (px)':>12} {'break+ (norm)':>14} {'break- (px)':>12} {'break- (norm)':>14}")
    for r in rows:
        low, high = r["support"]
        support = f"{low:g}-{high:g}"
        print(f"{r['run']:>18} {r['skill']:>9} {support:>14} "
              f"{r['span']:>6.0f} {r['break_above_px']:>12.1f} {r['break_above_norm']:>14.3f} "
              f"{r['break_below_px']:>12.1f} {r['break_below_norm']:>14.3f}")

    for skill in {r["skill"] for r in rows}:
        group = [r for r in rows if r["skill"] == skill]
        if len(group) < 2:
            continue
        px = np.array([r["break_above_px"] for r in group], dtype=float)
        norm = np.array([r["break_above_norm"] for r in group], dtype=float)
        cv_px = float(px.std() / px.mean()) if px.mean() else float("nan")
        cv_norm = float(norm.std() / norm.mean()) if norm.mean() else float("nan")
        print(f"\n{skill}: across supports, break distance varies by "
              f"CV={cv_px:.2f} in pixels and CV={cv_norm:.2f} in span-normalised units.")
        verdict = ("scales with the training support (support-relative limit)"
                   if cv_norm < cv_px else
                   "is roughly constant in pixels (numeric-scale limit)")
        print(f"    -> the break point {verdict}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
