#!/usr/bin/env python
"""Re-score dumped count generations with three counters, and measure overlap.

Completes the answer to Reviewer yLdq's question 3.  counter_crossval.py
established what the counters can do on ground truth; this measures what they
say about the model's actual generations, and how often those generations place
objects on top of each other -- the regime in which every counter collapses.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.counter_crossval import COUNTERS, CIRCLE_RADIUS, _binary_mask
from paper_runs.rebuttal.manifest import EVAL_ROOT


def detect_centers(image, radius=CIRCLE_RADIUS):
    """Circle centres via Hough, used to measure inter-object distance."""
    _, binary = _binary_mask(image)
    if binary.max() == 0:
        return []
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=int(radius * 1.2),
        param1=100, param2=8, minRadius=max(2, radius - 4), maxRadius=radius + 4,
    )
    if circles is None:
        return []
    return [(float(c[0]), float(c[1])) for c in circles[0]]


def overlap_stats(centers, radius=CIRCLE_RADIUS):
    """Fraction of object pairs whose centres are closer than 2r."""
    if len(centers) < 2:
        return 0.0, None
    distances = []
    overlapping = 0
    total = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = float(np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
            distances.append(d)
            overlapping += d < 2 * radius
            total += 1
    return overlapping / total, float(np.min(distances))


def score_seed(seed_root):
    manifest = json.loads((seed_root / "manifest.json").read_text())
    per_split = {}
    for split_name, values in manifest["splits"].items():
        per_value = {}
        for value_key in values:
            value_dir = seed_root / split_name / value_key
            paths = sorted(value_dir.glob("*.png"))
            if not paths:
                continue
            target = float(value_key)
            hits = defaultdict(int)
            overlap_fracs = []
            any_overlap = 0
            for path in paths:
                image = np.asarray(Image.open(path).convert("RGB"))
                for name, fn in COUNTERS.items():
                    hits[name] += fn(image) == target
                frac, _ = overlap_stats(detect_centers(image))
                overlap_fracs.append(frac)
                any_overlap += frac > 0
            n = len(paths)
            per_value[value_key] = {
                "n": n,
                **{f"{name}_acc": 100.0 * hits[name] / n for name in COUNTERS},
                "mean_overlapping_pair_fraction": float(np.mean(overlap_fracs)),
                "pct_images_with_any_overlap": 100.0 * any_overlap / n,
            }
        per_split[split_name] = per_value
    return per_split


def main():
    parser = argparse.ArgumentParser(description="Score dumped count generations.")
    parser.add_argument("--samples-root", type=Path, default=EVAL_ROOT / "samples" / "count" / "baseline")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "count_samples_scored.json")
    args = parser.parse_args()

    payload = {"seeds": {}}
    seed_dirs = sorted(p for p in args.samples_root.glob("seed_*") if (p / "manifest.json").exists())
    if not seed_dirs:
        print(f"no dumped samples under {args.samples_root}")
        return

    for seed_root in seed_dirs:
        payload["seeds"][seed_root.name] = score_seed(seed_root)
        print(f"scored {seed_root.name}", flush=True)

    # aggregate across seeds
    print(f"\n{'split':>7} {'N':>5} " + "".join(f"{n:>12}" for n in COUNTERS) + f"{'overlap%':>10}")
    summary = {}
    for split_name in payload["seeds"][seed_dirs[0].name]:
        summary[split_name] = {}
        keys = payload["seeds"][seed_dirs[0].name][split_name].keys()
        for value_key in keys:
            row = {}
            for name in COUNTERS:
                vals = [payload["seeds"][s][split_name][value_key][f"{name}_acc"]
                        for s in payload["seeds"]]
                row[name] = (float(np.mean(vals)), float(np.std(vals)))
            ov = [payload["seeds"][s][split_name][value_key]["pct_images_with_any_overlap"]
                  for s in payload["seeds"]]
            row["overlap"] = float(np.mean(ov))
            summary[split_name][value_key] = row
            print(f"{split_name:>7} {float(value_key):>5.1f} "
                  + "".join(f"{row[n][0]:>8.1f}+-{row[n][1]:<3.0f}" for n in COUNTERS)
                  + f"{row['overlap']:>9.0f}%")

    # split-level means
    print()
    for split_name, values in summary.items():
        means = {n: np.mean([v[n][0] for v in values.values()]) for n in COUNTERS}
        print(f"{split_name:>7} mean: " + ", ".join(f"{n}={means[n]:.1f}%" for n in COUNTERS))
        payload.setdefault("split_means", {})[split_name] = {n: float(means[n]) for n in COUNTERS}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
