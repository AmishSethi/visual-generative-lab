#!/usr/bin/env python
"""Evaluate the 128x128 replication with scale-matched thresholds.

The 128x128 datasets test whether the 64x64 results hold at a higher
resolution.  They double every length, so the images are geometrically
identical to the paper's and DiT-S/4 at 128 sees the same 32x32 token grid
DiT-S/2 sees at 64.  Resolution is therefore the only variable.

Thresholds are scaled so they mean the same thing:
    size      IoU >= 0.90            scale-free, unchanged
    position  <= 4 px                2 px at 64x64-equivalent
    rotation  <= 5 deg               scale-free, unchanged
    count     exact match            unchanged, min blob area x4

The rotation detector builds its templates at a fixed arrow size, so 128x128
samples are downscaled to 64x64 before detection.  Because the renders are an
exact 2x scaling, this recovers the 64x64-equivalent image and lets the
paper-locked detector run unmodified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paper_runs.table2.evaluate_table2 as t2
from paper_runs.appendix import subsample
from paper_runs.appendix.manifest import (
    DATASET_ROOT, EVAL_ROOT, HIRES_THRESHOLDS, RES_SCALE, RESULTS_ROOT,
)

_original_rotation_metrics = t2.calculate_rotation_metrics_robust


def rotation_metrics_downscaled(image, expected_angle, **kwargs):
    if image.shape[0] != 64:
        image = np.asarray(
            Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((64, 64), Image.Resampling.LANCZOS)
        )
    return _original_rotation_metrics(image, expected_angle, **kwargs)


class EvalArgs:
    def __init__(self, num_samples, steps, batch_size, max_conditions):
        self.num_samples_per_condition = num_samples
        self.num_sampling_steps = steps
        self.eval_batch_size = batch_size
        self.max_conditions_per_split = max_conditions
        # 2x the paper's 64x64 extrapolation grid (1..4 and 21..30 -> 2..8, 42..60)
        self.size_extrap_min = 2
        self.size_extrap_max = 30 * RES_SCALE
        self.position_extrap_margin = t2.POSITION_EXTRAP_MARGIN * RES_SCALE
        self.position_extrap_min_margin = t2.POSITION_EXTRAP_MIN_MARGIN * RES_SCALE


def main():
    parser = argparse.ArgumentParser(description="Evaluate the 128x128 replication.")
    parser.add_argument("--skills", nargs="+", default=["size", "position", "rotation", "count"])
    parser.add_argument("--family", default="hires128", choices=["hires128", "hires128_vae"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--max-conditions-per-split", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # Position queries are generated as nested x,y loops, so the evaluator's
    # prefix truncation would sample only the left-hand columns of the canvas.
    subsample.install(t2)

    out_path = args.out or EVAL_ROOT / f"{args.family}_results.json"

    # Repoint to the appendix tree and scale the length-based thresholds.
    t2.RESULTS_ROOT = RESULTS_ROOT / args.family
    t2.DATASET_ROOT = DATASET_ROOT / "hires128"
    t2.POSITION_ERROR_THRESHOLD = HIRES_THRESHOLDS["position_px"]
    t2.COUNT_MIN_AREA = t2.COUNT_MIN_AREA * RES_SCALE ** 2
    t2.calculate_rotation_metrics_robust = rotation_metrics_downscaled

    eval_args = EvalArgs(args.num_samples, args.steps, args.eval_batch_size,
                         args.max_conditions_per_split)
    payload = {"thresholds": {"size_iou": t2.SIZE_IOU_THRESHOLD,
                              "position_px": t2.POSITION_ERROR_THRESHOLD,
                              "rotation_deg": t2.ROTATION_ERROR_THRESHOLD,
                              "count_min_area": t2.COUNT_MIN_AREA},
               "skills": {}}

    for skill in args.skills:
        # These runs have no variant level (results/hires128/<skill>/seed_N), and
        # pathlib drops an empty component, so "" slots in where evaluate_table2
        # expects the variant name.
        try:
            selection = t2.latest_finished_run(skill, "", args.seed)
        except FileNotFoundError as exc:
            print(f"[skip] {skill}: no finished run ({exc})")
            continue

        print(f"\n=== {args.family} {skill} ===", flush=True)
        target = EVAL_ROOT / args.family / skill / f"seed_{args.seed}"
        target.mkdir(parents=True, exist_ok=True)
        metrics = t2.EVALUATORS[skill](eval_args, selection, target)
        payload["skills"][skill] = {
            "checkpoint": str(selection.checkpoint),
            "splits": {k: v["accuracy"] for k, v in metrics["splits"].items()},
        }
        (target / "results.json").write_text(
            json.dumps({"paper_metrics": metrics, "checkpoint": str(selection.checkpoint)},
                       indent=2, default=str))
        for split, res in metrics["splits"].items():
            print(f"  {split:>7}: {res['accuracy']:.1f}%")

    print(f"\n{'skill':>9} {'train':>8} {'interp':>8} {'extra':>8}")
    for skill, res in payload["skills"].items():
        s = res["splits"]
        print(f"{skill:>9} {s.get('train', float('nan')):>7.1f}% "
              f"{s.get('interp', float('nan')):>7.1f}% {s.get('extra', float('nan')):>7.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
