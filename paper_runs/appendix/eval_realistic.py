#!/usr/bin/env python
"""Evaluate the visually complex (realistic-render) runs.

The paper's extractors assume a flat subject on a near-white ground and key off
Otsu thresholding, which does not survive textured backgrounds and shaded
objects.  Geometry here is recovered from the saturation mask instead -- objects
are rendered saturated, backgrounds desaturated -- and every metric keeps the
paper's own threshold so the numbers stay comparable to Table 1:

    size      IoU >= 0.90 against the ground-truth disc
    position  centroid within 2 px
    rotation  arrow direction within 5 deg
    count     exact match, watershed over the saturation mask

`--validate-only` scores ground-truth renders with the identical code first and
reports the ceiling, so any gap between the model and 100% can be attributed
correctly.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paper_runs.table2.evaluate_table2 as t2
from paper_runs.appendix import subsample
from paper_runs.appendix.manifest import DATASET_ROOT, EVAL_ROOT, RESULTS_ROOT
from paper_runs.appendix.realistic import (
    circle_mask, object_mask, render_arrow, render_circle, render_circles,
)
from paper_runs.table2.generate_canonical_datasets import sample_non_overlapping_centers
from vgl.eval_radius import generate_samples as generate_radius_samples, load_model as load_radius_model
from vgl.eval_rotation import generate_samples as generate_rotation_samples, load_model as load_rotation_model

IMAGE_SIZE = 64
SIZE_IOU_THRESHOLD = 0.90
POSITION_ERROR_THRESHOLD = 2.0
ROTATION_ERROR_THRESHOLD = 5.0
MIN_BLOB_AREA = 30


def disc_mask(radius, cx=0.0, cy=0.0, size=IMAGE_SIZE):
    """Ground-truth mask built by the renderer's own rasteriser.

    An analytic disc sits half a pixel off the rasterised one, which costs up to
    0.14 IoU at small radii and would have been charged to the model.
    """
    return circle_mask(radius, cx, cy, size)


def score_size(image, radius):
    mask = object_mask(image)
    gt = disc_mask(radius)
    union = np.logical_or(mask > 0, gt > 0).sum()
    iou = float(np.logical_and(mask > 0, gt > 0).sum() / union) if union else 0.0
    area = int(mask.sum())
    return {"iou": iou, "estimated_radius": math.sqrt(area / math.pi) if area else 0.0}


def score_position(image, target):
    mask = object_mask(image)
    ys, xs = np.nonzero(mask)
    if len(ys) < MIN_BLOB_AREA:
        return {"error": float("inf"), "detected": False}
    cx, cy = xs.mean() - IMAGE_SIZE / 2, ys.mean() - IMAGE_SIZE / 2
    return {"error": float(np.hypot(cx - target[0], cy - target[1])), "detected": True}


def score_rotation(image, angle):
    """Principal axis, signed by which end carries the arrowhead's extra width."""
    mask = object_mask(image)
    ys, xs = np.nonzero(mask)
    if len(ys) < MIN_BLOB_AREA:
        return {"error": float("inf"), "detected": False}
    points = np.stack([xs, ys], 1).astype(np.float64)
    centred = points - points.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    axial = centred @ axis
    perp = np.abs(centred @ np.array([-axis[1], axis[0]]))
    head = axial >= np.quantile(axial, 0.75)
    tail = axial <= np.quantile(axial, 0.25)
    direction = axis if perp[head].mean() > perp[tail].mean() else -axis
    detected = float(np.degrees(np.arctan2(-direction[1], direction[0])) % 360.0)
    diff = abs((detected - angle + 180.0) % 360.0 - 180.0)
    return {"error": diff, "detected": True, "detected_angle": detected}


def score_count(image):
    mask = object_mask(image) * 255
    if mask.max() == 0:
        return 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    unknown = cv2.subtract(cv2.dilate(mask, kernel, iterations=2), sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    if n <= 1:
        return 0
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    return sum(1 for label in np.unique(markers)
               if label > 1 and int(np.count_nonzero(markers == label)) > MIN_BLOB_AREA)


def validate(n=60, seed=0):
    """Metric ceilings on ground-truth complex renders."""
    rng = np.random.default_rng(seed)
    out = {}

    ious = [score_size(render_circle(r, 0, 0, IMAGE_SIZE, rng), r)["iou"]
            for r in range(5, 21) for _ in range(n // 16 + 1)]
    out["size_pct_iou_above_threshold"] = 100.0 * float(np.mean([i >= SIZE_IOU_THRESHOLD for i in ious]))
    out["size_mean_iou"] = float(np.mean(ious))

    errs = []
    for _ in range(n):
        x, y = float(rng.integers(-18, 19)), float(rng.integers(-18, 19))
        errs.append(score_position(render_circle(8, x, y, IMAGE_SIZE, rng), (x, y))["error"])
    out["position_pct_within_threshold"] = 100.0 * float(np.mean([e <= POSITION_ERROR_THRESHOLD for e in errs]))
    out["position_mean_error_px"] = float(np.mean(errs))

    errs = []
    for angle in range(0, 360, 6):
        errs.append(score_rotation(render_arrow(float(angle), IMAGE_SIZE, 20, rng), float(angle))["error"])
    out["rotation_pct_within_threshold"] = 100.0 * float(np.mean([e <= ROTATION_ERROR_THRESHOLD for e in errs]))
    out["rotation_mean_error_deg"] = float(np.mean(errs))

    hits = total = 0
    for c in range(2, 8):
        for _ in range(n // 6 + 1):
            centers = sample_non_overlapping_centers(c, 8, IMAGE_SIZE, rng)
            hits += score_count(render_circles(centers, 8, IMAGE_SIZE, rng)) == c
            total += 1
    out["count_accuracy"] = 100.0 * hits / total
    return out


def evaluate(skill, seed, num_samples, steps, batch_size, max_conditions, device):
    t2.DATASET_ROOT = DATASET_ROOT / "realistic"
    t2.RESULTS_ROOT = RESULTS_ROOT / "realistic"
    metadata = t2.load_dataset_metadata(skill)
    selection = t2.latest_finished_run(skill, "", seed)

    if skill == "size":
        queries, config = t2.build_size_queries(metadata), t2.build_size_config(selection)
        load, gen = load_radius_model, generate_radius_samples
    elif skill == "count":
        queries, config = t2.build_count_queries(metadata), t2.build_count_config(selection)
        load, gen = load_radius_model, generate_radius_samples
    elif skill == "position":
        queries = t2.build_position_queries(metadata)
        config = t2.build_position_config(selection, metadata)
        from vgl.eval_position import (generate_samples as gen_pos, load_model as load_pos)
        load, gen = load_pos, gen_pos
    else:
        queries = t2.build_rotation_queries(metadata)
        config = t2.build_rotation_config(selection, metadata)
        load, gen = load_rotation_model, generate_rotation_samples

    queries = t2.maybe_truncate_queries(queries, max_conditions)
    model = load(config, device)
    sampler = t2.create_sampler(config, steps)

    splits = {}
    for split_name, values in queries.items():
        samples = gen(model, sampler, values, config, device, num_samples=num_samples,
                      cfg_scale=1.0, batch_size=batch_size, num_sampling_steps=steps)
        hits = total = 0
        per_condition = {}
        for value in values:
            ok = 0
            for sample in samples[value]:
                image = np.asarray(sample, dtype=np.uint8)
                if skill == "size":
                    ok += score_size(image, value)["iou"] >= SIZE_IOU_THRESHOLD
                elif skill == "position":
                    ok += score_position(image, value)["error"] <= POSITION_ERROR_THRESHOLD
                elif skill == "rotation":
                    ok += score_rotation(image, value)["error"] <= ROTATION_ERROR_THRESHOLD
                else:
                    # Compare against the requested value itself, as
                    # evaluate_table2.py does.  int(round(v)) would banker's-round
                    # the interpolation midpoints [2.5,3.5,4.5,5.5,6.5] to
                    # [2,4,4,6,6] -- all trained values -- so the interp split
                    # would re-measure training accuracy instead of interpolation.
                    ok += float(score_count(image)) == float(value)
            n = len(samples[value])
            per_condition[t2.serialize_query(value)] = {"accuracy": 100.0 * ok / n, "num_samples": n}
            hits += ok
            total += n
        splits[split_name] = {"accuracy": 100.0 * hits / total if total else 0.0,
                              "num_conditions": len(values),
                              "condition_accuracies": per_condition}
        print(f"  {split_name:>7}: {splits[split_name]['accuracy']:.1f}%", flush=True)

    del model
    torch.cuda.empty_cache()
    return {"checkpoint": str(selection.checkpoint), "splits": splits}


def main():
    parser = argparse.ArgumentParser(description="Evaluate visually complex runs.")
    parser.add_argument("--skills", nargs="+", default=["size", "position", "rotation", "count"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--max-conditions-per-split", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "realistic_results.json")
    args = parser.parse_args()

    # Position queries are generated as nested x,y loops, so the evaluator's
    # prefix truncation would sample only the left-hand columns of the canvas.
    subsample.install(t2)

    payload = {"metric_ceilings_on_ground_truth": validate()}
    print("metric ceilings on ground-truth complex renders:")
    for key, value in payload["metric_ceilings_on_ground_truth"].items():
        print(f"    {key:>32}: {value:.2f}")
    if args.validate_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload["skills"] = {}
    for skill in args.skills:
        try:
            print(f"\n=== realistic {skill} ===", flush=True)
            payload["skills"][skill] = evaluate(
                skill, args.seed, args.num_samples, args.steps,
                args.eval_batch_size, args.max_conditions_per_split, device)
        except FileNotFoundError as exc:
            print(f"[skip] {skill}: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
