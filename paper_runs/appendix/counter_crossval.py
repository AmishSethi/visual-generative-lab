#!/usr/bin/env python
"""Cross-validate three independent object counters on VGL count data.

The count skill's failure could be poor generalisation over discrete numbers,
or an artefact of object placement, overlap and the connected-component
evaluator.  Separating those needs two measurements:

  1. the counters' accuracy on *ground-truth* renders, which upper-bounds what
     any generator can score;
  2. the counters' agreement on *generated* samples, plus how often the
     generated objects actually overlap.

This script performs measurement 1 and defines the counters;
score_count_samples.py imports them for measurement 2.

Counters:
  watershed  the paper's own detector, imported unchanged from evaluate_table2
  hough      cv2.HoughCircles, a shape prior the watershed does not use
  log        Laplacian-of-Gaussian blob detection, scale-space based

If all three agree that accuracy is low, the model is at fault.  If the
counters disagree, and disagree most where objects touch, the evaluator is.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.manifest import EVAL_ROOT
from paper_runs.table2.evaluate_table2 import COUNT_MIN_AREA, detect_count_with_watershed
from paper_runs.table2.generate_canonical_datasets import (
    background_color, finalize, foreground_color, sample_non_overlapping_centers,
)

CIRCLE_RADIUS = 8
IMAGE_SIZE = 64
GT_SAMPLES_PER_COUNT = 100


def _binary_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return gray, binary


def detect_count_with_hough(image, radius=CIRCLE_RADIUS):
    gray, binary = _binary_mask(image)
    if binary.max() == 0:
        return 0
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=int(radius * 1.2),
        param1=100, param2=8, minRadius=max(2, radius - 4), maxRadius=radius + 4,
    )
    return 0 if circles is None else int(circles.shape[1])


def detect_count_with_log(image, radius=CIRCLE_RADIUS):
    """Laplacian-of-Gaussian blob detection via cv2.SimpleBlobDetector."""
    gray, binary = _binary_mask(image)
    if binary.max() == 0:
        return 0
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = COUNT_MIN_AREA
    params.maxArea = float(IMAGE_SIZE * IMAGE_SIZE)
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 255
    detector = cv2.SimpleBlobDetector_create(params)
    return len(detector.detect(binary))


COUNTERS = {
    "watershed": detect_count_with_watershed,
    "hough": detect_count_with_hough,
    "log": detect_count_with_log,
}


def overlap_fraction(centers, radius=CIRCLE_RADIUS):
    """Fraction of object pairs closer than 2r (i.e. actually overlapping)."""
    if len(centers) < 2:
        return 0.0
    close = total = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            total += 1
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            close += (dx * dx + dy * dy) < (2 * radius) ** 2
    return close / total


def render_count_image(count, rng, allow_overlap=False):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), background_color(rng))
    draw = ImageDraw.Draw(image)
    if allow_overlap:
        lo = -IMAGE_SIZE // 2 + CIRCLE_RADIUS
        hi = IMAGE_SIZE // 2 - CIRCLE_RADIUS
        centers = [(int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
                   for _ in range(count)]
    else:
        centers = sample_non_overlapping_centers(count, CIRCLE_RADIUS, IMAGE_SIZE, rng)
    for cx, cy in centers:
        draw.ellipse(
            [IMAGE_SIZE / 2 + cx - CIRCLE_RADIUS, IMAGE_SIZE / 2 + cy - CIRCLE_RADIUS,
             IMAGE_SIZE / 2 + cx + CIRCLE_RADIUS, IMAGE_SIZE / 2 + cy + CIRCLE_RADIUS],
            fill=foreground_color(rng),
        )
    return np.asarray(finalize(image, rng)), centers


def validate_on_ground_truth(counts, allow_overlap):
    """Counter accuracy on renders whose true count is known exactly."""
    results = {name: {} for name in COUNTERS}
    for count in counts:
        rng = np.random.default_rng(1234 + count)
        hits = defaultdict(int)
        for _ in range(GT_SAMPLES_PER_COUNT):
            image, _ = render_count_image(count, rng, allow_overlap=allow_overlap)
            for name, fn in COUNTERS.items():
                hits[name] += fn(image) == count
        for name in COUNTERS:
            results[name][count] = 100.0 * hits[name] / GT_SAMPLES_PER_COUNT
    return results


def main():
    parser = argparse.ArgumentParser(description="Three-counter cross-validation on VGL count data.")
    parser.add_argument("--counts", nargs="+", type=int, default=list(range(0, 11)))
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "counter_crossval.json")
    args = parser.parse_args()

    payload = {"counters": list(COUNTERS), "samples_per_count": GT_SAMPLES_PER_COUNT}

    for label, allow_overlap in (("non_overlapping", False), ("random_placement", True)):
        # The paper's training data places circles without overlap; generated
        # samples need not respect that, so we measure both regimes.
        counts = args.counts if not allow_overlap else [c for c in args.counts if c <= 9]
        res = validate_on_ground_truth(counts, allow_overlap)
        payload[label] = res

        print(f"\n=== ground-truth accuracy, {label.replace('_', ' ')} ===")
        header = f"{'N':>3} " + "".join(f"{n:>12}" for n in COUNTERS)
        print(header)
        for count in counts:
            row = f"{count:>3} " + "".join(f"{res[n][count]:>11.0f}%" for n in COUNTERS)
            print(row)
        train_counts = [c for c in counts if 2 <= c <= 7]
        print("  mean over the paper's training range N=2..7: " + ", ".join(
            f"{n}={np.mean([res[n][c] for c in train_counts]):.1f}%" for n in COUNTERS))
        payload[f"{label}_train_range_mean"] = {
            n: float(np.mean([res[n][c] for c in train_counts])) for n in COUNTERS
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
