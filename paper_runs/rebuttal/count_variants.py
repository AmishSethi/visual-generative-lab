#!/usr/bin/env python
"""Count dataset variants, to find a geometry the baseline can actually fit.

The paper's count dataset places circles of radius 8 at random non-overlapping
positions in a 64x64 frame, counts 2 to 7.  DiT-S/2 fits that to 51.8% exact
match while every other skill reaches 100%.

Seven radius-8 circles need centres >= 18 px apart inside a 48x48 usable box,
which is about 67% of the hexagonal packing limit.  Placements at the high
counts are therefore strongly constrained and nearly crystalline, and a small
placement error merges two circles.  Each variant below relaxes one part of that
geometry while holding the count range, image budget, render style and colour
sampling identical to the paper's.

    r5, r6      smaller circles, so the same counts sit far below the packing
                limit (21% and 31% instead of 67%)
    grid_r5     circles snapped to lattice cells, removing continuous placement
    grid_r8     the same at the paper's radius, isolating placement from size

Widening the separation at radius 8 is not an option and that is itself the
finding: the hexagonal capacity of the 48x48 usable box is 8.2 centres at the
paper's 18 px spacing, so seven objects barely fit.  Going to 20 px drops
capacity to 6.7 and 24 px to 4.6, both below seven.  The paper's count geometry
sits at the edge of what the frame admits, which is the likely reason the
n=6 and n=7 placements are so constrained.  Only shrinking the circles buys room.

Radius 4 is deliberately excluded: at 50 px^2 it is only 1.7x COUNT_MIN_AREA,
so a slightly eroded blob would fall below the counter's floor and be dropped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.manifest import DATASET_ROOT
from paper_runs.table2.generate_canonical_datasets import (
    background_color, distribute_counts, finalize, foreground_color,
)

IMAGE_SIZE = 64
COUNTS = list(range(2, 8))
TOTAL_IMAGES = 10000
SEED = 42

VARIANTS = {
    "r5":      {"radius": 5, "placement": "random", "sep_pad": 1},
    "r6":      {"radius": 6, "placement": "random", "sep_pad": 1},
    "r5_sep":  {"radius": 5, "placement": "random", "sep_pad": 3},
    # Most aggressive random variant.  At 50 px^2 a radius-4 disc is only 1.7x
    # COUNT_MIN_AREA, so verify the counter still sees every object on ground
    # truth before trusting any accuracy measured on it.
    "r4":      {"radius": 4, "placement": "random", "sep_pad": 1},
    "grid_r4": {"radius": 4, "placement": "grid", "gap": 8},
    "grid_r5": {"radius": 5, "placement": "grid", "gap": 8},
    # Small objects plus real clearance.  Radius 4 alone ceilings at 89.6%
    # because neighbours still merge at n=7; the padding is what lifts it.
    # Radius 3 is not an option: 28 px^2 is below COUNT_MIN_AREA of 30.
    "r4_sep":  {"radius": 4, "placement": "random", "sep_pad": 3},
}


def random_centers(count, radius, sep_pad, rng):
    """Rejection sampling, same shape as the canonical generator but with the
    separation margin exposed so it can be widened independently of radius."""
    lo, hi = -IMAGE_SIZE // 2 + radius, IMAGE_SIZE // 2 - radius
    min_sq = (2 * (radius + sep_pad)) ** 2
    for _ in range(4096):
        centers = []
        for _ in range(count):
            for _ in range(4096):
                cand = (int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
                if all((cand[0] - x) ** 2 + (cand[1] - y) ** 2 >= min_sq for x, y in centers):
                    centers.append(cand)
                    break
            else:
                break
        if len(centers) == count:
            return centers
    raise RuntimeError(f"could not place {count} circles of radius {radius}")


def grid_centers(count, radius, rng, gap=8):
    """Snap to lattice cells, so the model chooses which cells to fill rather
    than where to put anything.

    `gap` is the clearance between neighbouring discs.  The first version used
    gap=2, which put lattice neighbours at minimum separation and perfectly
    axis-aligned: watershed merged them and the counter could only read 29-52%
    of the ground truth.  Grid placement needs MORE clearance than random
    placement, not less, because neighbours are guaranteed adjacent.
    """
    cell = 2 * radius + gap
    per_axis = IMAGE_SIZE // cell
    if per_axis ** 2 < count:
        raise RuntimeError(f"grid holds {per_axis ** 2} cells, need {count}")
    picks = rng.choice(per_axis ** 2, size=count, replace=False)
    offset = (IMAGE_SIZE - per_axis * cell) / 2.0
    out = []
    for p in picks:
        row, col = divmod(int(p), per_axis)
        cx = offset + col * cell + cell / 2.0 - IMAGE_SIZE / 2.0
        cy = offset + row * cell + cell / 2.0 - IMAGE_SIZE / 2.0
        out.append((int(round(cx)), int(round(cy))))
    return out


def build(name, spec, out_dir, force=False):
    if (out_dir / "dataset_metadata.json").exists() and not force:
        print(f"[skip] {name} exists")
        return
    rng = np.random.default_rng(SEED)
    radius = spec["radius"]
    per_class = distribute_counts(TOTAL_IMAGES, COUNTS)

    for count in tqdm(COUNTS, desc=name):
        class_dir = out_dir / str(count)
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(per_class[count]):
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), background_color(rng))
            draw = ImageDraw.Draw(image)
            if spec["placement"] == "grid":
                centers = grid_centers(count, radius, rng, spec.get("gap", 8))
            else:
                centers = random_centers(count, radius, spec["sep_pad"], rng)
            for cx, cy in centers:
                bbox = [IMAGE_SIZE / 2 + cx - radius, IMAGE_SIZE / 2 + cy - radius,
                        IMAGE_SIZE / 2 + cx + radius, IMAGE_SIZE / 2 + cy + radius]
                draw.ellipse(bbox, fill=foreground_color(rng))
            finalize(image, rng).save(class_dir / f"count_{idx:05d}.png")

    # Key names match what evaluate_table2.build_count_queries expects, so the
    # paper-locked evaluator runs on these unchanged.
    (out_dir / "dataset_metadata.json").write_text(json.dumps({
        "dataset_type": "count",
        "counts": COUNTS,
        "circle_radius": radius,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "total_images": sum(per_class.values()),
        "class_counts": {str(k): v for k, v in per_class.items()},
        "variant": name,
        "placement": spec["placement"],
        "sep_pad": spec.get("sep_pad"),
    }, indent=2))
    print(f"wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for name in a.variants:
        build(name, VARIANTS[name], DATASET_ROOT / "count_variants" / name, a.force)


if __name__ == "__main__":
    main()
