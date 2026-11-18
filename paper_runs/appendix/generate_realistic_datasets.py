#!/usr/bin/env python
"""Generate the visually complex VGL datasets (textured backgrounds, shading,
varied colours).

Same skill grids, same folder-name convention and same 10k budget as the
paper's canonical datasets, so training and evaluation code paths are unchanged
-- only the appearance of the pixels differs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.manifest import DATASET_ROOT, DATASET_SEED
from paper_runs.appendix.realistic import (
    render_arrow, render_circle, render_circles, validate_extraction,
)
from paper_runs.table2.generate_canonical_datasets import (
    distribute_counts, position_folder_name, sample_non_overlapping_centers,
)

IMAGE_SIZE = 64
TOTAL_IMAGES = 10000


def _write(output_dir, class_name, index, array):
    class_dir = output_dir / str(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(class_dir / f"img_{index:05d}.png")


def build_size(output_dir, rng):
    radii = list(range(5, 21))
    counts = distribute_counts(TOTAL_IMAGES, radii)
    for radius in tqdm(radii, desc="size"):
        for i in range(counts[radius]):
            _write(output_dir, radius, i, render_circle(radius, 0, 0, IMAGE_SIZE, rng))
    return {"dataset_type": "size", "train_radii": radii}


def build_position(output_dir, rng):
    coords = np.linspace(-18.0, 18.0, 32)
    positions = [(float(x), float(y)) for x in coords for y in coords]
    counts = distribute_counts(TOTAL_IMAGES, positions)
    for x, y in tqdm(positions, desc="position"):
        for i in range(counts[(x, y)]):
            _write(output_dir, position_folder_name(x, y), i,
                   render_circle(8, x, y, IMAGE_SIZE, rng))
    return {"dataset_type": "position", "positions": positions, "circle_radius": 8,
            "grid_min": -18.0, "grid_max": 18.0, "positions_per_axis": 32}


def build_rotation(output_dir, rng):
    angles = list(range(45, 316))
    counts = distribute_counts(TOTAL_IMAGES, angles)
    for angle in tqdm(angles, desc="rotation"):
        for i in range(counts[angle]):
            _write(output_dir, angle, i, render_arrow(float(angle), IMAGE_SIZE, 20, rng))
    return {"dataset_type": "rotation", "angles": angles, "shape_size": 20,
            "min_angle": 45, "max_angle": 315, "step_degrees": 1, "shape_type": "arrow"}


def build_count(output_dir, rng):
    counts_list = list(range(2, 8))
    counts = distribute_counts(TOTAL_IMAGES, counts_list)
    for count in tqdm(counts_list, desc="count"):
        for i in range(counts[count]):
            centers = sample_non_overlapping_centers(count, 8, IMAGE_SIZE, rng)
            _write(output_dir, count, i, render_circles(centers, 8, IMAGE_SIZE, rng))
    return {"dataset_type": "count", "counts": counts_list, "circle_radius": 8}


BUILDERS = {"size": build_size, "position": build_position,
            "rotation": build_rotation, "count": build_count}


def main():
    parser = argparse.ArgumentParser(description="Generate visually complex VGL datasets.")
    parser.add_argument("--skills", nargs="+", default=list(BUILDERS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report = validate_extraction(n=200)
    print("extractor validation on known geometry:", flush=True)
    for key, value in report.items():
        print(f"    {key:>26}: {value:.3f}", flush=True)

    for skill in args.skills:
        output_dir = DATASET_ROOT / "realistic" / skill
        if not args.force and (output_dir / "dataset_metadata.json").exists():
            print(f"[skip] {output_dir} exists", flush=True)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(DATASET_SEED)
        meta = BUILDERS[skill](output_dir, rng)
        meta.update({"image_size": IMAGE_SIZE, "total_images": TOTAL_IMAGES,
                     "seed": DATASET_SEED, "render": "realistic",
                     "extractor_validation": report})
        (output_dir / "dataset_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
