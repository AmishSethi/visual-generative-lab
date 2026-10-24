#!/usr/bin/env python
"""Generate appendix datasets.

Four families:

  scaling   80k-image pools for all four skills at 64x64, plus nested
            10k/20k/40k subsets built as symlink trees.  Nesting means the
            data-scaling curve varies exactly one thing: the number of unique
            training images.

  hires     10k-image datasets for all four skills at 128x128, with every
            length doubled so the renders are geometrically identical to the
            64x64 ones.

  ranges    10k-image datasets at 64x64 whose training support is narrower
            than or shifted from the paper's (RANGE_SPECS in manifest.py),
            for testing whether the extrapolation boundary follows the
            support.

  count_random  the 10k-image count set of radius-8 circles at random positions
            (not the radius-4 lattice of Table 2) on which the text-conditioned
            count model and its numeric comparator train.

All four reuse the paper-locked renderers in
paper_runs/table2/generate_canonical_datasets.py so pixel statistics (colour
palettes, antialiasing, noise) match the paper exactly.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.generate_canonical_datasets import GENERATORS
from paper_runs.appendix.manifest import (
    DATASET_ROOT,
    DATASET_SEED,
    HIRES_SPECS,
    RANGE_SPECS,
    SCALING_SIZES,
    SCALING_SKILLS,
)

# 64x64 specs. size and position equal the Table 2 specs apart from total_images;
# rotation and count differ as noted below.
BASE_SPECS = {
    "size": {
        "image_size": 64,
        "train_radii": list(range(5, 21)),
        "seed": DATASET_SEED,
    },
    "position": {
        "image_size": 64,
        "circle_radius": 8,
        "grid_min": -18.0,
        "grid_max": 18.0,
        "positions_per_axis": 32,
        "seed": DATASET_SEED,
    },
    # Matches paper_runs.table2.manifest DATASET_SPECS["rotation"] apart from
    # total_images, which build_scaling overrides, and the fixed arrow and
    # background colours of the Table 2 set: this one uses the random palettes
    # and noise.
    "rotation": {
        "image_size": 64,
        "shape_size": 20,
        "min_angle": 45,
        "max_angle": 315,
        "step_degrees": 1,
        "shape_type": "arrow",
        "seed": DATASET_SEED,
    },
    # Radius-8 circles at random positions, not the radius-4 lattice of the Table 2 count set.
    "count": {
        "image_size": 64,
        "counts": list(range(2, 8)),
        "circle_radius": 8,
        "seed": DATASET_SEED,
    },
}


def generate_pool(skill: str, output_dir: Path, total_images: int) -> None:
    spec = dict(BASE_SPECS[skill])
    spec["total_images"] = total_images
    output_dir.mkdir(parents=True, exist_ok=True)
    GENERATORS[skill](output_dir, spec)


def make_nested_subset(pool_dir: Path, subset_dir: Path, fraction: float) -> dict:
    """Symlink the first `fraction` of each class directory into subset_dir.

    Taking a per-class prefix keeps the subset class-balanced and guarantees
    the 10k subset is nested in 20k, 20k in 40k, and 40k in the 80k pool.
    """
    if subset_dir.exists():
        for path in sorted(subset_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                path.rmdir()
    subset_dir.mkdir(parents=True, exist_ok=True)

    class_counts = {}
    total = 0
    for class_dir in sorted(p for p in pool_dir.iterdir() if p.is_dir()):
        images = sorted(class_dir.glob("*.png"))
        keep = max(1, int(round(len(images) * fraction)))
        target = subset_dir / class_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for image_path in images[:keep]:
            (target / image_path.name).symlink_to(image_path.resolve())
        class_counts[class_dir.name] = keep
        total += keep

    meta_src = pool_dir / "dataset_metadata.json"
    metadata = json.loads(meta_src.read_text()) if meta_src.exists() else {}
    metadata.update(
        {
            "derived_from": str(pool_dir),
            "subset_fraction": fraction,
            "class_counts": class_counts,
            "total_images": total,
        }
    )
    (subset_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def build_scaling(skills, force: bool) -> None:
    largest = max(SCALING_SIZES)
    for skill in skills:
        pool_dir = DATASET_ROOT / "scaling" / f"{skill}_{largest // 1000}k"
        if force or not (pool_dir / "dataset_metadata.json").exists():
            print(f"[scaling] generating {skill} pool of {largest} images -> {pool_dir}", flush=True)
            generate_pool(skill, pool_dir, largest)
        else:
            print(f"[scaling] reusing existing pool {pool_dir}", flush=True)

        for n_images in SCALING_SIZES:
            if n_images == largest:
                continue
            subset_dir = DATASET_ROOT / "scaling" / f"{skill}_{n_images // 1000}k"
            meta = make_nested_subset(pool_dir, subset_dir, n_images / largest)
            print(f"[scaling] {subset_dir.name}: {meta['total_images']} images", flush=True)


def build_hires(skills, force: bool) -> None:
    for skill in skills:
        output_dir = DATASET_ROOT / "hires128" / skill
        if not force and (output_dir / "dataset_metadata.json").exists():
            print(f"[hires] reusing existing {output_dir}", flush=True)
            continue
        print(f"[hires] generating {skill} at 128x128 -> {output_dir}", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        GENERATORS[skill](output_dir, HIRES_SPECS[skill])


def build_ranges(names, force: bool) -> None:
    for name in names:
        spec = dict(RANGE_SPECS[name])
        skill = spec.pop("skill")
        output_dir = DATASET_ROOT / "ranges" / name
        if not force and (output_dir / "dataset_metadata.json").exists():
            print(f"[range] reusing existing {output_dir}", flush=True)
            continue
        print(f"[range] generating {name} ({skill}) -> {output_dir}", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        GENERATORS[skill](output_dir, spec)


def build_count_random(force: bool) -> None:
    output_dir = DATASET_ROOT / "count_random"
    if not force and (output_dir / "dataset_metadata.json").exists():
        print(f"[count_random] reusing existing {output_dir}", flush=True)
        return
    print(f"[count_random] generating 10000 count images -> {output_dir}", flush=True)
    generate_pool("count", output_dir, 10000)


def main():
    parser = argparse.ArgumentParser(description="Generate appendix datasets.")
    parser.add_argument("--family", required=True, choices=["scaling", "hires", "ranges", "count_random"])
    parser.add_argument("--skills", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.family == "scaling":
        build_scaling(args.skills or list(SCALING_SKILLS), args.force)
    elif args.family == "ranges":
        build_ranges(args.skills or list(RANGE_SPECS.keys()), args.force)
    elif args.family == "count_random":
        build_count_random(args.force)
    else:
        build_hires(args.skills or list(HIRES_SPECS.keys()), args.force)


if __name__ == "__main__":
    main()
