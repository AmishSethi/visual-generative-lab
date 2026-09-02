#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.manifest import DATASET_ROOT, DATASET_SPECS


FOREGROUND_PALETTE = np.array(
    [
        (240, 90, 90),
        (80, 170, 255),
        (90, 210, 140),
        (255, 175, 70),
        (215, 105, 240),
        (80, 210, 210),
        (255, 120, 170),
        (160, 120, 255),
    ],
    dtype=np.float32,
)

BACKGROUND_PALETTE = np.array(
    [
        (255, 255, 255),
        (247, 247, 244),
        (245, 249, 255),
        (250, 244, 255),
        (255, 247, 242),
        (244, 252, 246),
    ],
    dtype=np.float32,
)


def distribute_counts(total, values):
    base = total // len(values)
    remainder = total % len(values)
    return {value: base + (idx < remainder) for idx, value in enumerate(values)}


def foreground_color(rng):
    base = FOREGROUND_PALETTE[rng.integers(len(FOREGROUND_PALETTE))]
    color = np.clip(base + rng.normal(0.0, 10.0, size=3), 0, 255)
    return tuple(color.astype(np.uint8).tolist())


def background_color(rng):
    base = BACKGROUND_PALETTE[rng.integers(len(BACKGROUND_PALETTE))]
    color = np.clip(base + rng.normal(0.0, 5.0, size=3), 220, 255)
    return tuple(color.astype(np.uint8).tolist())


def add_image_noise(image, rng):
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr + rng.normal(0.0, 5.0, size=arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def finalize(image, rng):
    return add_image_noise(image, rng)


def render_circle(radius, center_x, center_y, image_size, fill_rgb, background_rgb, antialias=4):
    large_size = image_size * antialias
    large_radius = radius * antialias
    large_center_x = large_size // 2 + center_x * antialias
    large_center_y = large_size // 2 + center_y * antialias

    image = Image.new("RGB", (large_size, large_size), background_rgb)
    draw = ImageDraw.Draw(image)
    bbox = [
        large_center_x - large_radius,
        large_center_y - large_radius,
        large_center_x + large_radius,
        large_center_y + large_radius,
    ]
    draw.ellipse(bbox, fill=fill_rgb, outline=fill_rgb)
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def render_arrow(angle_degrees, image_size, shape_size, fill_rgb, background_rgb, antialias=4):
    large_size = image_size * antialias
    large_shape_size = shape_size * antialias
    image = Image.new("RGB", (large_size, large_size), background_rgb)
    draw = ImageDraw.Draw(image)
    center_x = large_size // 2
    center_y = large_size // 2
    angle_rad = math.radians(-angle_degrees + 90.0)

    arrow_points = [
        (0.0, -large_shape_size * 0.5),
        (-large_shape_size * 0.25, -large_shape_size * 0.2),
        (-large_shape_size * 0.15, -large_shape_size * 0.2),
        (-large_shape_size * 0.10, large_shape_size * 0.4),
        (0.0, large_shape_size * 0.5),
        (large_shape_size * 0.10, large_shape_size * 0.4),
        (large_shape_size * 0.15, -large_shape_size * 0.2),
        (large_shape_size * 0.25, -large_shape_size * 0.2),
    ]

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated = []
    for x_coord, y_coord in arrow_points:
        rot_x = x_coord * cos_a - y_coord * sin_a
        rot_y = x_coord * sin_a + y_coord * cos_a
        rotated.append((center_x + rot_x, center_y + rot_y))

    draw.polygon(rotated, fill=fill_rgb, outline=fill_rgb)
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def sample_non_overlapping_centers(count, radius, image_size, rng):
    valid_min = -image_size // 2 + radius
    valid_max = image_size // 2 - radius
    min_distance_sq = (2 * (radius + 1)) ** 2
    max_restarts = 4096
    max_attempts_per_circle = 4096

    for _ in range(max_restarts):
        centers = []
        for _ in range(count):
            for _ in range(max_attempts_per_circle):
                candidate = (
                    int(rng.integers(valid_min, valid_max + 1)),
                    int(rng.integers(valid_min, valid_max + 1)),
                )
                if all(
                    (candidate[0] - x_coord) ** 2 + (candidate[1] - y_coord) ** 2 >= min_distance_sq
                    for x_coord, y_coord in centers
                ):
                    centers.append(candidate)
                    break
            else:
                break
        if len(centers) == count:
            return centers

    raise RuntimeError(f"Could not place {count} circles without overlap.")


def position_folder_name(x_coord, y_coord):
    x_str = f"{x_coord:.6f}".rstrip("0").rstrip(".").replace("-", "neg").replace(".", "p")
    y_str = f"{y_coord:.6f}".rstrip("0").rstrip(".").replace("-", "neg").replace(".", "p")
    return f"{x_str}_{y_str}"


def write_metadata(output_dir, metadata):
    with open(output_dir / "dataset_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def generate_size_dataset(output_dir, spec):
    rng = np.random.default_rng(spec["seed"])
    radii = spec["train_radii"]
    class_counts = distribute_counts(spec["total_images"], radii)

    for radius in tqdm(radii, desc="size"):
        class_dir = output_dir / str(radius)
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(class_counts[radius]):
            image = render_circle(
                radius=radius,
                center_x=0,
                center_y=0,
                image_size=spec["image_size"],
                fill_rgb=foreground_color(rng),
                background_rgb=background_color(rng),
            )
            finalize(image, rng).save(class_dir / f"circle_{sample_idx:05d}.png")

    write_metadata(
        output_dir,
        {
            **spec,
            "dataset_type": "size",
            "class_counts": class_counts,
            "total_images": sum(class_counts.values()),
        },
    )


def generate_position_dataset(output_dir, spec):
    rng = np.random.default_rng(spec["seed"])
    coords = np.linspace(spec["grid_min"], spec["grid_max"], spec["positions_per_axis"])
    positions = [(float(x_coord), float(y_coord)) for x_coord in coords for y_coord in coords]
    class_counts = distribute_counts(spec["total_images"], positions)

    for x_coord, y_coord in tqdm(positions, desc="position"):
        class_dir = output_dir / position_folder_name(x_coord, y_coord)
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(class_counts[(x_coord, y_coord)]):
            image = render_circle(
                radius=spec["circle_radius"],
                center_x=x_coord,
                center_y=y_coord,
                image_size=spec["image_size"],
                fill_rgb=foreground_color(rng),
                background_rgb=background_color(rng),
            )
            finalize(image, rng).save(class_dir / f"circle_{sample_idx:05d}.png")

    write_metadata(
        output_dir,
        {
            **spec,
            "dataset_type": "position",
            "class_counts": {
                position_folder_name(x_coord, y_coord): class_counts[(x_coord, y_coord)]
                for x_coord, y_coord in positions
            },
            "positions": positions,
            "total_positions": len(positions),
            "total_images": sum(class_counts.values()),
        },
    )


def generate_rotation_dataset(output_dir, spec):
    rng = np.random.default_rng(spec["seed"])
    angles = list(range(spec["min_angle"], spec["max_angle"] + 1, spec["step_degrees"]))
    class_counts = distribute_counts(spec["total_images"], angles)

    for angle in tqdm(angles, desc="rotation"):
        class_dir = output_dir / str(angle)
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(class_counts[angle]):
            image = render_arrow(
                angle_degrees=angle,
                image_size=spec["image_size"],
                shape_size=spec["shape_size"],
                fill_rgb=foreground_color(rng),
                background_rgb=background_color(rng),
            )
            finalize(image, rng).save(class_dir / f"arrow_{sample_idx:05d}.png")

    write_metadata(
        output_dir,
        {
            **spec,
            "dataset_type": "rotation",
            "angles": angles,
            "num_angles": len(angles),
            "class_counts": class_counts,
            "total_images": sum(class_counts.values()),
        },
    )


def generate_count_dataset(output_dir, spec):
    rng = np.random.default_rng(spec["seed"])
    counts = spec["counts"]
    class_counts = distribute_counts(spec["total_images"], counts)

    for count in tqdm(counts, desc="count"):
        class_dir = output_dir / str(count)
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(class_counts[count]):
            image = Image.new("RGB", (spec["image_size"], spec["image_size"]), background_color(rng))
            draw = ImageDraw.Draw(image)
            centers = sample_non_overlapping_centers(
                count=count,
                radius=spec["circle_radius"],
                image_size=spec["image_size"],
                rng=rng,
            )
            for center_x, center_y in centers:
                fill_rgb = foreground_color(rng)
                bbox = [
                    spec["image_size"] / 2 + center_x - spec["circle_radius"],
                    spec["image_size"] / 2 + center_y - spec["circle_radius"],
                    spec["image_size"] / 2 + center_x + spec["circle_radius"],
                    spec["image_size"] / 2 + center_y + spec["circle_radius"],
                ]
                draw.ellipse(bbox, fill=fill_rgb, outline=fill_rgb)
            finalize(image, rng).save(class_dir / f"count_{sample_idx:05d}.png")

    write_metadata(
        output_dir,
        {
            **spec,
            "dataset_type": "count",
            "class_counts": class_counts,
            "total_images": sum(class_counts.values()),
        },
    )


GENERATORS = {
    "size": generate_size_dataset,
    "position": generate_position_dataset,
    "rotation": generate_rotation_dataset,
    "count": generate_count_dataset,
}


def main():
    parser = argparse.ArgumentParser(description="Generate canonical Table-2 datasets.")
    parser.add_argument(
        "--skills",
        nargs="+",
        default=list(GENERATORS.keys()),
        choices=sorted(GENERATORS.keys()),
        help="Datasets to generate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root directory for generated datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and regenerate an existing dataset directory.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    for skill in args.skills:
        output_dir = args.output_root / skill
        if output_dir.exists() and args.force:
            for path in sorted(output_dir.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                else:
                    path.rmdir()
        output_dir.mkdir(parents=True, exist_ok=True)
        GENERATORS[skill](output_dir, DATASET_SPECS[skill])


if __name__ == "__main__":
    main()
