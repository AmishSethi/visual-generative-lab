#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.diffusion import create_diffusion
from vgl.eval_position import (
    calculate_position_metrics_enhanced,
    load_model as load_position_model,
    generate_samples as generate_position_samples,
)
from vgl.eval_radius import (
    calculate_circle_metrics,
    generate_samples as generate_radius_samples,
    load_model as load_radius_model,
)
from vgl.eval_rotation import (
    calculate_rotation_metrics,
    generate_samples as generate_rotation_samples,
    load_model as load_rotation_model,
)
from vgl.flow_matching import FlowMatching
from paper_runs.table2.generate_canonical_datasets import render_circle
from paper_runs.table2.manifest import DATASET_ROOT, EVAL_ROOT, RESULTS_ROOT, TABLE_ROWS


SIZE_IOU_THRESHOLD = 0.90
POSITION_ERROR_THRESHOLD = 2.0
ROTATION_ERROR_THRESHOLD = 5.0


def calculate_rotation_metrics_robust(
    image: np.ndarray,
    expected_angle: float,
    **kwargs,
) -> dict:
    """
    Detect arrow angle using multi-scale template matching — the method
    that produced the paper's rotation numbers (verified with paper checkpoint).
    """
    from vgl.eval_rotation import extract_arrow_angle_template_matching

    detected, ok = extract_arrow_angle_template_matching(image)
    if not ok:
        return {"angle_error": float("inf"), "angle_error_abs": float("inf"),
                "detection_success": False, "expected_angle": expected_angle,
                "detected_angle": 0.0}

    # Wrap error to [-180, 180]
    diff = float(detected) - expected_angle
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    return {
        "angle_error": diff,
        "angle_error_abs": abs(diff),
        "detection_success": True,
        "expected_angle": expected_angle,
        "detected_angle": float(detected),
    }
SIZE_EXTRAP_MAX = 30  # Table 2 protocol (radii 21..30); the base grid used 25
SIZE_EXTRAP_MIN = 1
POSITION_EXTRAP_MARGIN = 6.0
POSITION_EXTRAP_MIN_MARGIN = 4.0  # Table 2 protocol (L_inf >= 4 px outside training); the base grid used 0.0
COUNT_EXTRAP_VALUES = (0.0, 1.0, 8.0, 9.0)
COUNT_MIN_AREA = 30


@dataclass(frozen=True)
class RunSelection:
    run_dir: Path
    checkpoint: Path
    run_config: dict


def round_scalar(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def round_pair(pair: tuple[float, float], digits: int = 6) -> tuple[float, float]:
    return (round_scalar(pair[0], digits), round_scalar(pair[1], digits))


def serialize_query(query):
    if isinstance(query, tuple):
        return f"{query[0]:.6f},{query[1]:.6f}"
    return f"{float(query):.6f}"


def load_json(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def latest_finished_run(skill: str, variant: str, seed: int) -> RunSelection:
    seed_root = RESULTS_ROOT / skill / variant / f"seed_{seed}"
    if not seed_root.exists():
        raise FileNotFoundError(f"Missing seed directory: {seed_root}")

    candidates = []
    for child in sorted(seed_root.iterdir()):
        if not child.is_dir():
            continue
        prefix = child.name.split("-", 1)[0]
        if not prefix.isdigit():
            continue
        checkpoints = sorted((child / "checkpoints").glob("final_*.pt"))
        if checkpoints:
            candidates.append((int(prefix), child, checkpoints[-1]))

    if not candidates:
        raise FileNotFoundError(f"No finished run found under {seed_root}")

    _, run_dir, checkpoint = max(candidates, key=lambda item: item[0])
    run_config = load_json(run_dir / "run_config.json")
    return RunSelection(run_dir=run_dir, checkpoint=checkpoint, run_config=run_config)


def load_dataset_metadata(skill: str) -> dict:
    metadata_path = DATASET_ROOT / skill / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    return load_json(metadata_path)


def build_size_queries(
    metadata: dict,
    extrap_min: int = SIZE_EXTRAP_MIN,
    extrap_max: int = SIZE_EXTRAP_MAX,
) -> dict[str, list[float]]:
    train = [round_scalar(radius) for radius in metadata["train_radii"]]
    interp = [
        round_scalar((train[idx] + train[idx + 1]) / 2.0)
        for idx in range(len(train) - 1)
    ]
    train_min = int(round(train[0]))
    train_max = int(round(train[-1]))
    extrap = [
        float(radius)
        for radius in range(extrap_min, extrap_max + 1)
        if radius < train_min or radius > train_max
    ]
    return {"train": train, "interp": interp, "extra": extrap}


def build_count_queries(metadata: dict) -> dict[str, list[float]]:
    train = [round_scalar(count) for count in metadata["counts"]]
    interp = [
        round_scalar((train[idx] + train[idx + 1]) / 2.0)
        for idx in range(len(train) - 1)
    ]
    extra = [float(value) for value in COUNT_EXTRAP_VALUES]
    return {"train": train, "interp": interp, "extra": extra}


def build_rotation_queries(metadata: dict) -> dict[str, list[float]]:
    train = [round_scalar(angle) for angle in metadata["angles"]]
    interp = [
        round_scalar((train[idx] + train[idx + 1]) / 2.0)
        for idx in range(len(train) - 1)
    ]
    train_min = int(round(train[0]))
    train_max = int(round(train[-1]))
    extra = [float(angle) for angle in range(0, train_min)] + [
        float(angle) for angle in range(train_max + 1, 360)
    ]
    return {"train": train, "interp": interp, "extra": extra}


def build_position_queries(
    metadata: dict,
    extrap_margin: float = POSITION_EXTRAP_MARGIN,
    extrap_min_margin: float = POSITION_EXTRAP_MIN_MARGIN,
) -> dict[str, list[tuple[float, float]]]:
    train_positions = [round_pair(tuple(position)) for position in metadata["positions"]]

    x_values = sorted({position[0] for position in train_positions})
    y_values = sorted({position[1] for position in train_positions})
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("Position metadata must contain at least two grid coordinates per axis.")

    interp_x = [
        round_scalar((x_values[idx] + x_values[idx + 1]) / 2.0)
        for idx in range(len(x_values) - 1)
    ]
    interp_y = [
        round_scalar((y_values[idx] + y_values[idx + 1]) / 2.0)
        for idx in range(len(y_values) - 1)
    ]
    interp = [round_pair((x_coord, y_coord)) for x_coord in interp_x for y_coord in interp_y]

    valid_min = -metadata["image_size"] / 2.0 + metadata["circle_radius"]
    valid_max = metadata["image_size"] / 2.0 - metadata["circle_radius"]
    step_x = x_values[1] - x_values[0]
    step_y = y_values[1] - y_values[0]
    train_min_x = x_values[0]
    train_max_x = x_values[-1]
    train_min_y = y_values[0]
    train_max_y = y_values[-1]
    extrap_limit_min = train_min_x - extrap_margin
    extrap_limit_max = train_max_x + extrap_margin

    extended_x = list(x_values)
    probe = train_min_x - step_x
    while probe >= valid_min - 1e-6 and probe >= extrap_limit_min - 1e-6:
        extended_x.insert(0, round_scalar(probe))
        probe -= step_x
    probe = train_max_x + step_x
    while probe <= valid_max + 1e-6 and probe <= extrap_limit_max + 1e-6:
        extended_x.append(round_scalar(probe))
        probe += step_x

    extended_y = list(y_values)
    probe = train_min_y - step_y
    while probe >= valid_min - 1e-6 and probe >= extrap_limit_min - 1e-6:
        extended_y.insert(0, round_scalar(probe))
        probe -= step_y
    probe = train_max_y + step_y
    while probe <= valid_max + 1e-6 and probe <= extrap_limit_max + 1e-6:
        extended_y.append(round_scalar(probe))
        probe += step_y

    extrap = []
    for x_coord in extended_x:
        for y_coord in extended_y:
            in_train_box = (
                train_min_x - 1e-6 <= x_coord <= train_max_x + 1e-6
                and train_min_y - 1e-6 <= y_coord <= train_max_y + 1e-6
            )
            if in_train_box:
                continue
            l_inf_dist = max(
                abs(x_coord) - train_max_x,
                abs(y_coord) - train_max_y,
                0.0,
            )
            if l_inf_dist + 1e-6 < extrap_min_margin:
                continue
            extrap.append(round_pair((x_coord, y_coord)))

    return {"train": train_positions, "interp": interp, "extra": extrap}


def maybe_truncate_queries(queries: dict[str, list], max_conditions_per_split: int | None) -> dict[str, list]:
    if max_conditions_per_split is None:
        return queries
    return {
        split_name: values[:max_conditions_per_split]
        for split_name, values in queries.items()
    }


def build_common_run_metadata(selection: RunSelection) -> tuple[dict, dict]:
    run_args = selection.run_config["args"]
    folder_name = selection.run_dir.name
    is_latent = run_args.get("use_latent_diffusion", False)
    # scripts/eval_radius.py detects VAE models by checking for "vae" in folder_name.
    # Ensure the folder name signals this so the loader uses the correct input size.
    if is_latent and "vae" not in folder_name.lower():
        folder_name = f"vae_{folder_name}"
    return run_args, {
        "folder_name": folder_name,
        "architecture": run_args["architecture"],
        "model": run_args["model"],
        "image_size": run_args["image_size"],
        "conditioning_method": run_args["conditioning_method"],
        "use_flow_matching": run_args.get("use_flow_matching", False),
        "use_latent_diffusion": is_latent,
    }


def build_size_config(selection: RunSelection) -> dict:
    run_args, common = build_common_run_metadata(selection)
    return {
        "folder_name": common["folder_name"],
        "ckpt_path": str(selection.checkpoint),
        "architecture": common["architecture"],
        "model_name": common["model"],
        "image_size": common["image_size"],
        "radius_embedding_type": run_args["radius_embedding_type"],
        "conditioning_method": common["conditioning_method"],
        "radius_dropout_prob": run_args["radius_dropout_prob"],
        "null_radius": run_args["null_radius"],
        "null_embedding_type": run_args["null_embedding_type"],
        "use_flow_matching": common["use_flow_matching"],
        "use_latent_diffusion": common["use_latent_diffusion"],
        "radius_text_table": run_args.get("radius_text_table"),
        "use_cfg": False,
    }


def build_count_config(selection: RunSelection) -> dict:
    run_args, common = build_common_run_metadata(selection)
    return {
        "folder_name": common["folder_name"],
        "ckpt_path": str(selection.checkpoint),
        "architecture": common["architecture"],
        "model_name": common["model"],
        "image_size": common["image_size"],
        "radius_embedding_type": run_args["count_embedding_type"],
        "conditioning_method": common["conditioning_method"],
        "radius_dropout_prob": run_args["count_dropout_prob"],
        "null_radius": run_args["null_count"],
        "null_embedding_type": run_args["null_embedding_type"],
        "use_flow_matching": common["use_flow_matching"],
        "use_latent_diffusion": common["use_latent_diffusion"],
        "radius_text_table": run_args.get("radius_text_table"),
        "use_cfg": False,
    }


def build_position_config(selection: RunSelection, metadata: dict) -> dict:
    run_args, common = build_common_run_metadata(selection)
    return {
        "folder_name": common["folder_name"],
        "ckpt_path": str(selection.checkpoint),
        "architecture": common["architecture"],
        "model_type": common["model"],
        "image_size": common["image_size"],
        "position_embedding_type": run_args["position_embedding_type"],
        "conditioning_method": common["conditioning_method"],
        "position_dropout_prob": run_args["position_dropout_prob"],
        "null_position": tuple(run_args["null_position"]),
        "null_embedding_type": run_args["null_embedding_type"],
        "use_flow_matching": common["use_flow_matching"],
        "use_latent_diffusion": common["use_latent_diffusion"],
        "use_cfg": False,
        "circle_radius": metadata["circle_radius"],
        "train_min_pos": metadata["grid_min"],
        "train_max_pos": metadata["grid_max"],
        "square_size": metadata["grid_max"] - metadata["grid_min"],
        "density": metadata["positions_per_axis"],
        "valid_min": -metadata["image_size"] / 2.0 + metadata["circle_radius"],
        "valid_max": metadata["image_size"] / 2.0 - metadata["circle_radius"],
    }


def build_rotation_config(selection: RunSelection, metadata: dict) -> dict:
    run_args, common = build_common_run_metadata(selection)
    return {
        "folder_name": common["folder_name"],
        "ckpt_path": str(selection.checkpoint),
        "architecture": common["architecture"],
        "model_type": common["model"],
        "image_size": common["image_size"],
        "rotation_embedding_type": run_args["rotation_embedding_type"],
        "rotation_text_table": run_args.get("rotation_text_table"),
        "conditioning_method": common["conditioning_method"],
        "rotation_dropout_prob": run_args["rotation_dropout_prob"],
        "null_angle": run_args["null_angle"],
        "null_embedding_type": run_args["null_embedding_type"],
        "use_flow_matching": common["use_flow_matching"],
        "use_latent_diffusion": common["use_latent_diffusion"],
        "use_cfg": False,
        "shape_type": metadata["shape_type"],
    }


def create_sampler(config: dict, num_sampling_steps: int):
    if config.get("use_flow_matching", False):
        return FlowMatching(sigma_min=0.0, sigma_data=1.0, use_sigmoid_time=True)
    if config.get("architecture") == "songunet":
        return create_diffusion(str(num_sampling_steps), learn_sigma=False)
    return create_diffusion(str(num_sampling_steps))


def gt_circle_image(radius: float, image_size: int) -> np.ndarray:
    image = render_circle(
        radius=radius,
        center_x=0,
        center_y=0,
        image_size=image_size,
        fill_rgb=(255, 128, 100),
        background_rgb=(255, 255, 255),
    )
    return np.asarray(image)


def detect_count_with_watershed(image: np.ndarray, min_area: int = COUNT_MIN_AREA) -> int:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if binary.max() == 0:
        return 0

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return 0

    markers = markers + 1
    markers[unknown == 255] = 0
    watershed_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_input, markers)

    count = 0
    for label in np.unique(markers):
        if label <= 1:
            continue
        if int(np.count_nonzero(markers == label)) > min_area:
            count += 1
    return count


def evaluate_size_split(
    values: list[float],
    samples_dict: dict[float, list[np.ndarray]],
    config: dict,
) -> dict:
    total = 0
    successes = 0
    all_ious = []
    condition_accuracies = {}
    for value in values:
        gt_image = gt_circle_image(value, config["image_size"])
        split_successes = 0
        split_ious = []
        for sample in samples_dict[value]:
            metrics = calculate_circle_metrics(sample, gt_image, value)
            success = metrics["iou"] >= SIZE_IOU_THRESHOLD
            split_successes += int(success)
            successes += int(success)
            total += 1
            split_ious.append(float(metrics["iou"]))
            all_ious.append(float(metrics["iou"]))
        condition_accuracies[Serialize_key := serialize_query(value)] = {
            "accuracy": 100.0 * split_successes / len(samples_dict[value]),
            "mean_iou": float(np.mean(split_ious)),
            "num_samples": len(samples_dict[value]),
        }
    return {
        "accuracy": 100.0 * successes / total if total else 0.0,
        "successes": successes,
        "total_samples": total,
        "num_conditions": len(values),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "threshold": SIZE_IOU_THRESHOLD,
        "condition_accuracies": condition_accuracies,
    }


def evaluate_count_split(
    values: list[float],
    samples_dict: dict[float, list[np.ndarray]],
) -> dict:
    total = 0
    successes = 0
    abs_errors = []
    condition_accuracies = {}
    for value in values:
        split_successes = 0
        split_abs_errors = []
        for sample in samples_dict[value]:
            estimated_count = detect_count_with_watershed(sample, min_area=COUNT_MIN_AREA)
            abs_error = abs(float(estimated_count) - float(value))
            success = float(estimated_count) == float(value)
            split_successes += int(success)
            successes += int(success)
            total += 1
            split_abs_errors.append(float(abs_error))
            abs_errors.append(float(abs_error))
        condition_accuracies[Serialize_key := serialize_query(value)] = {
            "accuracy": 100.0 * split_successes / len(samples_dict[value]),
            "mean_abs_count_error": float(np.mean(split_abs_errors)),
            "num_samples": len(samples_dict[value]),
        }
    return {
        "accuracy": 100.0 * successes / total if total else 0.0,
        "successes": successes,
        "total_samples": total,
        "num_conditions": len(values),
        "mean_abs_count_error": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "threshold": "exact_match",
        "condition_accuracies": condition_accuracies,
    }


def evaluate_rotation_split(
    values: list[float],
    samples_dict: dict[float, list[np.ndarray]],
    config: dict,
) -> dict:
    total = 0
    successes = 0
    abs_errors = []
    detection_successes = 0
    condition_accuracies = {}
    for value in values:
        split_successes = 0
        split_abs_errors = []
        split_detection = 0
        for sample in samples_dict[value]:
            metrics = calculate_rotation_metrics_robust(
                sample,
                expected_angle=value,
            )
            detected = bool(metrics["detection_success"])
            success = detected and metrics["angle_error_abs"] <= ROTATION_ERROR_THRESHOLD
            split_successes += int(success)
            successes += int(success)
            split_detection += int(detected)
            detection_successes += int(detected)
            total += 1
            if detected:
                split_abs_errors.append(float(metrics["angle_error_abs"]))
                abs_errors.append(float(metrics["angle_error_abs"]))
        condition_accuracies[Serialize_key := serialize_query(value)] = {
            "accuracy": 100.0 * split_successes / len(samples_dict[value]),
            "detection_rate": 100.0 * split_detection / len(samples_dict[value]),
            "mean_abs_angle_error": float(np.mean(split_abs_errors)) if split_abs_errors else float("inf"),
            "num_samples": len(samples_dict[value]),
        }
    return {
        "accuracy": 100.0 * successes / total if total else 0.0,
        "successes": successes,
        "total_samples": total,
        "num_conditions": len(values),
        "detection_rate": 100.0 * detection_successes / total if total else 0.0,
        "mean_abs_angle_error": float(np.mean(abs_errors)) if abs_errors else float("inf"),
        "threshold": ROTATION_ERROR_THRESHOLD,
        "condition_accuracies": condition_accuracies,
    }


def evaluate_position_split(
    values: list[tuple[float, float]],
    samples_dict: dict[tuple[float, float], list[np.ndarray]],
    config: dict,
    training_positions: list[tuple[float, float]],
) -> dict:
    total = 0
    successes = 0
    errors = []
    detection_successes = 0
    condition_accuracies = {}
    for value in values:
        split_successes = 0
        split_detection = 0
        split_errors = []
        for sample in samples_dict[value]:
            metrics = calculate_position_metrics_enhanced(
                sample,
                expected_pos=value,
                training_positions=training_positions,
                image_size=config["image_size"],
                debug=False,
            )
            detected = bool(metrics["detection_success"])
            success = detected and metrics["position_error"] <= POSITION_ERROR_THRESHOLD
            split_successes += int(success)
            successes += int(success)
            split_detection += int(detected)
            detection_successes += int(detected)
            total += 1
            if detected:
                split_errors.append(float(metrics["position_error"]))
                errors.append(float(metrics["position_error"]))
        condition_accuracies[Serialize_key := serialize_query(value)] = {
            "accuracy": 100.0 * split_successes / len(samples_dict[value]),
            "detection_rate": 100.0 * split_detection / len(samples_dict[value]),
            "mean_position_error": float(np.mean(split_errors)) if split_errors else float("inf"),
            "num_samples": len(samples_dict[value]),
        }
    return {
        "accuracy": 100.0 * successes / total if total else 0.0,
        "successes": successes,
        "total_samples": total,
        "num_conditions": len(values),
        "detection_rate": 100.0 * detection_successes / total if total else 0.0,
        "mean_position_error": float(np.mean(errors)) if errors else float("inf"),
        "threshold": POSITION_ERROR_THRESHOLD,
        "condition_accuracies": condition_accuracies,
    }


def evaluate_size(args, selection: RunSelection, output_dir: Path) -> dict:
    metadata = load_dataset_metadata("size")
    queries = maybe_truncate_queries(
        build_size_queries(
            metadata,
            extrap_min=args.size_extrap_min,
            extrap_max=args.size_extrap_max,
        ),
        args.max_conditions_per_split,
    )
    config = build_size_config(selection)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_radius_model(config, device)
    sampler = create_sampler(config, args.num_sampling_steps)

    split_results = {}
    for split_name, values in queries.items():
        samples = generate_radius_samples(
            model,
            sampler,
            values,
            config,
            device,
            num_samples=args.num_samples_per_condition,
            cfg_scale=1.0,
            batch_size=args.eval_batch_size,
            num_sampling_steps=args.num_sampling_steps,
        )
        split_results[split_name] = evaluate_size_split(values, samples, config)
    return {"metadata": metadata, "splits": split_results}


def evaluate_count(args, selection: RunSelection, output_dir: Path) -> dict:
    metadata = load_dataset_metadata("count")
    queries = maybe_truncate_queries(build_count_queries(metadata), args.max_conditions_per_split)
    config = build_count_config(selection)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_radius_model(config, device)
    sampler = create_sampler(config, args.num_sampling_steps)

    split_results = {}
    for split_name, values in queries.items():
        samples = generate_radius_samples(
            model,
            sampler,
            values,
            config,
            device,
            num_samples=args.num_samples_per_condition,
            cfg_scale=1.0,
            batch_size=args.eval_batch_size,
            num_sampling_steps=args.num_sampling_steps,
        )
        split_results[split_name] = evaluate_count_split(values, samples)
    return {"metadata": metadata, "splits": split_results}


def evaluate_rotation(args, selection: RunSelection, output_dir: Path) -> dict:
    metadata = load_dataset_metadata("rotation")
    queries = maybe_truncate_queries(build_rotation_queries(metadata), args.max_conditions_per_split)
    config = build_rotation_config(selection, metadata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_rotation_model(config, device)
    sampler = create_sampler(config, args.num_sampling_steps)

    split_results = {}
    for split_name, values in queries.items():
        samples = generate_rotation_samples(
            model,
            sampler,
            values,
            config,
            device,
            num_samples=args.num_samples_per_condition,
            cfg_scale=1.0,
            batch_size=args.eval_batch_size,
            num_sampling_steps=args.num_sampling_steps,
        )
        split_results[split_name] = evaluate_rotation_split(values, samples, config)
    return {"metadata": metadata, "splits": split_results}


def evaluate_position(args, selection: RunSelection, output_dir: Path) -> dict:
    metadata = load_dataset_metadata("position")
    queries = maybe_truncate_queries(
        build_position_queries(
            metadata,
            extrap_margin=args.position_extrap_margin,
            extrap_min_margin=args.position_extrap_min_margin,
        ),
        args.max_conditions_per_split,
    )
    config = build_position_config(selection, metadata)
    training_positions = queries["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_position_model(config, device)
    sampler = create_sampler(config, args.num_sampling_steps)

    split_results = {}
    for split_name, values in queries.items():
        samples = generate_position_samples(
            model,
            sampler,
            values,
            config,
            device,
            num_samples=args.num_samples_per_condition,
            cfg_scale=1.0,
            batch_size=args.eval_batch_size,
            num_sampling_steps=args.num_sampling_steps,
        )
        split_results[split_name] = evaluate_position_split(values, samples, config, training_positions)
    return {"metadata": metadata, "splits": split_results}


EVALUATORS: dict[str, Callable[[argparse.Namespace, RunSelection, Path], dict]] = {
    "size": evaluate_size,
    "count": evaluate_count,
    "rotation": evaluate_rotation,
    "position": evaluate_position,
}


def default_output_root(num_samples_per_condition: int, num_sampling_steps: int) -> Path:
    return EVAL_ROOT / f"samples{num_samples_per_condition}_steps{num_sampling_steps}_cfg1"


def main():
    parser = argparse.ArgumentParser(description="Paper-locked Table 2 evaluator.")
    parser.add_argument("--skill", required=True, choices=sorted(EVALUATORS.keys()))
    parser.add_argument("--variant", required=True, help="Canonical variant name (baseline, sinusoidal, rotary, adaln, vae, flow, unet).")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-samples-per-condition", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-conditions-per-split", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--size-extrap-min",
        type=int,
        default=SIZE_EXTRAP_MIN,
        help="Minimum radius (inclusive) for size 'extra' split.",
    )
    parser.add_argument(
        "--size-extrap-max",
        type=int,
        default=SIZE_EXTRAP_MAX,
        help="Maximum radius (inclusive) for size 'extra' split.",
    )
    parser.add_argument(
        "--position-extrap-margin",
        type=float,
        default=POSITION_EXTRAP_MARGIN,
        help="Outer margin (px) for position 'extra' split (max distance outside training rectangle).",
    )
    parser.add_argument(
        "--position-extrap-min-margin",
        type=float,
        default=POSITION_EXTRAP_MIN_MARGIN,
        help="Inner margin (px) for position 'extra' split (min L-inf distance from training rectangle).",
    )
    args = parser.parse_args()

    output_root = args.output_root or default_output_root(args.num_samples_per_condition, args.num_sampling_steps)
    output_dir = output_root / args.skill / args.variant / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = latest_finished_run(args.skill, args.variant, args.seed)
    table_rows = [row for row, mapped_variant in TABLE_ROWS.items() if mapped_variant == args.variant]

    result = {
        "skill": args.skill,
        "variant": args.variant,
        "table_rows": table_rows,
        "seed": args.seed,
        "num_samples_per_condition": args.num_samples_per_condition,
        "num_sampling_steps": args.num_sampling_steps,
        "eval_batch_size": args.eval_batch_size,
        "size_extrap_min": args.size_extrap_min,
        "size_extrap_max": args.size_extrap_max,
        "position_extrap_margin": args.position_extrap_margin,
        "position_extrap_min_margin": args.position_extrap_min_margin,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(selection.run_dir),
        "checkpoint": str(selection.checkpoint),
        "git_commit": selection.run_config.get("git_commit"),
        "run_config": selection.run_config,
        "paper_metrics": EVALUATORS[args.skill](args, selection, output_dir),
    }

    output_path = output_dir / "results.json"
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)

    summary = result["paper_metrics"]["splits"]
    print(f"Saved {output_path}")
    print(
        f"{args.skill}/{args.variant}/seed_{args.seed}: "
        f"train={summary['train']['accuracy']:.2f} "
        f"interp={summary['interp']['accuracy']:.2f} "
        f"extra={summary['extra']['accuracy']:.2f}"
    )


if __name__ == "__main__":
    main()
