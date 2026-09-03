#!/usr/bin/env python
"""Regenerate and dump samples from a paper checkpoint.

The paper evaluator scores samples in memory and keeps only aggregate numbers,
but two appendix probes need the pixels themselves:

  * the three-counter cross-validation must re-score the *same* generations
    with independent counters and measure how often objects overlap;
  * the nearest-neighbour memorisation probe must compare generations against
    the training set.

Sampling reuses the paper's own loaders, sampler and query construction, so the
dumped images are drawn from exactly the distribution the paper reports on.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from paper_runs.appendix.manifest import EVAL_ROOT
from paper_runs.table2.evaluate_table2 import (
    build_count_config, build_count_queries, build_rotation_config,
    build_rotation_queries, build_size_config, build_size_queries,
    create_sampler, latest_finished_run, load_dataset_metadata,
    maybe_truncate_queries, serialize_query,
)
from vgl.eval_radius import generate_samples as generate_radius_samples, load_model as load_radius_model
from vgl.eval_rotation import generate_samples as generate_rotation_samples, load_model as load_rotation_model


def build(skill, selection):
    metadata = load_dataset_metadata(skill)
    if skill == "count":
        return metadata, build_count_queries(metadata), build_count_config(selection), \
            load_radius_model, generate_radius_samples
    if skill == "size":
        return metadata, build_size_queries(metadata), build_size_config(selection), \
            load_radius_model, generate_radius_samples
    if skill == "rotation":
        return metadata, build_rotation_queries(metadata), build_rotation_config(selection, metadata), \
            load_rotation_model, generate_rotation_samples
    raise ValueError(f"unsupported skill {skill}")


def main():
    parser = argparse.ArgumentParser(description="Dump generated samples from a paper checkpoint.")
    parser.add_argument("--skill", required=True, choices=["count", "size", "rotation"])
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--max-conditions-per-split", type=int, default=None)
    parser.add_argument("--out-root", type=Path, default=EVAL_ROOT / "samples")
    args = parser.parse_args()

    selection = latest_finished_run(args.skill, args.variant, args.seed)
    metadata, queries, config, load_model, generate_samples = build(args.skill, selection)
    queries = maybe_truncate_queries(queries, args.max_conditions_per_split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    sampler = create_sampler(config, args.num_sampling_steps)

    out_root = args.out_root / args.skill / args.variant / f"seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {"checkpoint": str(selection.checkpoint), "skill": args.skill,
                "variant": args.variant, "seed": args.seed,
                "num_samples_per_condition": args.num_samples, "splits": {}}

    for split_name, values in queries.items():
        samples = generate_samples(
            model, sampler, values, config, device,
            num_samples=args.num_samples, cfg_scale=1.0,
            batch_size=args.eval_batch_size, num_sampling_steps=args.num_sampling_steps,
        )
        split_dir = out_root / split_name
        for value in values:
            value_dir = split_dir / serialize_query(value)
            value_dir.mkdir(parents=True, exist_ok=True)
            for idx, sample in enumerate(samples[value]):
                Image.fromarray(np.asarray(sample, dtype=np.uint8)).save(value_dir / f"{idx:03d}.png")
        manifest["splits"][split_name] = [serialize_query(v) for v in values]
        print(f"{split_name}: {len(values)} conditions x {args.num_samples} samples", flush=True)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
