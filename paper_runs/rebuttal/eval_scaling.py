#!/usr/bin/env python
"""Evaluate the data-scaling sweep with the paper's own metrics (HeVs w3).

The rebuttal results tree is laid out as
    RESULTS_ROOT/scaling/{skill}/{10k,20k,40k,80k}/seed_N/<run>/checkpoints/
which is exactly the shape `evaluate_table2.latest_finished_run` expects once
its RESULTS_ROOT is repointed, with the dataset-size tag playing the role of
"variant".  Repointing rather than reimplementing means the scaling curve is
measured by the same code that produced the paper's Table 1 numbers -- the
train/interp/extra query grids still come from the canonical table2 dataset
metadata, so every point on the curve is evaluated on identical conditions and
only the training-set size differs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paper_runs.table2.evaluate_table2 as t2
from paper_runs.rebuttal import subsample
from paper_runs.rebuttal.manifest import EVAL_ROOT, RESULTS_ROOT, SCALING_SIZES


class EvalArgs:
    """Mimics the argparse namespace evaluate_table2's evaluators expect."""

    def __init__(self, num_samples, steps, batch_size, max_conditions):
        self.num_samples_per_condition = num_samples
        self.num_sampling_steps = steps
        self.eval_batch_size = batch_size
        self.max_conditions_per_split = max_conditions
        self.size_extrap_min = t2.SIZE_EXTRAP_MIN
        self.size_extrap_max = t2.SIZE_EXTRAP_MAX
        self.position_extrap_margin = t2.POSITION_EXTRAP_MARGIN
        self.position_extrap_min_margin = t2.POSITION_EXTRAP_MIN_MARGIN


def main():
    parser = argparse.ArgumentParser(description="Evaluate the data-scaling sweep.")
    parser.add_argument("--skills", nargs="+", default=["size", "position", "count"])
    parser.add_argument("--sizes", nargs="+", default=[f"{n // 1000}k" for n in SCALING_SIZES])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--max-conditions-per-split", type=int, default=None,
                        help="Position has 1024 conditions per split; cap it for a "
                             "sub-sampled but paper-comparable number.")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "scaling_results.json")
    args = parser.parse_args()

    # Position queries are generated as nested x,y loops, so the evaluator's
    # prefix truncation would sample only the left-hand columns of the canvas.
    subsample.install(t2)

    # Repoint checkpoint discovery at the rebuttal tree; dataset metadata (and
    # therefore the evaluation grid) stays on the canonical table2 datasets.
    t2.RESULTS_ROOT = RESULTS_ROOT / "scaling"

    eval_args = EvalArgs(args.num_samples, args.steps, args.eval_batch_size,
                         args.max_conditions_per_split)
    out_dir = EVAL_ROOT / "scaling"
    payload = {}

    for skill in args.skills:
        payload[skill] = {}
        for tag in args.sizes:
            try:
                selection = t2.latest_finished_run(skill, tag, args.seed)
            except FileNotFoundError as exc:
                print(f"[skip] {skill}/{tag}: {exc}")
                continue
            print(f"\n=== {skill} @ {tag} ===", flush=True)
            target = out_dir / skill / tag / f"seed_{args.seed}"
            target.mkdir(parents=True, exist_ok=True)
            metrics = t2.EVALUATORS[skill](eval_args, selection, target)
            payload[skill][tag] = {
                "checkpoint": str(selection.checkpoint),
                "splits": {k: {"accuracy": v["accuracy"], "num_conditions": v["num_conditions"]}
                           for k, v in metrics["splits"].items()},
            }
            (target / "results.json").write_text(
                json.dumps({"paper_metrics": metrics, "checkpoint": str(selection.checkpoint)},
                           indent=2, default=str))
            for split, res in metrics["splits"].items():
                print(f"  {split:>7}: {res['accuracy']:.1f}%")

    print(f"\n{'skill':>9} " + "".join(f"{t:>22}" for t in args.sizes))
    print(f"{'':>9} " + "".join(f"{'train/interp/extra':>22}" for _ in args.sizes))
    for skill, by_size in payload.items():
        row = f"{skill:>9} "
        for tag in args.sizes:
            if tag in by_size:
                s = by_size[tag]["splits"]
                row += f"{s['train']['accuracy']:6.1f}/{s['interp']['accuracy']:5.1f}/{s['extra']['accuracy']:5.1f}   ".rjust(22)
            else:
                row += f"{'--':>22}"
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
