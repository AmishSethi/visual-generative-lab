#!/usr/bin/env python
"""Evaluate appendix families that reuse the paper's metrics unchanged.

  ranges     narrower / shifted training supports; tests whether the
             extrapolation boundary follows the support.  Each run has its own
             dataset, so the train/interp grids come from that run's metadata
             while the extrapolation grid is widened to a common span --
             otherwise each model would be probed over a different range and
             the break points would not be comparable.

  text_cond  frozen text-encoder embeddings of the skill value in place of
             numeric conditioning, trained on the paper's canonical datasets,
             so only the conditioning differs.

Both use evaluate_table2's own evaluators, so numbers land on the same scale as
the paper's Table 1.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paper_runs.table2.evaluate_table2 as t2
from paper_runs.appendix import subsample
from paper_runs.appendix.manifest import (
    DATASET_ROOT, EVAL_ROOT, RANGE_SPECS, RESULTS_ROOT, TABLE2_DATASETS,
)

# Widest radius that still fits a 64x64 canvas.
SIZE_EXTRAP_MIN = 1
SIZE_EXTRAP_MAX = 32


class EvalArgs:
    def __init__(self, num_samples, steps, batch_size, max_conditions):
        self.num_samples_per_condition = num_samples
        self.num_sampling_steps = steps
        self.eval_batch_size = batch_size
        self.max_conditions_per_split = max_conditions
        self.size_extrap_min = SIZE_EXTRAP_MIN
        self.size_extrap_max = SIZE_EXTRAP_MAX
        self.position_extrap_margin = t2.POSITION_EXTRAP_MARGIN
        self.position_extrap_min_margin = t2.POSITION_EXTRAP_MIN_MARGIN


def run_one(skill, path_key, dataset_dir, results_root, seed, eval_args, out_dir):
    """Evaluate one run, reading metadata from an explicit dataset directory.

    `path_key` locates the run on disk (results_root/<path_key>/seed_N), while
    `skill` selects the evaluator.  For the ranges family these differ -- the
    directory is `size_shifted` but the evaluator is `size` -- so they cannot be
    the same argument.
    """
    original_loader = t2.load_dataset_metadata
    t2.load_dataset_metadata = lambda _skill: json.loads(
        (Path(dataset_dir) / "dataset_metadata.json").read_text())
    t2.RESULTS_ROOT = results_root
    try:
        selection = t2.latest_finished_run(path_key, "", seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = t2.EVALUATORS[skill](eval_args, selection, out_dir)
    finally:
        t2.load_dataset_metadata = original_loader
    (out_dir / "results.json").write_text(
        json.dumps({"paper_metrics": metrics, "checkpoint": str(selection.checkpoint)},
                   indent=2, default=str))
    return selection, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranges / text_cond families.")
    parser.add_argument("--family", required=True, choices=["ranges", "text_cond"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--max-conditions-per-split", type=int, default=100)
    parser.add_argument("--only", nargs="+", default=None,
                        help="Evaluate only these family members, so one long serial "
                             "job can be split into short parallel ones that backfill.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # Position queries are generated as nested x,y loops, so the evaluator's
    # prefix truncation would sample only the left-hand columns of the canvas.
    subsample.install(t2)

    out_path = args.out or EVAL_ROOT / f"{args.family}_results.json"
    eval_args = EvalArgs(args.num_samples, args.steps, args.eval_batch_size,
                         args.max_conditions_per_split)

    if args.family == "ranges":
        jobs = [(name, spec["skill"], DATASET_ROOT / "ranges" / name,
                 RESULTS_ROOT / "ranges", name) for name, spec in RANGE_SPECS.items()]
    else:
        jobs = [(skill, skill, TABLE2_DATASETS / skill,
                 RESULTS_ROOT / "text_cond", skill)
                for skill in ("size", "count", "rotation")]

    if args.only:
        jobs = [j for j in jobs if j[0] in set(args.only)]
        if not jobs:
            parser.error(f"--only {args.only} matched no member of family {args.family}")

    payload = {}
    for label, skill, dataset_dir, results_root, variant in jobs:
        out_dir = EVAL_ROOT / args.family / label / f"seed_{args.seed}"
        try:
            print(f"\n=== {args.family} {label} ({skill}) ===", flush=True)
            selection, metrics = run_one(skill, variant, dataset_dir, results_root,
                                         args.seed, eval_args, out_dir)
        except FileNotFoundError as exc:
            print(f"[skip] {label}: {exc}")
            continue
        payload[label] = {
            "skill": skill,
            "checkpoint": str(selection.checkpoint),
            "splits": {k: v["accuracy"] for k, v in metrics["splits"].items()},
            "per_condition": {k: v.get("condition_accuracies", {})
                              for k, v in metrics["splits"].items()},
        }
        for split, res in metrics["splits"].items():
            print(f"  {split:>7}: {res['accuracy']:.1f}%")

    if payload:
        print(f"\n{'run':>18} {'skill':>9} {'train':>8} {'interp':>8} {'extra':>8}")
        for label, res in payload.items():
            s = res["splits"]
            print(f"{label:>18} {res['skill']:>9} {s.get('train', 0):>7.1f}% "
                  f"{s.get('interp', 0):>7.1f}% {s.get('extra', 0):>7.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
