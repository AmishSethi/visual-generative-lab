#!/usr/bin/env python
"""Score the count-fit diagnostic runs with the paper-locked evaluator.

Fills the capacity ladder between the DiT-S/2 baseline (51.8 train) and the
published DiT-L row (80.3 under the same watershed counter), and records the
per-count breakdown so a gain can be attributed to particular counts rather
than read as a single number.
"""
import argparse
import json
import sys
from pathlib import Path

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TABLE2 = Path(f"{VGL_ROOT}/MORE_SEEDS/table2")
REB = Path(f"{VGL_ROOT}/REBUTTAL")


class EvalArgs:
    """Defaults must match evaluate_table2's module constants, or these runs get
    a different query grid than the published rows they are compared against."""

    def __init__(self, num_samples, steps, batch):
        self.num_samples_per_condition = num_samples
        self.num_sampling_steps = steps
        self.eval_batch_size = batch
        self.max_conditions_per_split = None
        self.size_extrap_min = 1
        self.size_extrap_max = 25
        self.position_extrap_margin = 6.0
        self.position_extrap_min_margin = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["cap", "long"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--eval-batch-size", type=int, default=20)
    ap.add_argument("--out", type=Path, default=REB / "eval" / "count_fit_results.json")
    a = ap.parse_args()

    import paper_runs.table2.evaluate_table2 as t2
    from paper_runs.rebuttal import subsample
    subsample.install(t2)

    t2.DATASET_ROOT = TABLE2 / "datasets"
    t2.RESULTS_ROOT = REB / "results" / "count_fit"

    payload = json.loads(a.out.read_text()) if a.out.exists() else {}
    eval_args = EvalArgs(a.num_samples, a.steps, a.eval_batch_size)

    for cell in a.cells:
        try:
            selection = t2.latest_finished_run(cell, "", a.seed)
        except FileNotFoundError as exc:
            print(f"[skip] {cell}: {exc}", flush=True)
            continue

        print(f"\n=== count_fit {cell} ===", flush=True)
        print(f"checkpoint: {selection.checkpoint}", flush=True)
        target = REB / "eval" / "count_fit" / cell
        target.mkdir(parents=True, exist_ok=True)
        metrics = t2.EVALUATORS["count"](eval_args, selection, target)

        splits = {k: v["accuracy"] for k, v in metrics["splits"].items()}
        payload[cell] = {
            "checkpoint": str(selection.checkpoint),
            "splits": splits,
            "per_condition": {
                k: v.get("condition_accuracies") for k, v in metrics["splits"].items()
            },
        }
        print(f"   {splits}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=1))

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
