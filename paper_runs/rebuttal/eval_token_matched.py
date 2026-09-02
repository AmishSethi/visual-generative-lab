#!/usr/bin/env python
"""Evaluate the token-matched runs with the paper-locked evaluator.

Fills in the sequence-length ladder at a fixed 64x64 resolution:

    space    model      tokens
    latent   DiT-S/2      16     published
    pixel    DiT-S/16     16     this script
    latent   DiT-S/1      64     this script
    pixel    DiT-S/2    1024     published

If pixel at 16 tokens lands near the latent baseline, sequence length explains the
pixel-versus-latent gap.  If it stays near the pixel baseline, the representation matters and
our current wording is wrong.
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

REB = Path(f"{VGL_ROOT}/REBUTTAL")
T2 = Path(f"{VGL_ROOT}/MORE_SEEDS/table2")

# tag -> (skill, is_latent)
RUNS = {
    "tok16-pixel-size": ("size", False),
    "tok64-latent-size": ("size", True),
    "tok16-pixel-position": ("position", False),
    "tok64-latent-position": ("position", True),
    "tok16-pixel-rotation": ("rotation", False),
    "tok64-latent-rotation": ("rotation", True),
}


class EvalArgs:
    """Mirrors the evaluator's argparse namespace. Defaults must match the module
    constants in evaluate_table2.py, or these runs get a different query grid than
    the published rows they are compared against."""

    def __init__(self, num_samples, steps, batch, max_cond):
        self.num_samples_per_condition = num_samples
        self.num_sampling_steps = steps
        self.eval_batch_size = batch
        self.max_conditions_per_split = max_cond
        self.size_extrap_min = 1          # SIZE_EXTRAP_MIN
        self.size_extrap_max = 25         # SIZE_EXTRAP_MAX
        self.position_extrap_margin = 6.0        # POSITION_EXTRAP_MARGIN
        self.position_extrap_min_margin = 0.0    # POSITION_EXTRAP_MIN_MARGIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=list(RUNS))
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--eval-batch-size", type=int, default=20)
    ap.add_argument("--max-conditions-per-split", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REB / "eval" / "token_matched_results.json")
    a = ap.parse_args()

    import paper_runs.table2.evaluate_table2 as t2
    from paper_runs.rebuttal import subsample
    subsample.install(t2)

    # Datasets live in the paper tree; checkpoints live under the rebuttal tree.
    t2.DATASET_ROOT = T2 / "datasets"
    t2.RESULTS_ROOT = REB / "results" / "token_matched"

    payload = {}
    if a.out.exists():
        payload = json.loads(a.out.read_text())

    eval_args = EvalArgs(a.num_samples, a.steps, a.eval_batch_size, a.max_conditions_per_split)

    for tag in a.tags:
        skill, is_latent = RUNS[tag]
        try:
            # "" slots in where the evaluator expects a variant level, which these runs lack.
            selection = t2.latest_finished_run(tag, "", 0)
        except FileNotFoundError as exc:
            print(f"[skip] {tag}: {exc}", flush=True)
            continue

        print(f"\n=== {tag} ({skill}, {'latent' if is_latent else 'pixel'}) ===", flush=True)
        target = REB / "eval" / "token_matched" / tag
        target.mkdir(parents=True, exist_ok=True)
        metrics = t2.EVALUATORS[skill](eval_args, selection, target)
        payload[tag] = {
            "skill": skill,
            "latent": is_latent,
            "checkpoint": str(selection.checkpoint),
            "splits": {k: v["accuracy"] for k, v in metrics["splits"].items()},
        }
        print("   ", payload[tag]["splits"], flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=1))

    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
