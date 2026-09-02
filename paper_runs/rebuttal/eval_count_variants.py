#!/usr/bin/env python
"""Score the count geometry variants with the paper-locked evaluator.

Each variant has its own radius and spacing, so its query grid comes from its own
dataset_metadata.json rather than the paper's.  Everything else -- sampler, seed
count, exact-match criterion, watershed counter -- is the published pipeline.

Reference: the paper's geometry (radius 8, 2 px clearance) fits to 51.8% over
nine seeds.  A variant is only usable if its counter ceiling on ground truth
exceeds the target, otherwise the metric caps the result before the model does.
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


class EvalArgs:
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
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--eval-batch-size", type=int, default=20)
    ap.add_argument("--out", type=Path, default=REB / "eval" / "count_variants_results.json")
    a = ap.parse_args()

    import paper_runs.table2.evaluate_table2 as t2
    from paper_runs.rebuttal import subsample
    subsample.install(t2)

    t2.RESULTS_ROOT = REB / "results" / "count_variants"
    payload = json.loads(a.out.read_text()) if a.out.exists() else {}
    eval_args = EvalArgs(a.num_samples, a.steps, a.eval_batch_size)

    for name in a.variants:
        data = REB / "datasets" / "count_variants" / name
        meta = json.loads((data / "dataset_metadata.json").read_text())
        # The evaluator resolves metadata by skill name; point it at this variant
        # so the query grid matches the geometry the model actually trained on.
        t2.load_dataset_metadata = lambda skill, _m=meta: _m

        try:
            selection = t2.latest_finished_run(name, "", a.seed)
        except FileNotFoundError as exc:
            print(f"[skip] {name}: {exc}", flush=True)
            continue

        print(f"\n=== {name}  (radius {meta['circle_radius']}, "
              f"sep_pad {meta.get('sep_pad')}, {meta['placement']}) ===", flush=True)
        target = REB / "eval" / "count_variants" / name
        target.mkdir(parents=True, exist_ok=True)
        metrics = t2.EVALUATORS["count"](eval_args, selection, target)

        splits = {k: v["accuracy"] for k, v in metrics["splits"].items()}
        payload[name] = {
            "radius": meta["circle_radius"],
            "sep_pad": meta.get("sep_pad"),
            "placement": meta["placement"],
            "checkpoint": str(selection.checkpoint),
            "splits": splits,
            "per_condition": {k: v.get("condition_accuracies")
                              for k, v in metrics["splits"].items()},
        }
        print(f"    {splits}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=1))

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
