#!/usr/bin/env python
"""Is count training saturated at 1000 epochs, or would more steps help?

Training loss is not a usable proxy here.  On the paper's geometry the loss was
flat at ~0.017 from epoch 1000 onward, yet tripling the schedule moved count
accuracy from 51.8 to 76.7.  The diffusion objective is dominated by high-noise
timesteps and says little about whether the sample has the right number of
objects.  So measure accuracy directly against step count.

Scores intermediate checkpoints of one run with the paper-locked evaluator, on
the train split only (the question is about fitting, and it keeps the sweep cheap).
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
    ap.add_argument("--variant", default="grid_r4",
                    help="results directory under results/count_variants/")
    ap.add_argument("--dataset", default=None,
                    help="dataset directory, if it differs from --variant "
                         "(e.g. a longer-schedule run trained on the same data)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-list", nargs="+", type=int,
                    default=[20000, 35000, 50000, 65000, 78000])
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--sampling-steps", type=int, default=250)
    ap.add_argument("--out", type=Path, default=REB / "eval" / "count_saturation.json")
    a = ap.parse_args()

    import paper_runs.table2.evaluate_table2 as t2
    from paper_runs.rebuttal import subsample
    subsample.install(t2)
    t2.RESULTS_ROOT = REB / "results" / "count_variants"

    data = REB / "datasets" / "count_variants" / (a.dataset or a.variant)
    meta = json.loads((data / "dataset_metadata.json").read_text())
    t2.load_dataset_metadata = lambda skill, _m=meta: _m

    try:
        base = t2.latest_finished_run(a.variant, "", a.seed)
        ckpt_dir = base.checkpoint.parent
    except FileNotFoundError:
        # The run is still training, so there is no final_*.pt yet.  Score the
        # intermediate checkpoints anyway: the point is the shape of the curve,
        # and waiting for the run to end defeats that.
        seed_root = t2.RESULTS_ROOT / a.variant / "" / f"seed_{a.seed}"
        runs = [d for d in sorted(seed_root.iterdir())
                if d.is_dir() and (d / "checkpoints").is_dir()]
        if not runs:
            raise
        ckpt_dir = runs[-1] / "checkpoints"
        # Pass the whole document: build_common_run_metadata does its own ["args"].
        cfg = json.loads((runs[-1] / "run_config.json").read_text())
        base = t2.RunSelection(run_dir=runs[-1],
                               checkpoint=sorted(ckpt_dir.glob("*.pt"))[-1],
                               run_config=cfg)
        print(f"[in-progress] scoring {ckpt_dir}", flush=True)
    eval_args = EvalArgs(a.num_samples, a.sampling_steps, a.num_samples)

    payload = json.loads(a.out.read_text()) if a.out.exists() else {}
    payload.setdefault(a.variant, {})

    for step in a.steps_list:
        # final_ is only written for the last step; intermediates are bare.
        cand = [ckpt_dir / f"{step:07d}.pt", ckpt_dir / f"final_{step:07d}.pt"]
        ckpt = next((c for c in cand if c.exists()), None)
        if ckpt is None:
            print(f"[skip] step {step}: no checkpoint", flush=True)
            continue

        sel = t2.RunSelection(run_dir=base.run_dir, checkpoint=ckpt,
                              run_config=base.run_config)
        target = REB / "eval" / "count_saturation" / a.variant / str(step)
        target.mkdir(parents=True, exist_ok=True)
        metrics = t2.EVALUATORS["count"](eval_args, sel, target)
        train = metrics["splits"]["train"]["accuracy"]
        payload[a.variant][str(step)] = {
            "train": train,
            "extra": metrics["splits"]["extra"]["accuracy"],
            "per_count": {k: v["accuracy"] for k, v
                          in metrics["splits"]["train"]["condition_accuracies"].items()},
        }
        print(f"  step {step:>6}: train {train:5.1f}%", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=1))

    rows = payload[a.variant]
    if len(rows) > 1:
        ks = sorted(rows, key=int)
        print("\nstep      train")
        for k in ks:
            print(f"{int(k):>7}   {rows[k]['train']:5.1f}")
        d = rows[ks[-1]]["train"] - rows[ks[-2]]["train"]
        print(f"\nlast interval moved {d:+.1f} pts -> "
              f"{'still climbing' if d > 2 else 'flat, saturated'}")


if __name__ == "__main__":
    main()
