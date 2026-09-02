#!/usr/bin/env python
"""Can any configuration fit the count training distribution?

The reviewer's remaining objection is that count never fits its own training
support, so the count row cannot support a generalization claim.  Our 9-seed
exact-match measurement is 51.8 +- 4.1, and the per-value breakdown decays
monotonically with count (85, 75, 60, 45, 35, 35 for n = 2..7).

Two explanations are already ruled out:

  * The metric.  On non-overlapping ground truth the tuned Hough counter scores
    100% at every count from 0 to 9 (eval/tuned_counter_validation.json), so the
    ceiling is not the problem.
  * Resolution.  The 128x128 run is worse, not better, at 33.3% train
    (eval/hires128_results.json), so small objects are not the problem.

That leaves capacity, optimisation length, and how the integer reaches the model.
Each cell below changes exactly one of those against the paper baseline
(DiT-S/2, linear, 1000 epochs).
"""

import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.manifest import (


    LOG_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    SBATCH_TEMPLATE,
    SKILLS,
    SLURM_ROOT,
)

COUNT_DATA = Path(f"{VGL_ROOT}/MORE_SEEDS/table2/datasets/count")

# tag -> (model, embedding, epochs, walltime, what it tests)
CELLS = {
    "cap":  ("DiT-B/2", "linear",     1000, "24:00:00", "capacity"),
    "long": ("DiT-S/2", "linear",     3000, "36:00:00", "optimisation length"),
    "sinu": ("DiT-S/2", "sinusoidal", 1000, "16:00:00", "conditioning code"),
}

NUM_GPUS = 4


def build(tag, seed):
    model, embedding, epochs, walltime, _ = CELLS[tag]
    spec = SKILLS["count"]
    results = RESULTS_ROOT / "count_fit" / tag / f"seed_{seed}"
    cmd = [
        spec["train_script"],
        "--data-path", str(COUNT_DATA),
        "--results-dir", str(results),
        "--epochs", str(epochs),
        "--global-batch-size", "128",
        "--num-workers", "4",
        "--global-seed", str(seed),
        "--image-size", "64",
        "--conditioning-method", "concat",
        "--architecture", "dit",
        "--model", model,
        "--ckpt-every", "5000",
        f"--{spec['embedding_flag']}", embedding,
        f"--{spec['dropout_flag']}", "0.0",
        *spec["null_args"],
    ]
    return f"rb-cntfit-{tag}-s{seed}", cmd, walltime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    SLURM_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    for tag in a.cells:
        job_name, cmd, walltime = build(tag, a.seed)
        script = SBATCH_TEMPLATE.format(
            job_name=job_name,
            num_gpus=NUM_GPUS,
            mem="64G",
            time_limit=walltime,
            stdout_path=LOG_ROOT / f"{job_name}.out",
            stderr_path=LOG_ROOT / f"{job_name}.err",
            repo_root=REPO_ROOT,
            command=" ".join(shlex.quote(c) for c in cmd),
        )
        path = SLURM_ROOT / f"{job_name}.sbatch"
        path.write_text(script)
        if a.dry_run:
            print(f"[dry-run] {path}  ({CELLS[tag][4]})")
            continue
        out = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
        print(f"{job_name:24s} {CELLS[tag][4]:20s} {out.stdout.strip() or out.stderr.strip()}")


if __name__ == "__main__":
    main()
