#!/usr/bin/env python
"""Train the paper's baseline recipe on each count geometry variant.

Everything except the dataset is held at the Table 1 baseline: DiT-S/2, linear
count embedding, concat conditioning, pixel space, 1000 epochs, global batch 128.
So any change in training accuracy is attributable to the geometry alone.

Reference: the paper's geometry (random placement, radius 8, counts 2-7) fits to
51.8% exact match over nine seeds.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.count_variants import VARIANTS
from paper_runs.rebuttal.manifest import (
    VGL_SLURM_EXCLUDE,
    DATASET_ROOT, LOG_ROOT, REPO_ROOT, RESULTS_ROOT, SBATCH_TEMPLATE, SKILLS, SLURM_ROOT,
)

NUM_GPUS = 4
EPOCHS = 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    SLURM_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    spec = SKILLS["count"]

    for name in a.variants:
        data = DATASET_ROOT / "count_variants" / name
        results = RESULTS_ROOT / "count_variants" / name / f"seed_{a.seed}"
        cmd = [
            spec["train_script"],
            "--data-path", str(data),
            "--results-dir", str(results),
            "--epochs", str(EPOCHS),
            "--global-batch-size", "128",
            "--num-workers", "4",
            "--global-seed", str(a.seed),
            "--image-size", "64",
            "--conditioning-method", "concat",
            "--architecture", "dit",
            "--model", "DiT-S/2",
            "--ckpt-every", "5000",
            f"--{spec['embedding_flag']}", "linear",
            f"--{spec['dropout_flag']}", "0.0",
            *spec["null_args"],
        ]
        job_name = f"rb-cntvar-{name}-s{a.seed}"
        script = SBATCH_TEMPLATE.format(
            job_name=job_name, num_gpus=NUM_GPUS, mem="64G", time_limit="24:00:00",
            stdout_path=LOG_ROOT / f"{job_name}.out",
            stderr_path=LOG_ROOT / f"{job_name}.err",
            repo_root=REPO_ROOT,
            command=" ".join(shlex.quote(c) for c in cmd),
        )
        # neu306 throws illegal-memory-access faults in conv2d; it killed two
        # unrelated jobs before it was identified.
        script = script.replace("#SBATCH --gres=gpu:", VGL_SLURM_EXCLUDE + "\n#SBATCH --gres=gpu:")
        path = SLURM_ROOT / f"{job_name}.sbatch"
        path.write_text(script)
        if a.dry_run:
            print(f"[dry-run] {path}")
            continue
        out = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
        print(f"{job_name:26s} {out.stdout.strip() or out.stderr.strip()}")


if __name__ == "__main__":
    main()
