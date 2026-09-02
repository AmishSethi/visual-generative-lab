#!/usr/bin/env python
"""Rebuild the whole count column of Table 2 on a new count geometry.

If the count dataset changes, the baseline cell is not the only one that moves:
every ablation row in that column was measured on the old geometry, so leaving
them would make the column internally inconsistent.  This fires all eight
configurations at the requested seeds against one variant dataset.

Config specs come from paper_runs.table2.manifest.VARIANTS, the same source the
published runs used, so only the dataset differs from the paper's own pipeline.

    python -m paper_runs.rebuttal.submit_count_column --variant grid_r5 --seeds 0 1 2

Run the sweep in submit_count_variants.py first and pick the winner: there is no
point rebuilding twenty-four runs on a geometry that does not reach the target.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.count_variants import VARIANTS as GEOMETRIES
from paper_runs.rebuttal.manifest import (
    VGL_SLURM_EXCLUDE,
    DATASET_ROOT, LOG_ROOT, REPO_ROOT, RESULTS_ROOT, SBATCH_TEMPLATE, SKILLS, SLURM_ROOT,
)
from paper_runs.table2.manifest import VARIANTS
from paper_runs.table2.submit_training import MICRO_BATCH_OVERRIDES

# The count column of Table 2, in the order the table prints them.
COLUMN = ["baseline", "sinusoidal", "rotary", "adaln", "vae", "flow", "unet", "dit_large"]
NUM_GPUS = 4


def build(geometry, variant, seed, epochs):
    skill_spec = SKILLS["count"]
    vs = VARIANTS[variant]
    results = (RESULTS_ROOT / "count_column" / f"{geometry}_e{epochs}"
               / variant / f"seed_{seed}")
    cmd = [
        skill_spec["train_script"],
        "--data-path", str(DATASET_ROOT / "count_variants" / geometry),
        "--results-dir", str(results),
        "--epochs", str(epochs),
        "--global-batch-size", "128",
        "--num-workers", "4",
        "--global-seed", str(seed),
        "--image-size", "64",
        "--conditioning-method", vs["conditioning"],
        "--architecture", vs["architecture"],
        "--model", vs["model"],
        "--ckpt-every", "5000",
        f"--{skill_spec['embedding_flag']}", vs["embedding"],
        f"--{skill_spec['dropout_flag']}", "0.0",
        *skill_spec["null_args"],
    ]
    micro = MICRO_BATCH_OVERRIDES.get(("count", variant))
    if micro is not None:
        cmd.extend(["--micro-batch-size", str(micro)])
    if vs["latent"]:
        cmd.append("--use-latent-diffusion")
    if vs["flow_matching"]:
        cmd.append("--use-flow-matching")
    hours = 8 if epochs <= 1000 else 8 * (epochs / 1000.0)
    if vs["model"] != "DiT-S/2" or vs["architecture"] != "dit":
        hours *= 3
    return cmd, f"{min(int(hours) + 4, 160):d}:00:00"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=list(GEOMETRIES),
                    help="which count geometry to rebuild the column on")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rows", nargs="+", default=COLUMN, choices=COLUMN)
    ap.add_argument("--epochs", type=int, default=3000,
                    help="count is still climbing +18 pts over the last 13k steps at "
                         "1000 epochs, so the default schedule undertrains it")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = DATASET_ROOT / "count_variants" / a.variant
    if not (data / "dataset_metadata.json").exists():
        ap.error(f"{data} not generated yet; run count_variants.py first")

    SLURM_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    n = 0
    for variant in a.rows:
        for seed in a.seeds:
            cmd, walltime = build(a.variant, variant, seed, a.epochs)
            job_name = f"rb-cc-{a.variant}-{variant}-s{seed}-e{a.epochs}"
            script = SBATCH_TEMPLATE.format(
                job_name=job_name, num_gpus=NUM_GPUS, mem="64G", time_limit=walltime,
                stdout_path=LOG_ROOT / f"{job_name}.out",
                stderr_path=LOG_ROOT / f"{job_name}.err",
                repo_root=REPO_ROOT,
                command=" ".join(shlex.quote(c) for c in cmd),
            )
            # neu306 faults in conv2d with illegal memory access.
            script = script.replace("#SBATCH --gres=gpu:",
                                    VGL_SLURM_EXCLUDE + "\n#SBATCH --gres=gpu:")
            path = SLURM_ROOT / f"{job_name}.sbatch"
            path.write_text(script)
            n += 1
            if a.dry_run:
                print(f"[dry-run] {job_name}")
                continue
            out = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
            print(f"{job_name:38s} {out.stdout.strip() or out.stderr.strip()}")
    print(f"\n{n} jobs for geometry '{a.variant}'")


if __name__ == "__main__":
    main()
