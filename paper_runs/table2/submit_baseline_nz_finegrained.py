#!/usr/bin/env python
"""Submit additional rotation `baseline_nz_finegrained` seeds (null_embedding_type=zero).

Identical to `submit_baseline_nz_extra_seeds.py` (DiT-S/2 + linear + concat,
`--null-embedding-type zero --null-angle 180.0`) BUT writes checkpoints every
500 training steps instead of 2000 so the bifurcation point of rotation
extrapolation can be studied at high temporal resolution.

Output dir: `RESULTS_ROOT/rotation/baseline_nz_finegrained/seed_{N}/` to avoid
clashing with the existing `baseline_nz/seed_{N}/` runs.
"""

import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")
# Optional node exclusions for your cluster, e.g. "#SBATCH --exclude=node1,node2"
VGL_SLURM_EXCLUDE = _os.environ.get("VGL_SLURM_EXCLUDE", "")

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.manifest import (


    DATASET_ROOT,
    LOG_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    SLURM_ROOT,
)


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
{VGL_SLURM_EXCLUDE}
#SBATCH --time=30:00:00
#SBATCH --output={stdout_path}
#SBATCH --error={stderr_path}

export MASTER_PORT=$((29500 + SLURM_JOB_ID % 35000))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_USE_CUDA_DSA=1

cd {repo_root}
{VGL_CONDA_ACTIVATE}

torchrun --nproc_per_node=1 --master_port $MASTER_PORT {command}
"""
SBATCH_TEMPLATE = SBATCH_TEMPLATE.replace("{VGL_CONDA_ACTIVATE}", VGL_CONDA_ACTIVATE).replace("{VGL_SLURM_EXCLUDE}", VGL_SLURM_EXCLUDE)


SKILL = "rotation"
VARIANT = "baseline_nz_finegrained"
CKPT_EVERY = 500


def has_completed_run(seed):
    seed_root = RESULTS_ROOT / SKILL / VARIANT / f"seed_{seed}"
    if not seed_root.exists():
        return False
    for child in seed_root.iterdir():
        if not child.is_dir():
            continue
        if any((child / "checkpoints").glob("final_*.pt")):
            return True
    return False


def build_command(seed):
    data_path = DATASET_ROOT / "rotation"
    results_dir = RESULTS_ROOT / SKILL / VARIANT / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    return [
        "scripts/train_rotation.py",
        "--data-path", str(data_path),
        "--results-dir", str(results_dir),
        "--epochs", "1000",
        "--global-batch-size", "128",
        "--num-workers", "4",
        "--global-seed", str(seed),
        "--image-size", "64",
        "--conditioning-method", "concat",
        "--architecture", "dit",
        "--model", "DiT-S/2",
        "--ckpt-every", str(CKPT_EVERY),
        "--rotation-embedding-type", "linear",
        "--rotation-dropout-prob", "0.0",
        "--null-angle", "180.0",
        "--null-embedding-type", "zero",
    ]


def render_slurm_script(seed):
    job_name = f"t2-{SKILL}-{VARIANT}-s{seed}"
    stdout_path = LOG_ROOT / f"{job_name}-%j.out"
    stderr_path = LOG_ROOT / f"{job_name}-%j.err"
    command = " ".join(shlex.quote(part) for part in build_command(seed))
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        repo_root=REPO_ROOT,
        command=command,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12, 13, 14])
    parser.add_argument("--submit", action="store_true",
                        help="Submit with sbatch (otherwise dry-run prints sbatch commands).")
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    SLURM_ROOT.mkdir(parents=True, exist_ok=True)

    submitted = []
    for seed in args.seeds:
        seed_root = RESULTS_ROOT / SKILL / VARIANT / f"seed_{seed}"
        script_path = SLURM_ROOT / f"{SKILL}-{VARIANT}-seed{seed}.slurm"

        if has_completed_run(seed):
            print(f"[SKIP] seed={seed} already has a final_*.pt under {seed_root}")
            continue

        script_path.write_text(render_slurm_script(seed))
        final_ckpt_glob = f"{seed_root}/<run_id>-DiT-S-2-rotation/checkpoints/final_*.pt"
        cmd_str = f"sbatch {script_path}"
        print(f"[seed={seed}]")
        print(f"  results_dir : {seed_root}")
        print(f"  slurm script: {script_path}")
        print(f"  expected ckpt: {final_ckpt_glob}")
        print(f"  sbatch cmd  : {cmd_str}")

        if args.submit:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"  -> {result.stdout.strip()}")
            submitted.append((seed, result.stdout.strip()))

    if args.submit and submitted:
        print("\nSubmitted:")
        for seed, out in submitted:
            print(f"  seed={seed}: {out}")


if __name__ == "__main__":
    main()
