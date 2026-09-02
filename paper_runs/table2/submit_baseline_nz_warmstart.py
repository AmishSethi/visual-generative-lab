#!/usr/bin/env python
"""Warm-start rotation `baseline_nz` from seed 4's step-20k checkpoint.

This is the diagnosis-implies-fix experiment from Appendix Section I:
    Hypothesis: rotation collapse is determined by step 20k of training.
    Prediction: warm-starting from a healthy seed's step-20k checkpoint with
    FRESH downstream training should reduce the 3/10 collapse rate toward 0/N.

Source checkpoint (seed 4 step 20k, headed for healthy basin -- extra MAE 26 deg
at step 20k, dropping to <5 deg at later steps):
    {RESULTS_ROOT}/rotation/baseline_nz/seed_4/000-DiT-S-2-rotation/checkpoints/0020000.pt

For each new seed in {100..104}:
  1. Stage a COPY of the source checkpoint at
       {RESULTS_ROOT}/rotation/baseline_nz_warmstart/seed_{N}/000-DiT-S-2-rotation/checkpoints/0020000.pt
     (We copy rather than passing --resume-from at the source path because
     scripts/train_rotation.py infers experiment_dir from the checkpoint's parent dir
     -- using the source path directly would write into seed_4's dir.)
  2. Submit a 1-GPU SLURM job that runs scripts/train_rotation.py with
       --resume-from <staged_copy>  --global-seed {N}
     using the same baseline_nz config (linear emb / concat / null=zero).
  3. Training continues to step 78000 (1000 epochs, same as fresh runs), so
     58000 additional steps after the warm-start. The final checkpoint will be
     written to .../checkpoints/final_0078000.pt -- compatible with the
     existing eval pipeline (paper_runs/table2/evaluate_table2.py).

scripts/train_rotation.py resume logic (verified at lines 394-406) loads:
    - model state_dict
    - EMA state_dict
    - optimizer state_dict
    - train_steps (= 20000)
    - epoch (= 256, then start_epoch = 257)
so the warm-start replays the full optimizer-and-EMA state of seed 4 at step
20k. Only the data-shuffling sampler seed and CUDA RNG state diverge (driven by
--global-seed = 100..104), which is exactly the "fresh downstream randomness"
the experiment requires.
"""
import argparse
import shlex
import shutil

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")
# Optional node exclusions for your cluster, e.g. "#SBATCH --exclude=node1,node2"
VGL_SLURM_EXCLUDE = _os.environ.get("VGL_SLURM_EXCLUDE", "")

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
VARIANT = "baseline_nz_warmstart"
SOURCE_CHECKPOINT = (
    RESULTS_ROOT
    / "rotation"
    / "baseline_nz"
    / "seed_4"
    / "000-DiT-S-2-rotation"
    / "checkpoints"
    / "0020000.pt"
)
EXPERIMENT_SUBDIR = "000-DiT-S-2-rotation"


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


def stage_warmstart_checkpoint(seed):
    """Copy the source step-20k ckpt into the new seed dir.

    scripts/train_rotation.py derives experiment_dir from the checkpoint's parent dir,
    so to keep outputs isolated to the new seed dir we must resume from a copy
    *inside* that dir (not the original source path).
    """
    seed_root = RESULTS_ROOT / SKILL / VARIANT / f"seed_{seed}"
    experiment_dir = seed_root / EXPERIMENT_SUBDIR
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    staged_ckpt = checkpoint_dir / "0020000.pt"
    if not staged_ckpt.exists():
        shutil.copy2(SOURCE_CHECKPOINT, staged_ckpt)
    return seed_root, staged_ckpt


def build_command(seed, staged_ckpt):
    data_path = DATASET_ROOT / "rotation"
    results_dir = RESULTS_ROOT / SKILL / VARIANT / f"seed_{seed}"

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
        "--ckpt-every", "2000",
        "--rotation-embedding-type", "linear",
        "--rotation-dropout-prob", "0.0",
        "--null-angle", "180.0",
        "--null-embedding-type", "zero",
        "--resume-from", str(staged_ckpt),
    ]


def render_slurm_script(seed, staged_ckpt):
    job_name = f"t2-{SKILL}-{VARIANT}-s{seed}"
    stdout_path = LOG_ROOT / f"{job_name}-%j.out"
    stderr_path = LOG_ROOT / f"{job_name}-%j.err"
    command = " ".join(shlex.quote(part) for part in build_command(seed, staged_ckpt))
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        repo_root=REPO_ROOT,
        command=command,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--submit", action="store_true",
                        help="Submit with sbatch (otherwise dry-run prints sbatch commands).")
    args = parser.parse_args()

    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Source warm-start checkpoint missing: {SOURCE_CHECKPOINT}"
        )

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    SLURM_ROOT.mkdir(parents=True, exist_ok=True)

    submitted = []
    for seed in args.seeds:
        seed_root = RESULTS_ROOT / SKILL / VARIANT / f"seed_{seed}"
        script_path = SLURM_ROOT / f"{SKILL}-{VARIANT}-seed{seed}.slurm"

        if has_completed_run(seed):
            print(f"[SKIP] seed={seed} already has a final_*.pt under {seed_root}")
            continue

        _, staged_ckpt = stage_warmstart_checkpoint(seed)
        script_path.write_text(render_slurm_script(seed, staged_ckpt))

        final_ckpt_glob = f"{seed_root}/{EXPERIMENT_SUBDIR}/checkpoints/final_0078000.pt"
        cmd_str = f"sbatch {script_path}"
        print(f"[seed={seed}]")
        print(f"  results_dir  : {seed_root}")
        print(f"  staged ckpt  : {staged_ckpt}")
        print(f"  slurm script : {script_path}")
        print(f"  expected ckpt: {final_ckpt_glob}")
        print(f"  sbatch cmd   : {cmd_str}")

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
