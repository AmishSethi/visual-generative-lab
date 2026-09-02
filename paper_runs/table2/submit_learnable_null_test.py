#!/usr/bin/env python
"""Submit one seed per skill with null_embedding_type=learnable (instead of zero/none).

This tests whether using learnable null embeddings improves extrapolation for
size, position, and count (as it apparently does for rotation).

Results saved to a separate "learnable_null_test" subdirectory.
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
    ARTIFACT_ROOT,
)


LEARNABLE_TEST_ROOT = ARTIFACT_ROOT / "learnable_null_test"
LEARNABLE_SLURM_ROOT = LEARNABLE_TEST_ROOT / "slurm"
LEARNABLE_LOG_ROOT = LEARNABLE_TEST_ROOT / "logs"
LEARNABLE_RESULTS_ROOT = LEARNABLE_TEST_ROOT / "results"


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


# Skills to test with learnable null embedding (rotation already uses learnable)
SKILL_CONFIGS = {
    "size": {
        "script": "scripts/train.py",
        "dataset_dir": "size",
        "emb_flag": "--radius-embedding-type",
        "null_args": ["--null-radius", "0.0"],
        "dropout_flag": "--radius-dropout-prob",
    },
    "position": {
        "script": "scripts/train_position.py",
        "dataset_dir": "position",
        "emb_flag": "--position-embedding-type",
        "null_args": ["--null-position", "0.0", "0.0"],
        "dropout_flag": "--position-dropout-prob",
    },
    "count": {
        "script": "scripts/train_count.py",
        "dataset_dir": "count",
        "emb_flag": "--count-embedding-type",
        "null_args": ["--null-count", "0.0"],
        "dropout_flag": "--count-dropout-prob",
    },
}


def build_command(skill, seed):
    spec = SKILL_CONFIGS[skill]
    data_path = DATASET_ROOT / spec["dataset_dir"]
    results_dir = LEARNABLE_RESULTS_ROOT / skill / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    command = [
        spec["script"],
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
        spec["emb_flag"], "linear",
        spec["dropout_flag"], "0.0",
        *spec["null_args"],
        "--null-embedding-type", "learnable",  # <-- The key change
    ]
    return command


def render_slurm_script(skill, seed):
    job_name = f"t2-learn-{skill}-s{seed}"
    stdout_path = LEARNABLE_LOG_ROOT / f"{job_name}-%j.out"
    stderr_path = LEARNABLE_LOG_ROOT / f"{job_name}-%j.err"
    command = " ".join(shlex.quote(part) for part in build_command(skill, seed))
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        repo_root=REPO_ROOT,
        command=command,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", nargs="+", default=list(SKILL_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    LEARNABLE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    LEARNABLE_SLURM_ROOT.mkdir(parents=True, exist_ok=True)

    for skill in args.skills:
        for seed in args.seeds:
            script_path = LEARNABLE_SLURM_ROOT / f"{skill}-seed{seed}.slurm"
            script_path.write_text(render_slurm_script(skill, seed))
            if args.submit:
                result = subprocess.run(
                    ["sbatch", str(script_path)], check=True, capture_output=True, text=True
                )
                print(f"{script_path.name}: {result.stdout.strip()}")
            else:
                print(f"Generated: {script_path}")


if __name__ == "__main__":
    main()
