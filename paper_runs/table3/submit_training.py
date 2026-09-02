#!/usr/bin/env python
"""Generate and submit 3-seed compositional training jobs for Table 3."""
import argparse
import os as _os
import subprocess
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")
# Optional node exclusions for your cluster, e.g. "#SBATCH --exclude=node1,node2"
VGL_SLURM_EXCLUDE = _os.environ.get("VGL_SLURM_EXCLUDE", "")
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table3.manifest import (
    CANONICAL_SEEDS,
    COVERAGES,
    DATASET_ROOT,
    LOG_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    SKILL_PAIRS,
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
#SBATCH --time=40:00:00
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


def has_completed_run(pair_name, coverage, seed):
    seed_root = RESULTS_ROOT / pair_name / f"cov{coverage}" / f"seed_{seed}"
    if not seed_root.exists():
        return False
    for child in seed_root.iterdir():
        if not child.is_dir():
            continue
        prefix = child.name.split("-", 1)[0]
        if not prefix.isdigit():
            continue
        if any((child / "checkpoints").glob("final_*.pt")):
            return True
    return False


def build_command(pair_name, coverage, seed):
    pair_spec = SKILL_PAIRS[pair_name]
    data_path = DATASET_ROOT / f"{pair_spec['dataset_prefix']}_cov{coverage}"
    results_dir = RESULTS_ROOT / pair_name / f"cov{coverage}" / f"seed_{seed}"

    command = [
        "train_compositional.py",
        "--data-path", str(data_path / "train"),
        "--include-properties", *pair_spec["properties"],
        "--results-dir", str(results_dir),
        "--epochs", "1000",
        "--global-batch-size", "64",
        "--num-workers", "4",
        "--global-seed", str(seed),
        "--image-size", "64",
        "--conditioning-method", "concat",
        "--radius-embedding-type", "linear",
        "--position-embedding-type", "linear",
        "--rotation-embedding-type", "linear",
        "--property-dropout-prob", "0.0",
        "--null-radius", "0.0",
        "--null-position", "0.0", "0.0",
        "--null-shape-id", "0",
        "--null-color-id", "0",
        "--null-count", "1",
        "--null-rotation", "0.0",
        "--null-embedding-type", "learnable",
        "--model", "DiT-S/2",
        "--ckpt-every", "5000",
    ]
    return command


def render_slurm_script(pair_name, coverage, seed):
    job_name = f"t3-{pair_name}-c{coverage}-s{seed}"
    stdout_path = LOG_ROOT / f"{job_name}-%j.out"
    stderr_path = LOG_ROOT / f"{job_name}-%j.err"
    command = " ".join(f"'{part}'" if " " in part else part for part in build_command(pair_name, coverage, seed))
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        repo_root=REPO_ROOT,
        command=command,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate or submit Table 3 compositional training jobs.")
    parser.add_argument("--pairs", nargs="+", default=list(SKILL_PAIRS.keys()), choices=sorted(SKILL_PAIRS.keys()))
    parser.add_argument("--coverages", nargs="+", type=int, default=list(COVERAGES), choices=sorted(COVERAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CANONICAL_SEEDS))
    parser.add_argument("--submit", action="store_true", help="Submit the generated scripts with sbatch.")
    parser.add_argument("--dependency", default=None, help="Slurm job id; every submitted job runs afterok:<id>.")
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    SLURM_ROOT.mkdir(parents=True, exist_ok=True)

    submitted = []
    skipped = []
    for pair_name in args.pairs:
        for coverage in args.coverages:
            for seed in args.seeds:
                if has_completed_run(pair_name, coverage, seed):
                    skipped.append(f"{pair_name}/cov{coverage}/seed_{seed}")
                    continue
                script_path = SLURM_ROOT / f"{pair_name}-cov{coverage}-seed{seed}.slurm"
                script_path.write_text(render_slurm_script(pair_name, coverage, seed))
                if args.submit:
                    sbatch_cmd = ["sbatch"] + (["--dependency", f"afterok:{args.dependency}"] if args.dependency else [])
                    result = subprocess.run(
                        sbatch_cmd + [str(script_path)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    submitted.append((script_path.name, result.stdout.strip()))

    if skipped:
        print(f"Skipped {len(skipped)} completed runs")
    for script_name, output in submitted:
        print(f"{script_name}: {output}")


if __name__ == "__main__":
    main()
