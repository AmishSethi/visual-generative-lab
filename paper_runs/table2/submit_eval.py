#!/usr/bin/env python
from __future__ import annotations

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


    CANONICAL_SEEDS,
    EVAL_LOG_ROOT,
    EVAL_ROOT,
    EVAL_SLURM_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    SKILLS,
    VARIANTS,
)


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
{VGL_SLURM_EXCLUDE}
#SBATCH --time={time_limit}
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

python paper_runs/table2/evaluate_table2.py {command}
"""
SBATCH_TEMPLATE = SBATCH_TEMPLATE.replace("{VGL_CONDA_ACTIVATE}", VGL_CONDA_ACTIVATE).replace("{VGL_SLURM_EXCLUDE}", VGL_SLURM_EXCLUDE)


TIME_LIMITS = {
    "size": "16:00:00",
    "count": "16:00:00",
    "rotation": "30:00:00",
    "position": "36:00:00",
}


def has_completed_run(skill: str, variant: str, seed: int) -> bool:
    seed_root = RESULTS_ROOT / skill / variant / f"seed_{seed}"
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


def has_completed_eval(skill: str, variant: str, seed: int, output_root: Path) -> bool:
    return (output_root / skill / variant / f"seed_{seed}" / "results.json").exists()


def parse_dependency_entries(entries: list[str]) -> dict[tuple[str, str, int], str]:
    dependency_map = {}
    for entry in entries:
        try:
            skill, variant, seed, job_id = entry.split(":")
            dependency_map[(skill, variant, int(seed))] = job_id
        except ValueError as exc:
            raise ValueError(
                f"Invalid dependency entry '{entry}'. Expected skill:variant:seed:jobid."
            ) from exc
    return dependency_map


def build_command(skill: str, variant: str, seed: int, args: argparse.Namespace) -> list[str]:
    command = [
        "--skill",
        skill,
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--num-samples-per-condition",
        str(args.num_samples_per_condition),
        "--num-sampling-steps",
        str(args.num_sampling_steps),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--output-root",
        str(args.output_root),
        "--size-extrap-max", "25", "--position-extrap-min-margin", "0.0",  # base query grid (Table 2 uses the extended grid)
    ]
    return command


def render_slurm_script(skill: str, variant: str, seed: int, args: argparse.Namespace) -> str:
    job_name = f"t2eval-{skill}-{variant}-s{seed}"
    stdout_path = args.log_root / f"{job_name}-%j.out"
    stderr_path = args.log_root / f"{job_name}-%j.err"
    command = " ".join(shlex.quote(part) for part in build_command(skill, variant, seed, args))
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        time_limit=TIME_LIMITS[skill],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        repo_root=REPO_ROOT,
        command=command,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate or submit canonical Table 2 evaluation jobs.")
    parser.add_argument("--skills", nargs="+", default=list(SKILLS.keys()), choices=sorted(SKILLS.keys()))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=sorted(VARIANTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CANONICAL_SEEDS))
    parser.add_argument("--num-samples-per-condition", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=EVAL_ROOT / "samples20_steps250_cfg1")
    parser.add_argument("--slurm-root", type=Path, default=EVAL_SLURM_ROOT)
    parser.add_argument("--log-root", type=Path, default=EVAL_LOG_ROOT)
    parser.add_argument(
        "--dependency",
        action="append",
        default=[],
        help="Optional Slurm dependency in skill:variant:seed:jobid format.",
    )
    parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help="Render/submit jobs even if the final checkpoint is not present yet.",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.slurm_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    dependency_map = parse_dependency_entries(args.dependency)

    submitted = []
    for skill in args.skills:
        for variant in args.variants:
            for seed in args.seeds:
                if has_completed_eval(skill, variant, seed, args.output_root):
                    continue
                has_checkpoint = has_completed_run(skill, variant, seed)
                dependency = dependency_map.get((skill, variant, seed))
                if not has_checkpoint and not args.allow_missing_checkpoints and dependency is None:
                    continue
                script_path = args.slurm_root / f"{skill}-{variant}-seed{seed}.slurm"
                script_path.write_text(render_slurm_script(skill, variant, seed, args))
                if args.submit:
                    sbatch_command = ["sbatch"]
                    if dependency is not None:
                        sbatch_command.extend(["--dependency", f"afterok:{dependency}"])
                    sbatch_command.append(str(script_path))
                    result = subprocess.run(
                        sbatch_command,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    submitted.append((script_path.name, result.stdout.strip()))

    for script_name, output in submitted:
        print(f"{script_name}: {output}")


if __name__ == "__main__":
    main()
