#!/usr/bin/env python
"""Generate and submit 3-seed compositional evaluation jobs for Table 3."""
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
    EVAL_ROOT,
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

python paper_runs/table3/evaluate_table3.py {command}
"""
SBATCH_TEMPLATE = SBATCH_TEMPLATE.replace("{VGL_CONDA_ACTIVATE}", VGL_CONDA_ACTIVATE).replace("{VGL_SLURM_EXCLUDE}", VGL_SLURM_EXCLUDE)

# Pairs with continuous skills need longer eval time
TIME_LIMITS = {
    "color_shape": "08:00:00",
    "color_count": "08:00:00",
    "shape_count": "08:00:00",
    "color_position": "16:00:00",
    "position_size": "24:00:00",
    "position_shape": "16:00:00",
    "size_shape": "16:00:00",
}


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


def has_completed_eval(pair_name, coverage, seed, output_root):
    return (output_root / pair_name / f"cov{coverage}" / f"seed_{seed}" / "results.json").exists()


def parse_dependency_entries(entries):
    dependency_map = {}
    for entry in entries:
        try:
            pair, cov, seed, job_id = entry.split(":")
            dependency_map[(pair, int(cov), int(seed))] = job_id
        except ValueError as exc:
            raise ValueError(
                f"Invalid dependency entry '{entry}'. Expected pair:coverage:seed:jobid."
            ) from exc
    return dependency_map


def build_command(pair_name, coverage, seed, args):
    command = [
        "--pair", pair_name,
        "--coverage", str(coverage),
        "--seed", str(seed),
        "--num-samples", str(args.num_samples),
        "--num-sampling-steps", str(args.num_sampling_steps),
        "--eval-batch-size", str(args.eval_batch_size),
        "--output-root", str(args.output_root),
    ]
    return command


def render_slurm_script(pair_name, coverage, seed, args):
    job_name = f"t3eval-{pair_name}-c{coverage}-s{seed}"
    stdout_path = args.log_root / f"{job_name}-%j.out"
    stderr_path = args.log_root / f"{job_name}-%j.err"
    command = " ".join(build_command(pair_name, coverage, seed, args))
    time_limit = TIME_LIMITS.get(pair_name, "16:00:00")
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        time_limit=time_limit,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        repo_root=REPO_ROOT,
        command=command,
    )


def main():
    default_output = EVAL_ROOT / "samples20_steps250_cfg1"
    parser = argparse.ArgumentParser(description="Generate or submit Table 3 evaluation jobs.")
    parser.add_argument("--pairs", nargs="+", default=list(SKILL_PAIRS.keys()), choices=sorted(SKILL_PAIRS.keys()))
    parser.add_argument("--coverages", nargs="+", type=int, default=list(COVERAGES), choices=sorted(COVERAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CANONICAL_SEEDS))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--slurm-root", type=Path, default=SLURM_ROOT / "eval")
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT)
    parser.add_argument(
        "--dependency", action="append", default=[],
        help="Optional Slurm dependency in pair:coverage:seed:jobid format.",
    )
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.slurm_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    dependency_map = parse_dependency_entries(args.dependency)

    submitted = []
    skipped_done = 0
    skipped_no_ckpt = 0
    for pair_name in args.pairs:
        for coverage in args.coverages:
            for seed in args.seeds:
                if has_completed_eval(pair_name, coverage, seed, args.output_root):
                    skipped_done += 1
                    continue
                has_ckpt = has_completed_run(pair_name, coverage, seed)
                dep = dependency_map.get((pair_name, coverage, seed))
                if not has_ckpt and not args.allow_missing_checkpoints and dep is None:
                    skipped_no_ckpt += 1
                    continue

                script_path = args.slurm_root / f"{pair_name}-cov{coverage}-seed{seed}.slurm"
                script_path.write_text(render_slurm_script(pair_name, coverage, seed, args))

                if args.submit:
                    sbatch_cmd = ["sbatch"]
                    if dep is not None:
                        sbatch_cmd.extend(["--dependency", f"afterok:{dep}"])
                    sbatch_cmd.append(str(script_path))
                    result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
                    submitted.append((script_path.name, result.stdout.strip()))

    if skipped_done:
        print(f"Skipped {skipped_done} already-evaluated runs")
    if skipped_no_ckpt:
        print(f"Skipped {skipped_no_ckpt} runs without checkpoints (use --allow-missing-checkpoints)")
    for script_name, output in submitted:
        print(f"{script_name}: {output}")


if __name__ == "__main__":
    main()
