
import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")
# Optional node exclusions for your cluster, e.g. "#SBATCH --exclude=node1,node2"
VGL_SLURM_EXCLUDE = _os.environ.get("VGL_SLURM_EXCLUDE", "")

#!/usr/bin/env python
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.manifest import (


    CANONICAL_SEEDS,
    DATASET_ROOT,
    LOG_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    SKILLS,
    SLURM_ROOT,
    VARIANTS,
)


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:{num_gpus}
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

torchrun --nproc_per_node={num_gpus} --master_port $MASTER_PORT {command}
"""
SBATCH_TEMPLATE = SBATCH_TEMPLATE.replace("{VGL_CONDA_ACTIVATE}", VGL_CONDA_ACTIVATE).replace("{VGL_SLURM_EXCLUDE}", VGL_SLURM_EXCLUDE)


MICRO_BATCH_OVERRIDES = {
    ("size", "adaln"): 64,
    ("position", "adaln"): 32,
    ("rotation", "adaln"): 32,
    ("count", "adaln"): 64,
    ("size", "unet"): 16,
    ("position", "unet"): 16,
    ("rotation", "unet"): 16,
    ("count", "unet"): 16,
    ("size", "dit_large"): 16,
    ("position", "dit_large"): 16,
    ("rotation", "dit_large"): 16,
    ("count", "dit_large"): 16,
}


def has_completed_run(skill, variant, seed):
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


def build_command(skill, variant, seed):
    skill_spec = SKILLS[skill]
    variant_spec = VARIANTS[variant]
    data_path = DATASET_ROOT / skill_spec["dataset_dir"]
    results_dir = RESULTS_ROOT / skill / variant / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    command = [
        skill_spec["train_script"],
        "--data-path",
        str(data_path),
        "--results-dir",
        str(results_dir),
        "--epochs",
        str(skill_spec.get("epochs", 1000)),
        "--global-batch-size",
        "128",
        "--num-workers",
        "4",
        "--global-seed",
        str(seed),
        "--image-size",
        "64",
        "--conditioning-method",
        variant_spec["conditioning"],
        "--architecture",
        variant_spec["architecture"],
        "--model",
        variant_spec["model"],
        "--ckpt-every",
        "2000",
        f"--{skill_spec['embedding_flag']}",
        variant_spec["embedding"],
        f"--{skill_spec['dropout_flag']}",
        "0.0",
        *skill_spec["null_args"],
    ]

    micro_batch_size = MICRO_BATCH_OVERRIDES.get((skill, variant))
    if micro_batch_size is not None:
        command.extend(["--micro-batch-size", str(micro_batch_size)])

    if variant_spec["latent"]:
        command.append("--use-latent-diffusion")
    if variant_spec["flow_matching"]:
        command.append("--use-flow-matching")

    return command


def render_slurm_script(skill, variant, seed):
    variant_spec = VARIANTS[variant]
    job_name = f"t2-{skill}-{variant}-s{seed}"
    stdout_path = LOG_ROOT / f"{job_name}-%j.out"
    stderr_path = LOG_ROOT / f"{job_name}-%j.err"
    command = " ".join(shlex.quote(part) for part in build_command(skill, variant, seed))
    num_gpus = variant_spec.get("num_gpus", 1)
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        time_limit=variant_spec["time_limit"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        repo_root=REPO_ROOT,
        command=command,
        num_gpus=num_gpus,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate or submit canonical Table-2 training jobs.")
    parser.add_argument("--skills", nargs="+", default=list(SKILLS.keys()), choices=sorted(SKILLS.keys()))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=sorted(VARIANTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CANONICAL_SEEDS))
    parser.add_argument("--submit", action="store_true", help="Submit the generated scripts with sbatch.")
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    SLURM_ROOT.mkdir(parents=True, exist_ok=True)

    submitted = []
    for skill in args.skills:
        for variant in args.variants:
            for seed in args.seeds:
                if has_completed_run(skill, variant, seed):
                    continue
                script_path = SLURM_ROOT / f"{skill}-{variant}-seed{seed}.slurm"
                script_path.write_text(render_slurm_script(skill, variant, seed))
                if args.submit:
                    result = subprocess.run(
                        ["sbatch", str(script_path)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    submitted.append((script_path.name, result.stdout.strip()))

    for script_name, output in submitted:
        print(f"{script_name}: {output}")


if __name__ == "__main__":
    main()
