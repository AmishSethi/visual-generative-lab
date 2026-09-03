"""Paths and specs for the appendix experiments.

Everything here lives under APPENDIX_ROOT so the paper-locked table2/table3
artifacts are never touched.
"""
from pathlib import Path

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")
# Optional node exclusions for your cluster, e.g. "#SBATCH --exclude=node1,node2"
VGL_SLURM_EXCLUDE = _os.environ.get("VGL_SLURM_EXCLUDE", "")


REPO_ROOT = Path(__file__).resolve().parents[2]
APPENDIX_ROOT = Path(f"{VGL_ROOT}/appendix")

DATASET_ROOT = APPENDIX_ROOT / "datasets"
RESULTS_ROOT = APPENDIX_ROOT / "results"
SLURM_ROOT = APPENDIX_ROOT / "slurm"
LOG_ROOT = APPENDIX_ROOT / "logs"
EVAL_ROOT = APPENDIX_ROOT / "eval"

# Canonical table2 artifacts we read from (never write).
TABLE2_ROOT = Path(f"{VGL_ROOT}/MORE_SEEDS/table2")
TABLE2_DATASETS = TABLE2_ROOT / "datasets"
TABLE2_RESULTS = TABLE2_ROOT / "results"

DATASET_SEED = 42

# ---------------------------------------------------------------------------
# Data scaling: does more data fix extrapolation?
#
# Compute is held fixed at BASE_STEPS gradient steps for every dataset size, so
# the only thing that changes across the sweep is the number of unique training
# images.  epochs = BASE_STEPS * batch / n_images.
# ---------------------------------------------------------------------------
BASE_STEPS = 78125
BASE_BATCH = 128
SCALING_SIZES = (10000, 20000, 40000, 80000)
SCALING_SKILLS = ("size", "position", "count", "rotation")


def epochs_for(n_images: int) -> int:
    """Epoch count that yields BASE_STEPS gradient steps at BASE_BATCH."""
    return max(1, round(BASE_STEPS * BASE_BATCH / n_images))


# ---------------------------------------------------------------------------
# Resolution: 128x128 replication.
#
# Every length scales by 2 so the images are geometrically identical to the
# 64x64 ones, just rendered at twice the resolution.  DiT-S/4 at 128 gives the
# same 32x32=1024 token grid as DiT-S/2 at 64, so capacity and compute per step
# are matched and resolution is the only variable.
# ---------------------------------------------------------------------------
RES_SCALE = 2
HIRES_IMAGE_SIZE = 128
HIRES_MODEL = "DiT-S/4"

HIRES_SPECS = {
    "size": {
        "total_images": 10000,
        "image_size": HIRES_IMAGE_SIZE,
        "train_radii": list(range(10, 41, 2)),  # 2x of 5..20 -> 16 values
        "seed": DATASET_SEED,
    },
    "position": {
        "total_images": 10000,
        "image_size": HIRES_IMAGE_SIZE,
        "circle_radius": 16,
        "grid_min": -36.0,
        "grid_max": 36.0,
        "positions_per_axis": 32,
        "seed": DATASET_SEED,
    },
    "rotation": {
        "total_images": 10000,
        "image_size": HIRES_IMAGE_SIZE,
        "shape_size": 40,
        "min_angle": 45,
        "max_angle": 315,
        "step_degrees": 1,
        "shape_type": "arrow",
        "seed": DATASET_SEED,
    },
    "count": {
        "total_images": 10000,
        "image_size": HIRES_IMAGE_SIZE,
        "counts": list(range(2, 8)),
        "circle_radius": 16,
        "seed": DATASET_SEED,
    },
}

# Eval thresholds at 128x128, scaled so they mean the same thing as at 64x64.
HIRES_THRESHOLDS = {
    "size_iou": 0.90,          # scale-free
    "position_px": 4.0,        # 2 px at 64x64-equivalent
    "rotation_deg": 5.0,       # scale-free
}

# ---------------------------------------------------------------------------
# Ranges: does the extrapolation boundary follow the training support?
#
# Extrapolation failures could come from the condition encoding, numerical
# normalization, training range design, or sampling rather than from the model
# not learning the visual rule.  Encoding is already ablated in the paper.  This
# varies the training range instead: if the model always breaks a fixed number
# of *pixels* past the boundary regardless of where that boundary is, the limit
# is about numeric scale; if it breaks a fixed *fraction of the training span*,
# the limit is about the support, which is the paper's claim.
# ---------------------------------------------------------------------------
RANGE_SPECS = {
    "size_wide": {          # the paper's range, regenerated for a like-for-like base
        "skill": "size",
        "train_radii": list(range(5, 21)),
        "total_images": 10000,
        "image_size": 64,
        "seed": DATASET_SEED,
    },
    "size_narrow": {        # half the span, centred inside the paper's range
        "skill": "size",
        "train_radii": list(range(9, 17)),
        "total_images": 10000,
        "image_size": 64,
        "seed": DATASET_SEED,
    },
    "size_shifted": {       # same span as the paper, shifted upward
        "skill": "size",
        "train_radii": list(range(11, 27)),
        "total_images": 10000,
        "image_size": 64,
        "seed": DATASET_SEED,
    },
    "rotation_narrow": {    # 180 deg of support instead of 270
        "skill": "rotation",
        "min_angle": 90,
        "max_angle": 270,
        "step_degrees": 1,
        "shape_size": 20,
        "shape_type": "arrow",
        "total_images": 10000,
        "image_size": 64,
        "seed": DATASET_SEED,
    },
}

# ---------------------------------------------------------------------------
# Skill -> training script / flag names (mirrors table2 manifest).
# ---------------------------------------------------------------------------
SKILLS = {
    "size": {
        "train_script": "scripts/train.py",
        "embedding_flag": "radius-embedding-type",
        "dropout_flag": "radius-dropout-prob",
        "null_args": ["--null-radius", "0.0", "--null-embedding-type", "none"],
    },
    "position": {
        "train_script": "scripts/train_position.py",
        "embedding_flag": "position-embedding-type",
        "dropout_flag": "position-dropout-prob",
        "null_args": ["--null-position", "0.0", "0.0", "--null-embedding-type", "learnable"],
    },
    "rotation": {
        "train_script": "scripts/train_rotation.py",
        "embedding_flag": "rotation-embedding-type",
        "dropout_flag": "rotation-dropout-prob",
        "null_args": ["--null-angle", "180.0", "--null-embedding-type", "learnable"],
    },
    "count": {
        "train_script": "scripts/train_count.py",
        "embedding_flag": "count-embedding-type",
        "dropout_flag": "count-dropout-prob",
        "null_args": ["--null-count", "0.0", "--null-embedding-type", "none"],
    },
}

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
{VGL_SLURM_EXCLUDE}
#SBATCH --gres=gpu:{num_gpus}
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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_USE_CUDA_DSA=1

cd {repo_root}
{VGL_CONDA_ACTIVATE}

torchrun --nproc_per_node={num_gpus} --master_port $MASTER_PORT {command}
"""
SBATCH_TEMPLATE = SBATCH_TEMPLATE.replace("{VGL_CONDA_ACTIVATE}", VGL_CONDA_ACTIVATE).replace("{VGL_SLURM_EXCLUDE}", VGL_SLURM_EXCLUDE)

CPU_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={stdout_path}
#SBATCH --error={stderr_path}

cd {repo_root}
{VGL_CONDA_ACTIVATE}

{command}
"""
CPU_SBATCH_TEMPLATE = CPU_SBATCH_TEMPLATE.replace("{VGL_CONDA_ACTIVATE}", VGL_CONDA_ACTIVATE).replace("{VGL_SLURM_EXCLUDE}", VGL_SLURM_EXCLUDE)
