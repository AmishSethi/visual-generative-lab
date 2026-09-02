from pathlib import Path

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")



REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(f"{VGL_ROOT}/MORE_SEEDS/table2")
DATASET_ROOT = ARTIFACT_ROOT / "datasets"
RESULTS_ROOT = ARTIFACT_ROOT / "results"
SLURM_ROOT = ARTIFACT_ROOT / "slurm"
LOG_ROOT = ARTIFACT_ROOT / "logs"
EVAL_ROOT = ARTIFACT_ROOT / "eval"
EVAL_SLURM_ROOT = ARTIFACT_ROOT / "eval_slurm"
EVAL_LOG_ROOT = ARTIFACT_ROOT / "eval_logs"
EVAL_ROOT = ARTIFACT_ROOT / "eval"
EVAL_RESULTS_ROOT = EVAL_ROOT / "results"
EVAL_SLURM_ROOT = EVAL_ROOT / "slurm"
EVAL_LOG_ROOT = EVAL_ROOT / "logs"

CANONICAL_SEEDS = (0, 1, 2)
DATASET_SEED = 42
EVAL_NUM_SAMPLES = 20
EVAL_NUM_SAMPLING_STEPS = 250
EVAL_CFG_SCALE = 1.0

TABLE_ROWS = {
    "DiT-S/2 (baseline)": "baseline",
    "+ sinusoidal emb.": "sinusoidal",
    "+ rotary emb.": "rotary",
    "+ linear emb.": "baseline",
    "+ AdaLN": "adaln",
    "+ concatenation": "baseline",
    "+ VAE latent": "vae",
    "+ pixel": "baseline",
    "+ flow matching": "flow",
    "+ U-Net (capacity-matched)": "unet",
}

VARIANTS = {
    "baseline": {
        "architecture": "dit",
        "model": "DiT-S/2",
        "embedding": "linear",
        "conditioning": "concat",
        "latent": False,
        "flow_matching": False,
        "time_limit": "30:00:00",
    },
    "sinusoidal": {
        "architecture": "dit",
        "model": "DiT-S/2",
        "embedding": "sinusoidal",
        "conditioning": "concat",
        "latent": False,
        "flow_matching": False,
        "time_limit": "30:00:00",
    },
    "rotary": {
        "architecture": "dit",
        "model": "DiT-S/2",
        "embedding": "rotary",
        "conditioning": "concat",
        "latent": False,
        "flow_matching": False,
        "time_limit": "30:00:00",
    },
    "adaln": {
        "architecture": "dit",
        "model": "DiT-S/2",
        "embedding": "linear",
        "conditioning": "adaln",
        "latent": False,
        "flow_matching": False,
        "time_limit": "30:00:00",
    },
    "vae": {
        "architecture": "dit",
        "model": "DiT-S/2",
        "embedding": "linear",
        "conditioning": "concat",
        "latent": True,
        "flow_matching": False,
        "time_limit": "36:00:00",
    },
    "flow": {
        "architecture": "dit",
        "model": "DiT-S/2",
        "embedding": "linear",
        "conditioning": "concat",
        "latent": False,
        "flow_matching": True,
        "time_limit": "48:00:00",
    },
    "unet": {
        "architecture": "unet",
        "model": "UNet-DiT-S2-matched",
        "embedding": "linear",
        "conditioning": "concat",
        "latent": False,
        "flow_matching": False,
        "time_limit": "48:00:00",
        "num_gpus": 4,
    },
    "dit_large": {
        "architecture": "dit",
        "model": "DiT-L/2",
        "embedding": "linear",
        "conditioning": "concat",
        "latent": False,
        "flow_matching": False,
        "time_limit": "48:00:00",
        "num_gpus": 4,
    },
}

SKILLS = {
    "size": {
        "train_script": "scripts/train.py",
        "dataset_dir": "size",
        "embedding_flag": "radius-embedding-type",
        "dropout_flag": "radius-dropout-prob",
        "null_args": ["--null-radius", "0.0", "--null-embedding-type", "none"],
        "max_samples_flag": "max-samples-per-class",
    },
    "position": {
        "train_script": "scripts/train_position.py",
        "dataset_dir": "position",
        "embedding_flag": "position-embedding-type",
        "dropout_flag": "position-dropout-prob",
        "null_args": ["--null-position", "0.0", "0.0", "--null-embedding-type", "learnable"],
        "max_samples_flag": "max-samples-per-position",
    },
    "rotation": {
        "train_script": "scripts/train_rotation.py",
        "dataset_dir": "rotation",
        "embedding_flag": "rotation-embedding-type",
        "dropout_flag": "rotation-dropout-prob",
        "null_args": ["--null-angle", "180.0", "--null-embedding-type", "learnable"],
        "max_samples_flag": "max-samples-per-angle",
    },
    "count": {
        "train_script": "scripts/train_count.py",
        "dataset_dir": "count",
        "embedding_flag": "count-embedding-type",
        "dropout_flag": "count-dropout-prob",
        "null_args": ["--null-count", "0.0", "--null-embedding-type", "none"],
        "max_samples_flag": "max-samples-per-class",
    },
}

DATASET_SPECS = {
    "size": {
        "total_images": 10000,
        "image_size": 64,
        "train_radii": list(range(5, 21)),
        "seed": DATASET_SEED,
    },
    "position": {
        "total_images": 10000,
        "image_size": 64,
        "circle_radius": 8,
        "grid_min": -18.0,
        "grid_max": 18.0,
        "positions_per_axis": 32,
        "seed": DATASET_SEED,
    },
    "rotation": {
        "total_images": 10000,
        "image_size": 64,
        "shape_size": 20,
        "min_angle": 45,
        "max_angle": 315,
        "step_degrees": 1,
        "shape_type": "arrow",
        "seed": DATASET_SEED,
    },
    "count": {
        "total_images": 10000,
        "image_size": 64,
        "counts": list(range(2, 8)),
        "circle_radius": 8,
        "seed": DATASET_SEED,
    },
}
