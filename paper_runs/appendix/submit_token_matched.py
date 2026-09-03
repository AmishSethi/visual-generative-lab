#!/usr/bin/env python
"""Token-matched pixel-vs-latent runs.

TOKEN-MATCHED.  Table 2 compares pixel at 1024 tokens against latent at 16, so it
cannot say whether the gap is the representation or the sequence length.  These two runs
fill in the ladder at a fixed 64x64 resolution:

    space    model      tokens   source
    latent   DiT-S/2      16     Table 2 (size extra 6.1)
    pixel    DiT-S/16     16     this script  <- matched to the latent baseline
    latent   DiT-S/1      64     this script  <- more tokens, same resolution
    pixel    DiT-S/2    1024     Table 2 (size extra 47.3)

All four are ~22M parameters, so capacity is matched as well.
"""
import argparse
import subprocess
from pathlib import Path

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")
# Optional node exclusions for your cluster, e.g. "#SBATCH --exclude=node1,node2"
VGL_SLURM_EXCLUDE = _os.environ.get("VGL_SLURM_EXCLUDE", "")


REPO = _os.environ.get("VGL_REPO", str(Path(__file__).resolve().parents[2]))
APPENDIX = f"{VGL_ROOT}/appendix"
T2 = f"{VGL_ROOT}/MORE_SEEDS/table2"
CONDA = VGL_CONDA_ACTIVATE

# Null-embedding args differ by skill and must match paper_runs/table2/manifest.py, or the
# comparison against the published rows is not apples to apples.  Position also rejects
# --null-embedding-type none.
NULL_ARGS = {
    "size": "--null-radius 0.0 --null-embedding-type none",
    "position": "--null-position 0.0 0.0 --null-embedding-type learnable",
    # Matches paper_runs/table2/manifest.py SKILLS["rotation"]["null_args"].
    "rotation": "--null-angle 180.0 --null-embedding-type learnable",
}

# (tag, skill, model, latent?, train script, dataset, embedding flag)
TOKEN_RUNS = [
    ("tok16-pixel-size", "size", "DiT-S/16", False, "scripts/train.py", f"{T2}/datasets/size", "radius"),
    ("tok64-latent-size", "size", "DiT-S/1", True, "scripts/train.py", f"{T2}/datasets/size", "radius"),
    ("tok16-pixel-position", "position", "DiT-S/16", False, "scripts/train_position.py", f"{T2}/datasets/position", "position"),
    ("tok64-latent-position", "position", "DiT-S/1", True, "scripts/train_position.py", f"{T2}/datasets/position", "position"),
    # Rotation has the widest pixel-vs-latent gap in Table 2 (68.0 vs 21.2), so it is
    # the most informative skill for whether sequence length explains that gap.
    ("tok16-pixel-rotation", "rotation", "DiT-S/16", False, "scripts/train_rotation.py", f"{T2}/datasets/rotation", "rotation"),
    ("tok64-latent-rotation", "rotation", "DiT-S/1", True, "scripts/train_rotation.py", f"{T2}/datasets/rotation", "rotation"),
]

HEADER = """#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
{VGL_SLURM_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --time={time}
#SBATCH --output={logs}/{tag}-%j.out
#SBATCH --error={logs}/{tag}-%j.err

export MASTER_PORT=$((29500 + SLURM_JOB_ID % 35000))
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd {repo}
{conda}
"""


def write(path, body):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(body)
    return path


def token_scripts(slurm_dir, logs):
    out = []
    for tag, skill, model, latent, script, data, emb in TOKEN_RUNS:
        res_dir = f"{APPENDIX}/results/token_matched/{tag}/seed_0"
        cmd = (
            f"torchrun --nproc_per_node=1 --master_port $MASTER_PORT {script} "
            f"--data-path {data} --results-dir {res_dir} "
            f"--epochs 1000 --global-batch-size 128 --num-workers 4 --global-seed 0 "
            f"--image-size 64 --conditioning-method concat --architecture dit "
            f"--model {model} --ckpt-every 5000 "
            f"--{emb}-embedding-type linear {NULL_ARGS[skill]}"
        )
        if latent:
            cmd += " --use-latent-diffusion"
        body = HEADER.format(tag=tag, time="36:00:00", logs=logs, repo=REPO, conda=CONDA) + "\n" + cmd + "\n"
        out.append(write(f"{slurm_dir}/{tag}.slurm", body))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    scripts = []
    d = f"{APPENDIX}/slurm/token_matched"
    lg = f"{APPENDIX}/logs"
    Path(lg).mkdir(parents=True, exist_ok=True)
    scripts += token_scripts(d, lg)

    for s in scripts:
        if a.dry_run:
            print("[dry]", s)
            continue
        r = subprocess.run(["sbatch", s], capture_output=True, text=True)
        print(f"{Path(s).name}: {(r.stdout or r.stderr).strip()}")


if __name__ == "__main__":
    main()
