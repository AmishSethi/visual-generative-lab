#!/usr/bin/env python
"""Token-matched pixel-vs-latent runs, plus Hough re-evaluation of the count seeds.

TOKEN-MATCHED (HeVs W1/W2).  Table 2 compares pixel at 1024 tokens against latent at 16,
so it cannot say whether the gap is the representation or the sequence length.  These two
runs fill in the ladder at a fixed 64x64 resolution:

    space    model      tokens   status
    latent   DiT-S/2      16     published (size extra 6.1)
    pixel    DiT-S/16     16     NEW  <- matched to the latent baseline
    latent   DiT-S/1      64     NEW  <- more tokens, same resolution
    pixel    DiT-S/2    1024     published (size extra 47.3)

All four are ~22M parameters, so capacity is matched as well.

COUNT-HOUGH (yLdq W3 / Q3).  The paper's own footnote says the watershed counter
under-counts overlapping circles even on perfect ground truth, and reports the DiT-L count
cell with a Hough counter instead.  Our cross-validation agrees: on non-overlapping renders
watershed scores 93.8% against Hough's 99.7%, and with overlap 19.2% against 41.8%.  So we
rescore every count seed with Hough rather than retraining anything.
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
REB = f"{VGL_ROOT}/REBUTTAL"
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
        res_dir = f"{REB}/results/token_matched/{tag}/seed_0"
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


def hough_scripts(slurm_dir, logs, seeds):
    runner = f"{T2.replace('/table2', '')}/qualitative_debug/dit_large_count_inspection/run_count_hough_all.py"
    out = []
    for s in seeds:
        tag = f"cntH-baseline-s{s}"
        cmd = (
            f"python {runner} --variant baseline --seed {s} "
            f"--num-samples 20 --num-steps 250 --batch-size 20 "
            f"--output-root {T2}/eval/count_hough"
        )
        body = HEADER.format(tag=tag, time="01:00:00", logs=logs, repo=REPO, conda=CONDA) + "\n" + cmd + "\n"
        out.append(write(f"{slurm_dir}/{tag}.slurm", body))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["tokens", "hough", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    scripts = []
    if a.what in ("tokens", "both"):
        d = f"{REB}/slurm/token_matched"
        lg = f"{REB}/logs"
        Path(lg).mkdir(parents=True, exist_ok=True)
        scripts += token_scripts(d, lg)
    if a.what in ("hough", "both"):
        d = f"{T2}/eval/count_hough/slurm"
        lg = f"{T2}/eval/count_hough/logs"
        Path(lg).mkdir(parents=True, exist_ok=True)
        scripts += hough_scripts(d, lg, a.seeds)

    for s in scripts:
        if a.dry_run:
            print("[dry]", s)
            continue
        r = subprocess.run(["sbatch", s], capture_output=True, text=True)
        print(f"{Path(s).name}: {(r.stdout or r.stderr).strip()}")


if __name__ == "__main__":
    main()
