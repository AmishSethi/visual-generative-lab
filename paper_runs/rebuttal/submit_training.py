#!/usr/bin/env python
"""Submit rebuttal training jobs.

Families:
  scaling     size/position/count at 10k/20k/40k/80k images, compute held fixed
              at manifest.BASE_STEPS gradient steps (Reviewer HeVs w3).
  hires       all four skills at 128x128 with DiT-S/4, token-matched to the
              paper's DiT-S/2 at 64x64 (Reviewers HeVs w2/w5, yLdq w1).
  hires_vae   size/position at 128x128 in VAE latent space, to test whether the
              paper's weak latent result is a low-resolution artifact (HeVs w2).
  shapevocab  shape x colour at S=4/8 and coverage 25/50/75 (Reviewer XDWw).

CUDA_LAUNCH_BLOCKING is deliberately NOT set here.  The paper-locked scripts set
it, which serialises every kernel launch and costs ~5x throughput; it is a
debugging flag with no effect on numerics.  Seeds 3-9 keep the original template
so they stay bit-comparable with seeds 0-2; these new runs do not need to.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.manifest import (
    DATASET_ROOT,
    HIRES_MODEL,
    LOG_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    SBATCH_TEMPLATE,
    SCALING_SIZES,
    SCALING_SKILLS,
    SKILLS,
    SLURM_ROOT,
    epochs_for,
)

SHAPEVOCAB_EPOCHS = 500
SHAPEVOCAB_BATCH = 128


def common_args(skill, data_path, results_dir, epochs, image_size, model, seed):
    spec = SKILLS[skill]
    return [
        spec["train_script"],
        "--data-path", str(data_path),
        "--results-dir", str(results_dir),
        "--epochs", str(epochs),
        "--global-batch-size", "128",
        "--num-workers", "4",
        "--global-seed", str(seed),
        "--image-size", str(image_size),
        "--conditioning-method", "concat",
        "--architecture", "dit",
        "--model", model,
        "--ckpt-every", "5000",
        f"--{spec['embedding_flag']}", "linear",
        f"--{spec['dropout_flag']}", "0.0",
        *spec["null_args"],
    ]


def jobs_scaling(seed):
    out = []
    for skill in SCALING_SKILLS:
        for n_images in SCALING_SIZES:
            tag = f"{n_images // 1000}k"
            data = DATASET_ROOT / "scaling" / f"{skill}_{tag}"
            results = RESULTS_ROOT / "scaling" / skill / tag / f"seed_{seed}"
            cmd = common_args(skill, data, results, epochs_for(n_images), 64, "DiT-S/2", seed)
            out.append((f"rb-scale-{skill}-{tag}-s{seed}", cmd, 1, "32:00:00", "32G"))
    return out


def jobs_hires(seed):
    out = []
    for skill in SKILLS:
        data = DATASET_ROOT / "hires128" / skill
        results = RESULTS_ROOT / "hires128" / skill / f"seed_{seed}"
        cmd = common_args(skill, data, results, 1000, 128, HIRES_MODEL, seed)
        out.append((f"rb-hires-{skill}-s{seed}", cmd, 1, "36:00:00", "48G"))
    return out


def jobs_hires_vae(seed):
    out = []
    # All four skills: the pixel-space hires128 family covers four, so leaving the
    # latent arm at two makes the comparison table asymmetric.
    for skill in SKILLS:
        data = DATASET_ROOT / "hires128" / skill
        results = RESULTS_ROOT / "hires128_vae" / skill / f"seed_{seed}"
        cmd = common_args(skill, data, results, 1000, 128, "DiT-S/2", seed)
        cmd.append("--use-latent-diffusion")
        out.append((f"rb-hiresvae-{skill}-s{seed}", cmd, 1, "36:00:00", "48G"))
    return out


def jobs_shapevocab(seed):
    out = []
    for n_shapes in (4, 8):
        for cov in (25, 50, 75):
            name = f"s{n_shapes}_cov{cov}"
            data = DATASET_ROOT / "shape_vocab" / name / "train"
            results = RESULTS_ROOT / "shape_vocab" / name / f"seed_{seed}"
            cmd = [
                "scripts/train_compositional.py",
                "--data-path", str(data),
                "--results-dir", str(results),
                "--include-properties", "shape", "color",
                "--num-shapes", str(n_shapes),
                "--epochs", str(SHAPEVOCAB_EPOCHS),
                "--global-batch-size", str(SHAPEVOCAB_BATCH),
                "--num-workers", "4",
                "--global-seed", str(seed),
                "--image-size", "64",
                "--conditioning-method", "concat",
                "--architecture", "dit",
                "--model", "DiT-S/2",
                "--ckpt-every", "5000",
                "--property-dropout-prob", "0.0",
                "--null-embedding-type", "learnable",
            ]
            out.append((f"rb-shapevocab-{name}-s{seed}", cmd, 1, "28:00:00", "32G"))
    return out


def jobs_text(seed):
    """Frozen text-encoder conditioning instead of numeric (Reviewer HeVs w1/w6).

    Trains on the paper's own canonical datasets so the only thing that changes
    is how the skill value reaches the model: a CLIP embedding of a sentence
    describing it, the way a text-to-image system is conditioned, rather than
    the scalar itself.
    """
    from paper_runs.rebuttal.manifest import REBUTTAL_ROOT, TABLE2_DATASETS

    out = []
    for skill in ("size", "count"):
        table = REBUTTAL_ROOT / "text_tables" / f"{skill}.pt"
        data = TABLE2_DATASETS / skill
        results = RESULTS_ROOT / "text_cond" / skill / f"seed_{seed}"
        cmd = common_args(skill, data, results, 1000, 64, "DiT-S/2", seed)
        # swap the embedding type set by common_args and point at the table
        flag = f"--{SKILLS[skill]['embedding_flag']}"
        cmd[cmd.index(flag) + 1] = "text"
        cmd.extend(["--radius-text-table", str(table)])
        out.append((f"rb-text-{skill}-s{seed}", cmd, 1, "32:00:00", "32G"))
    return out


def jobs_realistic(seed):
    """Visually complex renders, identical skill grids (Reviewer HeVs w5)."""
    out = []
    for skill in SKILLS:
        data = DATASET_ROOT / "realistic" / skill
        results = RESULTS_ROOT / "realistic" / skill / f"seed_{seed}"
        cmd = common_args(skill, data, results, 1000, 64, "DiT-S/2", seed)
        out.append((f"rb-real-{skill}-s{seed}", cmd, 1, "32:00:00", "32G"))
    return out


def jobs_ranges(seed):
    """Narrower / shifted training supports (Reviewer yLdq w2)."""
    from paper_runs.rebuttal.manifest import RANGE_SPECS

    out = []
    for name, spec in RANGE_SPECS.items():
        skill = spec["skill"]
        data = DATASET_ROOT / "ranges" / name
        results = RESULTS_ROOT / "ranges" / name / f"seed_{seed}"
        cmd = common_args(skill, data, results, 1000, 64, "DiT-S/2", seed)
        out.append((f"rb-range-{name}-s{seed}", cmd, 1, "32:00:00", "32G"))
    return out


def jobs_budget5k(seed):
    """The same 12-cell grid at a 5,000-image budget (Reviewer XDWw / AC).

    At 15,000 images eleven of the twelve cells score exactly 100%, so the design
    cannot separate coverage from K -- the task is simply too easy.  Five
    thousand puts the grid back on the part of the curve where the paper's own
    Table 3 shows differences, and each run is ~6h rather than ~19h.
    """
    out = []
    for n_shapes in (4, 8):
        for n_colors in (8, 16):
            for cov in (25, 50, 75):
                suffix = "" if n_colors == 8 else f"_c{n_colors}"
                name = f"s{n_shapes}{suffix}_cov{cov}_b5k"
                data = DATASET_ROOT / "shape_vocab" / name / "train"
                results = RESULTS_ROOT / "shape_vocab" / name / f"seed_{seed}"
                cmd = [
                    "scripts/train_compositional.py",
                    "--data-path", str(data),
                    "--results-dir", str(results),
                    "--include-properties", "shape", "color",
                    "--num-shapes", str(n_shapes),
                    "--num-colors", str(n_colors),
                    "--epochs", str(SHAPEVOCAB_EPOCHS),
                    "--global-batch-size", str(SHAPEVOCAB_BATCH),
                    "--num-workers", "4",
                    "--global-seed", str(seed),
                    "--image-size", "64",
                    "--conditioning-method", "concat",
                    "--architecture", "dit",
                    "--model", "DiT-S/2",
                    "--ckpt-every", "5000",
                    "--property-dropout-prob", "0.0",
                    "--null-embedding-type", "learnable",
                ]
                out.append((f"rb-b5k-{name}-s{seed}", cmd, 1, "12:00:00", "32G"))
    return out


def jobs_colorvocab(seed):
    """Second vocabulary axis: 16 colours as well as 8 (Reviewer XDWw / AC).

    With both S and C variable, the same K appears at different coverage
    fractions and the same coverage at different K, so the two can be
    estimated separately instead of only compared in one matched pair.
    """
    out = []
    for n_shapes in (4, 8):
        for cov in (25, 50, 75):
            name = f"s{n_shapes}_c16_cov{cov}"
            data = DATASET_ROOT / "shape_vocab" / name / "train"
            results = RESULTS_ROOT / "shape_vocab" / name / f"seed_{seed}"
            cmd = [
                "scripts/train_compositional.py",
                "--data-path", str(data),
                "--results-dir", str(results),
                "--include-properties", "shape", "color",
                "--num-shapes", str(n_shapes),
                "--num-colors", "16",
                "--epochs", str(SHAPEVOCAB_EPOCHS),
                "--global-batch-size", str(SHAPEVOCAB_BATCH),
                "--num-workers", "4",
                "--global-seed", str(seed),
                "--image-size", "64",
                "--conditioning-method", "concat",
                "--architecture", "dit",
                "--model", "DiT-S/2",
                "--ckpt-every", "5000",
                "--property-dropout-prob", "0.0",
                "--null-embedding-type", "learnable",
            ]
            out.append((f"rb-colorvocab-{name}-s{seed}", cmd, 1, "28:00:00", "32G"))
    return out


MASK_COVERAGES = (50, 25)


def jobs_maskvariance(seed):
    """Same coverage, five different random combination masks (yLdq question 2).

    Run at two coverages.  At 50% the task is saturated (all five masks land at
    or near 100%), so the spread there says little about whether some masks are
    easier; 25% sits off the ceiling and is the informative operating point.
    """
    out = []
    for cov, mask_seed in [(c, m) for c in MASK_COVERAGES for m in (42, 1, 2, 3, 4)]:
        name = f"s4_cov{cov}_mask{mask_seed}"
        data = DATASET_ROOT / "shape_vocab_masks" / name / "train"
        results = RESULTS_ROOT / "shape_vocab_masks" / name / f"seed_{seed}"
        cmd = [
            "scripts/train_compositional.py",
            "--data-path", str(data),
            "--results-dir", str(results),
            "--include-properties", "shape", "color",
            "--num-shapes", "4",
            "--epochs", str(SHAPEVOCAB_EPOCHS),
            "--global-batch-size", str(SHAPEVOCAB_BATCH),
            "--num-workers", "4",
            "--global-seed", str(seed),
            "--image-size", "64",
            "--conditioning-method", "concat",
            "--architecture", "dit",
            "--model", "DiT-S/2",
            "--ckpt-every", "5000",
            "--property-dropout-prob", "0.0",
            "--null-embedding-type", "learnable",
        ]
        out.append((f"rb-mask-{name}-s{seed}", cmd, 1, "28:00:00", "32G"))
    return out


FAMILIES = {
    "scaling": jobs_scaling,
    "hires": jobs_hires,
    "hires_vae": jobs_hires_vae,
    "shapevocab": jobs_shapevocab,
    "colorvocab": jobs_colorvocab,
    "budget5k": jobs_budget5k,
    "ranges": jobs_ranges,
    "realistic": jobs_realistic,
    "text": jobs_text,
    "maskvariance": jobs_maskvariance,
}


def main():
    parser = argparse.ArgumentParser(description="Submit rebuttal training jobs.")
    parser.add_argument("--families", nargs="+", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    SLURM_ROOT.mkdir(parents=True, exist_ok=True)

    for family in args.families:
        for seed in args.seeds:
            for job_name, command, num_gpus, time_limit, mem in FAMILIES[family](seed):
                results_dir = Path(command[command.index("--results-dir") + 1])
                if list(results_dir.rglob("final_*.pt")):
                    print(f"{job_name}: already complete, skipping")
                    continue
                results_dir.mkdir(parents=True, exist_ok=True)
                script = SLURM_ROOT / f"{job_name}.slurm"
                script.write_text(
                    SBATCH_TEMPLATE.format(
                        job_name=job_name,
                        num_gpus=num_gpus,
                        mem=mem,
                        time_limit=time_limit,
                        stdout_path=LOG_ROOT / f"{job_name}-%j.out",
                        stderr_path=LOG_ROOT / f"{job_name}-%j.err",
                        repo_root=REPO_ROOT,
                        command=" ".join(shlex.quote(p) for p in command),
                    )
                )
                if args.submit:
                    res = subprocess.run(["sbatch", str(script)], check=True,
                                         capture_output=True, text=True)
                    print(f"{job_name}: {res.stdout.strip()}")
                else:
                    print(f"{job_name}: {script}")


if __name__ == "__main__":
    main()
