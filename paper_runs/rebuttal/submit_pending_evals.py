#!/usr/bin/env python
"""Submit each family's evaluation as soon as its training finishes.

Training runs finish at staggered times over roughly a day.  Rather than wait
for the slowest, this checks every family for completed checkpoints and submits
the matching evaluation once, recording a stamp file so re-running is safe.
"""

import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.manifest import (


    EVAL_ROOT, LOG_ROOT, RANGE_SPECS, REPO_ROOT, RESULTS_ROOT, SCALING_SIZES,
    SCALING_SKILLS, TABLE2_RESULTS,
)

DIT_ENV = VGL_CONDA_ACTIVATE
STAMP_DIR = EVAL_ROOT / ".submitted"


def has_final(path: Path) -> bool:
    return path.exists() and any(path.rglob("final_*.pt"))


def families():
    """(name, [required run dirs], sbatch command, gpus, time)."""
    out = []

    scaling_dirs = [RESULTS_ROOT / "scaling" / s / f"{n // 1000}k" / "seed_0"
                    for s in SCALING_SKILLS for n in SCALING_SIZES]
    out.append(("scaling", scaling_dirs,
                "python paper_runs/rebuttal/eval_scaling.py --max-conditions-per-split 100",
                1, "12:00:00"))

    out.append(("hires128", [RESULTS_ROOT / "hires128" / s / "seed_0"
                             for s in ("size", "position", "rotation", "count")],
                "python paper_runs/rebuttal/eval_hires.py --family hires128",
                1, "16:00:00"))

    out.append(("hires128_vae", [RESULTS_ROOT / "hires128_vae" / s / "seed_0"
                                 for s in ("size", "position")],
                "python paper_runs/rebuttal/eval_hires.py --family hires128_vae --skills size position",
                1, "12:00:00"))

    # Split each seed into the 8-colour and 16-colour halves.  The C=16 cells
    # carry four times the combinations, so a single job for all twelve lands on
    # its walltime; two jobs also halve the wall-clock.  Outputs differ by a
    # suffix that the analysis merges and de-duplicates.
    c8_runs = " ".join(f"s{s}_cov{c}" for s in (4, 8) for c in (25, 50, 75))
    c16_runs = " ".join(f"s{s}_c16_cov{c}" for s in (4, 8) for c in (25, 50, 75))
    for seed in (0, 1, 2):
        for half, runs, suffix, limit in (("c8", c8_runs, "", "06:00:00"),
                                          ("c16", c16_runs, "b", "10:00:00")):
            dirs = [RESULTS_ROOT / "shape_vocab" / name / f"seed_{seed}"
                    for name in runs.split()]
            out.append((f"shape_vocab_seed{seed}_{half}", dirs,
                        f"python paper_runs/rebuttal/eval_shape_vocab.py --seed {seed} "
                        f"--runs {runs} --out {EVAL_ROOT}/shape_vocab_results_seed{seed}{suffix}.json "
                        f"&& python -m paper_runs.rebuttal.analyze_coverage_vs_k",
                        1, limit))

    out.append(("shape_vocab_masks",
                [RESULTS_ROOT / "shape_vocab_masks" / f"s4_cov{c}_mask{m}" / "seed_0"
                 for c in (50, 25) for m in (42, 1, 2, 3, 4)],
                "python paper_runs/rebuttal/eval_shape_vocab.py --family shape_vocab_masks",
                1, "06:00:00"))

    out.append(("ranges", [RESULTS_ROOT / "ranges" / n / "seed_0" for n in RANGE_SPECS],
                "python paper_runs/rebuttal/eval_family.py --family ranges",
                1, "16:00:00"))

    out.append(("text_cond", [RESULTS_ROOT / "text_cond" / s / "seed_0"
                              for s in ("size", "count")],
                "python paper_runs/rebuttal/eval_family.py --family text_cond",
                1, "12:00:00"))

    out.append(("realistic", [RESULTS_ROOT / "realistic" / s / "seed_0"
                              for s in ("size", "position", "rotation", "count")],
                "python paper_runs/rebuttal/eval_realistic.py",
                1, "08:00:00"))
    return out


def submit(name, command, gpus, time_limit, dry_run):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    script = (
        f"cd {REPO_ROOT} && {DIT_ENV} && export MPLCONFIGDIR=/tmp/mpl_$SLURM_JOB_ID "
        f"&& mkdir -p $MPLCONFIGDIR && {command}"
    )
    cmd = ["sbatch", f"--job-name=rbev-{name}", "--nodes=1", "--ntasks=1",
           "--cpus-per-task=8", "--mem=48G", f"--gres=gpu:{gpus}",
           f"--time={time_limit}",
           f"--output={LOG_ROOT}/eval-{name}-%j.out",
           f"--error={LOG_ROOT}/eval-{name}-%j.err",
           f"--wrap={script}"]
    if dry_run:
        print(f"[dry-run] would submit {name}")
        return None
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def table2_seed_evals(dry_run):
    """Seeds 3-9 use the paper-locked submitter, one job per (skill, seed)."""
    submitted = []
    for skill in ("size", "position", "count"):
        ready = [s for s in range(3, 10)
                 if has_final(TABLE2_RESULTS / skill / "baseline" / f"seed_{s}")]
        pending = []
        for seed in ready:
            stamp = STAMP_DIR / f"table2-{skill}-seed{seed}"
            if stamp.exists():
                continue
            pending.append(seed)
        if not pending:
            continue
        if dry_run:
            print(f"[dry-run] would submit table2 eval {skill} seeds {pending}")
            continue
        seed_args = [str(s) for s in pending]
        commands = [[sys.executable, "paper_runs/table2/submit_eval.py",
                     "--skills", skill, "--variants", "baseline",
                     "--seeds", *seed_args, "--submit"]]
        # The paper's size and position extrapolation columns come from the
        # extended grids, so the extra seeds need that run too -- otherwise the
        # n=10 numbers would sit on a different definition than Table 2.
        if skill in ("size", "position"):
            commands.append([sys.executable, "paper_runs/table2/submit_eval_extended.py",
                             "--skills", skill, "--variants", "baseline",
                             "--seeds", *seed_args, "--submit"])
        results = [subprocess.run(c, cwd=REPO_ROOT, capture_output=True, text=True)
                   for c in commands]
        res = results[0]
        for extra in results[1:]:
            if extra.returncode != 0:
                print(f"[warn] extended eval submit failed for {skill}: {extra.stderr[-300:]}")
        if res.returncode == 0:
            for seed in pending:
                (STAMP_DIR / f"table2-{skill}-seed{seed}").write_text("submitted")
            submitted.append(f"{skill} seeds {pending}")
        else:
            print(f"[warn] table2 eval submit failed for {skill}: {res.stderr[-300:]}")
    return submitted


def main():
    parser = argparse.ArgumentParser(description="Submit evals for finished families.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    any_action = False

    for name, dirs, command, gpus, time_limit in families():
        stamp = STAMP_DIR / name
        done = [d for d in dirs if has_final(d)]
        if stamp.exists():
            continue
        if len(done) < len(dirs):
            continue
        result = submit(name, command, gpus, time_limit, args.dry_run)
        if result:
            stamp.write_text(result)
            print(f"submitted {name}: {result}")
            any_action = True

    for line in table2_seed_evals(args.dry_run):
        print(f"submitted table2 {line}")
        any_action = True

    if not any_action:
        status = {name: f"{sum(has_final(d) for d in dirs)}/{len(dirs)}"
                  for name, dirs, _, _, _ in families()}
        t2_done = {s: sum(has_final(TABLE2_RESULTS / s / "baseline" / f"seed_{i}")
                          for i in range(3, 10)) for s in ("size", "position", "count")}
        print("nothing ready yet | " + ", ".join(f"{k} {v}" for k, v in status.items())
              + " | table2 seeds " + ", ".join(f"{k} {v}/7" for k, v in t2_done.items()))


if __name__ == "__main__":
    main()
