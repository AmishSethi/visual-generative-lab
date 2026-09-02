#!/usr/bin/env python
"""Quantify how much pretrained T2I models actually use the skill query.

Raw accuracy alone cannot distinguish "the model controls the skill badly" from
"the model ignores the request and always draws the same thing".  Two statistics
separate them, both computed from the same generations:

  label-shuffled null   re-score every image against a randomly permuted
                        condition label.  This is the accuracy a model would get
                        by ignoring the prompt entirely and emitting its own
                        marginal output distribution.  Accuracy at or below this
                        null means no measurable control.

  request-response slope  for size, regress the measured diameter fraction on
                        the requested one.  Slope ~1 is faithful control, slope
                        ~0 is none.

Both are the external-validity evidence the AC asked for: they show the failures
VGL isolates under controlled conditions are the same failures production
text-to-image systems exhibit.
"""

import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Run as `python paper_runs/rebuttal/t2i_analyze.py` and sys.path[0] is this
# directory, not the repo root, so the absolute imports below fail.  The sibling
# eval_* scripts all carry this guard; without it the module form is the only
# one that works, which is easy to trip over from a job script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.t2i_prompts import POSITION_CELLS, ROTATION_DIRECTIONS, SIZE_LEVELS
from paper_runs.rebuttal.t2i_score import (


    ROTATION_TOLERANCE_DEG, SIZE_REL_TOLERANCE, angle_error, dirname_to_condition,
    protocol_compliant, score_count, score_position, score_rotation, score_size,
)

N_SHUFFLES = 500


def collect(model_dir, skill):
    """Per-image (requested condition, measured value) pairs."""
    records = []
    skill_dir = model_dir / skill
    if not skill_dir.exists():
        return records
    for cond_dir in sorted(p for p in skill_dir.iterdir() if p.is_dir()):
        condition = dirname_to_condition(skill, cond_dir.name)
        for path in sorted(cond_dir.glob("sample_*.png")):
            image = np.array(Image.open(path).convert("RGB"))
            if not protocol_compliant(image)[0]:
                continue
            if skill == "count":
                measured = score_count(image)
            elif skill == "position":
                measured = score_position(image)
            elif skill == "size":
                measured = score_size(image)
            else:
                measured = score_rotation(image)
            if measured is None:
                continue
            records.append((condition, measured))
    return records


def is_correct(skill, condition, measured):
    if skill == "count":
        return measured == int(condition)
    if skill == "position":
        return measured == dict(POSITION_CELLS)[condition]
    if skill == "size":
        target = dict(SIZE_LEVELS)[condition]
        return abs(measured - target) / target <= SIZE_REL_TOLERANCE
    return angle_error(measured, dict(ROTATION_DIRECTIONS)[condition]) <= ROTATION_TOLERANCE_DEG


def shuffled_null(skill, records, rng):
    """Accuracy from pairing each measurement with a random requested label."""
    conditions = [c for c, _ in records]
    measures = [m for _, m in records]
    scores = []
    for _ in range(N_SHUFFLES):
        permuted = rng.permutation(len(conditions))
        scores.append(np.mean([
            is_correct(skill, conditions[i], measures[j])
            for i, j in zip(range(len(conditions)), permuted)
        ]))
    return 100.0 * float(np.mean(scores)), 100.0 * float(np.percentile(scores, 95))


def size_slope(records):
    """OLS slope of measured diameter fraction on requested fraction."""
    x = np.array([dict(SIZE_LEVELS)[c] for c, _ in records], dtype=float)
    y = np.array([m for _, m in records], dtype=float)
    if len(x) < 3:
        return None
    A = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {"slope": float(beta[1]), "intercept": float(beta[0]), "pearson_r": corr}


def main():
    parser = argparse.ArgumentParser(description="Analyse T2I skill control.")
    parser.add_argument("--images-root", type=Path,
                        default=Path(f"{VGL_ROOT}/REBUTTAL/eval/t2i/images"))
    parser.add_argument("--out", type=Path,
                        default=Path(f"{VGL_ROOT}/REBUTTAL/eval/t2i/analysis.json"))
    parser.add_argument("--skills", nargs="+", default=["count", "position", "size", "rotation"])
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    payload = {}
    print(f"{'model':>8} {'skill':>9} {'accuracy':>9} {'null':>8} {'null p95':>9} {'verdict':>22}")
    for model_dir in sorted(p for p in args.images_root.iterdir() if p.is_dir()):
        payload[model_dir.name] = {}
        for skill in args.skills:
            records = collect(model_dir, skill)
            if not records:
                continue
            accuracy = 100.0 * float(np.mean([is_correct(skill, c, m) for c, m in records]))
            null_mean, null_p95 = shuffled_null(skill, records, rng)
            verdict = ("indistinguishable from null" if accuracy <= null_p95
                       else "above null")
            entry = {"n": len(records), "accuracy": accuracy,
                     "shuffled_null_mean": null_mean, "shuffled_null_p95": null_p95,
                     "above_null": accuracy > null_p95}
            if skill == "size":
                entry["request_response"] = size_slope(records)
            payload[model_dir.name][skill] = entry
            print(f"{model_dir.name:>8} {skill:>9} {accuracy:8.1f}% {null_mean:7.1f}% "
                  f"{null_p95:8.1f}% {verdict:>22}")

    print("\nsize request-response (slope 1 = faithful control, 0 = ignores request):")
    for model, skills in payload.items():
        rr = skills.get("size", {}).get("request_response")
        if rr:
            print(f"  {model:>8}: measured = {rr['intercept']:.3f} + {rr['slope']:.3f} * requested"
                  f"   (r = {rr['pearson_r']:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
