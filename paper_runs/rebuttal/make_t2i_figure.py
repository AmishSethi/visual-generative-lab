#!/usr/bin/env python
"""Qualitative figure: pretrained T2I models ignore the size request.

One row per model, one column per requested diameter.  Each panel is the median
sample by measured value, so the figure shows typical behaviour rather than a
best or arbitrary draw.  The measured diameter is printed under each panel so
the figure carries the number, not just the impression.  This is the visual companion to the request-response slopes
(0.04, 0.07, -0.01 against 1.0 for faithful control).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.t2i_score import score_count, score_size

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")


MODELS = [("sd15", "SD 1.5"), ("sdxl", "SDXL"), ("pixart", "PixArt-alpha")]
SIZE_COLUMNS = ["10pct", "30pct", "50pct", "70pct", "90pct"]
COUNT_COLUMNS = ["1", "2", "3", "5", "7"]
TILE = 150
LABEL_H = 26
HEADER_H = 30
ROW_LABEL_W = 108


def panel(images_root, model, skill, columns, scorer, fmt):
    rows = []
    for column in columns:
        cond_dir = images_root / model / skill / column
        paths = sorted(cond_dir.glob("sample_*.png"))
        if not paths:
            rows.append((None, ""))
            continue
        # The median sample by measured value: representative rather than either
        # cherry-picked or an arbitrary first draw that may be an outlier.
        scored = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            value = scorer(np.array(image))
            if value is not None:
                scored.append((value, path))
        if not scored:
            rows.append((None, "-"))
            continue
        scored.sort(key=lambda t: t[0])
        measured, chosen = scored[len(scored) // 2]
        image = Image.open(chosen).convert("RGB")
        rows.append((image.resize((TILE, TILE)), fmt(measured)))
    return rows


def build(images_root, out_path, skill):
    if skill == "size":
        columns, scorer = SIZE_COLUMNS, score_size
        header = [f"asked: {c.replace('pct', '%')}" for c in columns]
        fmt = lambda v: f"got {v:.2f}" if v is not None else "-"
    else:
        columns, scorer = COUNT_COLUMNS, score_count
        header = [f"asked: {c}" for c in columns]
        fmt = lambda v: f"got {v}" if v is not None else "-"

    width = ROW_LABEL_W + TILE * len(columns)
    height = HEADER_H + (TILE + LABEL_H) * len(MODELS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, text in enumerate(header):
        draw.text((ROW_LABEL_W + i * TILE + 8, 9), text, fill=(0, 0, 0))

    for r, (model, label) in enumerate(MODELS):
        top = HEADER_H + r * (TILE + LABEL_H)
        draw.text((8, top + TILE // 2), label, fill=(0, 0, 0))
        for c, (image, caption) in enumerate(
                panel(images_root, model, skill, columns, scorer, fmt)):
            x = ROW_LABEL_W + c * TILE
            if image is not None:
                canvas.paste(image, (x, top))
            draw.rectangle([x, top, x + TILE - 1, top + TILE - 1], outline=(200, 200, 200))
            draw.text((x + 8, top + TILE + 6), caption, fill=(0, 0, 0))

    canvas.save(out_path)
    print(f"wrote {out_path}  ({canvas.size[0]}x{canvas.size[1]})")


def main():
    parser = argparse.ArgumentParser(description="Build the T2I qualitative figure.")
    parser.add_argument("--images-root", type=Path,
                        default=Path(f"{VGL_ROOT}/REBUTTAL/eval/t2i/images"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path(f"{VGL_ROOT}/REBUTTAL/figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for skill in ("size", "count"):
        build(args.images_root, args.out_dir / f"t2i_{skill}.png", skill)


if __name__ == "__main__":
    main()
