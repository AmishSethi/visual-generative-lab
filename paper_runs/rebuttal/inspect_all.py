#!/usr/bin/env python
"""Generate and visually inspect samples from every finished rebuttal family.

Rule-based metrics can agree with each other and still be wrong about what the
model produced -- we have already been burned twice in this project, once by a
counter that could only read 29-52% of its own ground truth, and once by an
interp split that banker's-rounded its queries onto trained values.  So look at
the images.

For each family this samples a spread of conditions, scores each generation with
the paper's own extractor, and writes a labelled contact sheet with green/red
borders so a wrong number is visible rather than inferred.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REB = Path(f"{VGL_ROOT}/REBUTTAL")
T2 = Path(f"{VGL_ROOT}/MORE_SEEDS/table2")
CELL = 84

# family -> (results subdir, dataset dir, skill, split to sample)
FAMILIES = {
    "realistic-size":     ("realistic/size",      REB/"datasets/realistic/size",   "size",     "extra"),
    "realistic-rotation": ("realistic/rotation",  REB/"datasets/realistic/rotation","rotation", "extra"),
    "realistic-count":    ("realistic/count",     REB/"datasets/realistic/count",  "count",    "train"),
    "hires128-size":      ("hires128/size",       REB/"datasets/hires128/size",    "size",     "extra"),
    "hires128-rotation":  ("hires128/rotation",   REB/"datasets/hires128/rotation","rotation", "extra"),
    "textcond-size":      ("text_cond/size",      T2/"datasets/size",              "size",     "interp"),
    "textcond-rotation":  ("text_cond/rotation",  T2/"datasets/rotation",          "rotation", "extra"),
    "countvar-grid_r4":   ("count_variants/grid_r4", REB/"datasets/count_variants/grid_r4", "count", "train"),
}


def score(skill, image, value, realistic):
    """Use the same extractor the family's own evaluator uses."""
    import paper_runs.table2.evaluate_table2 as t2
    if realistic:
        from paper_runs.rebuttal import eval_realistic as er
        if skill == "size":
            return er.score_size(image, value)["iou"] >= er.SIZE_IOU_THRESHOLD, ""
        if skill == "rotation":
            r = er.score_rotation(image, value)
            return r["error"] <= er.ROTATION_ERROR_THRESHOLD, f"{r['error']:.0f}d"
        got = er.score_count(image)
        return got == int(round(value)), f"got {got}"
    if skill == "size":
        gt = t2.gt_circle_image(value, image.shape[0])
        m = t2.calculate_circle_metrics(image, gt, value)
        return m["iou"] >= t2.SIZE_IOU_THRESHOLD, f"iou {m['iou']:.2f}"
    if skill == "rotation":
        # The rotation detector builds its templates at a fixed arrow size, so a
        # 128x128 image must be downscaled first -- exactly what scripts/eval_hires.py
        # does.  Skipping this flips head-for-tail and yields 180-degree errors.
        if image.shape[0] != 64:
            image = np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.LANCZOS))
        m = t2.calculate_rotation_metrics_robust(image, value)
        e = m.get("angle_error_abs")
        return (e is not None and e <= t2.ROTATION_ERROR_THRESHOLD), (f"{e:.0f}d" if e is not None else "none")
    got = t2.detect_count_with_watershed(image, min_area=t2.COUNT_MIN_AREA)
    return got == int(round(value)), f"got {got}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--per-cond", type=int, default=3)
    ap.add_argument("--max-cond", type=int, default=6)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out-dir", type=Path, default=REB/"eval"/"visual_inspection")
    a = ap.parse_args()

    import paper_runs.table2.evaluate_table2 as t2
    from paper_runs.rebuttal import subsample
    subsample.install(t2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for name in a.families:
        sub, dsdir, skill, split = FAMILIES[name]
        realistic = sub.startswith("realistic")
        try:
            meta = json.loads((dsdir/"dataset_metadata.json").read_text())
        except FileNotFoundError:
            print(f"[skip] {name}: no dataset metadata", flush=True); continue
        t2.load_dataset_metadata = lambda s, _m=meta: _m
        t2.RESULTS_ROOT = REB/"results"/Path(sub).parent
        if realistic:
            t2.DATASET_ROOT = REB/"datasets/realistic"

        try:
            sel = t2.latest_finished_run(Path(sub).name, "", 0)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}", flush=True); continue

        builder = {"size": t2.build_size_queries, "rotation": t2.build_rotation_queries,
                   "count": t2.build_count_queries}[skill]
        values = builder(meta)[split]
        if len(values) > a.max_cond:
            idx = np.linspace(0, len(values)-1, a.max_cond).astype(int)
            values = [values[i] for i in idx]

        cfgb = {"size": t2.build_size_config, "rotation": t2.build_rotation_config,
                "count": t2.build_count_config}[skill]
        config = cfgb(sel, meta) if skill == "rotation" else cfgb(sel)
        if skill == "rotation":
            from vgl.eval_rotation import generate_samples as gen, load_model as load
        else:
            from vgl.eval_radius import generate_samples as gen, load_model as load
        model = load(config, device)
        sampler = t2.create_sampler(config, a.steps)
        samples = gen(model, sampler, values, config, device, num_samples=a.per_cond,
                      cfg_scale=1.0, batch_size=a.per_cond, num_sampling_steps=a.steps)
        del model; torch.cuda.empty_cache()

        rows, cols = len(values), a.per_cond
        sheet = Image.new("RGB", (CELL*cols, (CELL+16)*rows + 20), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), f"{name}  [{split}]", fill="black")
        hits = 0
        for r, v in enumerate(values):
            for c, s in enumerate(samples[v]):
                img = np.asarray(s, dtype=np.uint8)
                ok, note = score(skill, img, v, realistic)
                hits += ok
                x, y = c*CELL, 20 + r*(CELL+16)
                sheet.paste(Image.fromarray(img).resize((CELL, CELL), Image.NEAREST), (x, y))
                col = (0,150,0) if ok else (200,0,0)
                draw.rectangle([x, y, x+CELL-1, y+CELL-1], outline=col, width=3)
                draw.text((x+3, y+CELL+2), f"{v:g} {note}"[:16], fill=col)
        n = len(values)*a.per_cond
        sheet.save(a.out_dir/f"{name}.png")
        report[name] = {"split": split, "hits": int(hits), "n": int(n), "acc": float(100.0*hits/n),
                        "values": [float(v) for v in values]}
        print(f"{name:22s} [{split:6s}] {hits}/{n} = {100.0*hits/n:5.1f}%", flush=True)

    (a.out_dir/"report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote sheets to {a.out_dir}")


if __name__ == "__main__":
    main()
