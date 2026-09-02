#!/usr/bin/env python
"""Side-by-side generations from two count runs with byte-identical configs.

count_variants/grid_r4/seed_0 scored 85.0% train and
count_column/grid_r4/baseline/seed_0 scored 63.3%, on the same data, same seed,
same hyperparameters -- a diff of their run_config.json shows no difference at
all.  Evaluation sampling is seeded, so that is not the source.

This dumps the same conditions from both checkpoints with the same sampling
seed, so the difference can be seen rather than inferred.
"""
import argparse
import json
from pathlib import Path
import sys

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
CELL = 96

RUNS = {
    "variant_85": REB / "results/count_variants/grid_r4/seed_0",
    "column_63": REB / "results/count_column/grid_r4/baseline/seed_0",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-count", type=int, default=6)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out-dir", type=Path, default=REB / "eval" / "count_run_compare")
    a = ap.parse_args()

    import paper_runs.table2.evaluate_table2 as t2
    from vgl.eval_radius import generate_samples, load_model

    meta = json.loads((REB / "datasets/count_variants/grid_r4/dataset_metadata.json").read_text())
    t2.load_dataset_metadata = lambda skill, _m=meta: _m
    values = t2.build_count_queries(meta)["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out_dir.mkdir(parents=True, exist_ok=True)

    panels, summary = {}, {}
    for tag, run_root in RUNS.items():
        run_dir = sorted(d for d in run_root.iterdir() if d.is_dir())[0]
        ckpt = sorted((run_dir / "checkpoints").glob("final_*.pt"))[-1]
        cfg = json.loads((run_dir / "run_config.json").read_text())
        sel = t2.RunSelection(run_dir=run_dir, checkpoint=ckpt, run_config=cfg)
        config = t2.build_count_config(sel)

        model = load_model(config, device)
        sampler = t2.create_sampler(config, a.steps)
        samples = generate_samples(model, sampler, values, config, device,
                                   num_samples=a.per_count, cfg_scale=1.0,
                                   batch_size=a.per_count, num_sampling_steps=a.steps)
        panels[tag] = samples
        del model
        torch.cuda.empty_cache()

        hits = {}
        for v in values:
            want = int(round(v))
            got = [t2.detect_count_with_watershed(np.asarray(s, dtype=np.uint8),
                                                  min_area=t2.COUNT_MIN_AREA)
                   for s in samples[v]]
            hits[want] = (sum(g == want for g in got), len(got), got)
        summary[tag] = hits
        print(f"{tag}: " + "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(hits.items())),
              flush=True)

    # One sheet: each requested count is a row, the two runs side by side.
    rows, cols = len(values), a.per_count * 2 + 1
    sheet = Image.new("RGB", (CELL * cols, (CELL + 18) * rows + 24), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), "LEFT: variant run (scored 85.0)      RIGHT: column run (scored 63.3)", fill="black")
    for r, v in enumerate(values):
        want = int(round(v))
        y = 24 + r * (CELL + 18)
        for tag_i, tag in enumerate(RUNS):
            got = summary[tag][want][2]
            for c, img in enumerate(panels[tag][v]):
                x = (tag_i * (a.per_count + 1) + c) * CELL
                sheet.paste(Image.fromarray(np.asarray(img, dtype=np.uint8))
                            .resize((CELL, CELL), Image.NEAREST), (x, y))
                ok = got[c] == want
                draw.rectangle([x, y, x + CELL - 1, y + CELL - 1],
                               outline=(0, 160, 0) if ok else (200, 0, 0), width=3)
                draw.text((x + 4, y + CELL + 3), f"want {want} got {got[c]}",
                          fill=(0, 160, 0) if ok else (200, 0, 0))

    out = a.out_dir / "compare.png"
    sheet.save(out)
    (a.out_dir / "summary.json").write_text(json.dumps(
        {k: {str(c): {"hits": h[0], "n": h[1], "counts": h[2]} for c, h in v.items()}
         for k, v in summary.items()}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
