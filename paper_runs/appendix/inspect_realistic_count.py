#!/usr/bin/env python
"""Visually inspect realistic-render count generations.

The realistic count model scores 55.8% exact-match on its own training values.
That number is only meaningful if the failures are real -- the model drawing the
wrong number of objects -- rather than the counter miscounting correct images.

For each generation this dumps the image alongside three independent counts:

    watershed   the metric eval_realistic.score_count actually uses
    components  connected components of the saturation mask; merges touching
                objects, so watershed > components means objects are in contact
    area        total foreground area divided by the area of one radius-8 disc

Unanimous agreement on a wrong number means the model drew the wrong number.
Disagreement means the image is ambiguous and the metric is doing the work.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paper_runs.table2.evaluate_table2 as t2
from paper_runs.appendix.eval_realistic import MIN_BLOB_AREA, score_count
from paper_runs.appendix.manifest import DATASET_ROOT, EVAL_ROOT, RESULTS_ROOT
from paper_runs.appendix.realistic import object_mask
from vgl.eval_radius import generate_samples, load_model

CELL = 128           # display size per sample
DISC_AREA = np.pi * 8 ** 2   # renderer uses radius 8


def count_components(image):
    mask = (object_mask(image) * 255).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA)


def count_area(image):
    mask = object_mask(image)
    return int(round(float(mask.sum()) / DISC_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-count", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out-dir", type=Path, default=EVAL_ROOT / "realistic_count_inspect")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t2.DATASET_ROOT = DATASET_ROOT / "realistic"
    t2.RESULTS_ROOT = RESULTS_ROOT / "realistic"
    metadata = t2.load_dataset_metadata("count")
    selection = t2.latest_finished_run("count", "", a.seed)
    config = t2.build_count_config(selection)
    values = t2.build_count_queries(metadata)["train"]
    print(f"checkpoint: {selection.checkpoint}", flush=True)
    print(f"train values: {values}", flush=True)

    model = load_model(config, device)
    sampler = t2.create_sampler(config, a.steps)
    samples = generate_samples(model, sampler, values, config, device,
                               num_samples=a.per_count, cfg_scale=1.0,
                               batch_size=a.per_count, num_sampling_steps=a.steps)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (CELL * a.per_count, (CELL + 22) * len(values)), "white")
    draw = ImageDraw.Draw(sheet)
    records = []

    for row, value in enumerate(values):
        want = int(round(value))
        for col, sample in enumerate(samples[value]):
            image = np.asarray(sample, dtype=np.uint8)
            ws, cc, ar = score_count(image), count_components(image), count_area(image)
            agree = ws == cc == ar
            records.append({"want": want, "watershed": ws, "components": cc, "area": ar,
                            "metric_correct": ws == want, "counters_agree": agree})

            x, y = col * CELL, row * (CELL + 22)
            sheet.paste(Image.fromarray(image).resize((CELL, CELL), Image.NEAREST), (x, y))
            # green = metric says correct, red = metric says wrong
            colour = (0, 160, 0) if ws == want else (200, 0, 0)
            draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=colour, width=3)
            draw.text((x + 4, y + CELL + 4),
                      f"want {want}  ws{ws} cc{cc} ar{ar}" + ("" if agree else "  *"),
                      fill=colour)

    sheet_path = a.out_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    # Where does the metric disagree with the other two counters?
    wrong = [r for r in records if not r["metric_correct"]]
    unanimous_wrong = [r for r in wrong if r["counters_agree"]]
    print(f"\nsamples                     {len(records)}")
    print(f"metric says correct         {len(records) - len(wrong)} "
          f"({100.0 * (len(records) - len(wrong)) / len(records):.1f}%)")
    print(f"metric says wrong           {len(wrong)}")
    print(f"  all three counters agree  {len(unanimous_wrong)}  <- genuine model errors")
    print(f"  counters disagree         {len(wrong) - len(unanimous_wrong)}  <- ambiguous images")

    print("\nper requested count:")
    for want in sorted({r['want'] for r in records}):
        rows = [r for r in records if r["want"] == want]
        ok = sum(r["metric_correct"] for r in rows)
        drawn = [r["watershed"] for r in rows]
        print(f"  want {want}: metric {ok}/{len(rows)} correct, watershed counts {drawn}")

    (a.out_dir / "records.json").write_text(json.dumps(records, indent=1))
    print(f"\nwrote {sheet_path}")


if __name__ == "__main__":
    main()
