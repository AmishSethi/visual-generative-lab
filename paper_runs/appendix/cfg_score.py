#!/usr/bin/env python
"""Score the guidance sweep of cfg_sample.py with the Table 2 size metric.

Every sample's mask (vgl.eval_radius) is compared with the Table 2 ground-truth disk of the requested
radius (evaluate_table2.gt_circle_image). Writes <out>/scale_<s>.json with the IoU, the mask radius and
the two mask areas of every sample; cfg_aggregate.py counts a sample as correct at IoU >= 0.90.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.eval_radius import calculate_circle_metrics
from paper_runs.appendix.manifest import EVAL_ROOT
from paper_runs.table2.evaluate_table2 import gt_circle_image


def score_sample(image, gt, radius):
    m = calculate_circle_metrics(image, gt, radius)
    return {"iou": float(m["iou"]), "r_est": float(m["estimated_radius"]),
            "gt_px": int(m["gt_mask"].sum()), "gen_px": int(m["gen_mask"].sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=Path, default=EVAL_ROOT / "cfg" / "samples", help="cfg_sample.py --out")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "cfg" / "scores", help="directory for scale_<s>.json")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for d in sorted(p.parent for p in args.samples.glob("scale_*/manifest.json")):
        conditions = []
        for stem in json.loads((d / "manifest.json").read_text())["files"]:
            rec = json.loads((d / f"{stem}.json").read_text())
            images = np.load(d / f"{stem}.npz")["images"]
            assert images.shape[0] == rec["num_samples"], (d, stem)
            size = int(images.shape[1])
            gt = gt_circle_image(rec["value"], size)
            conditions.append({**{k: rec[k] for k in ("index", "split", "value", "key", "cfg_scale")},
                               "image_size": size,
                               "per_sample": [score_sample(x, gt, rec["value"]) for x in images]})
        path = args.out / f"{d.name}.json"
        path.write_text(json.dumps({"samples": str(d), "conditions": conditions}))
        print(f"scored {d.name}: {len(conditions)} conditions -> {path}", flush=True)


if __name__ == "__main__":
    main()
