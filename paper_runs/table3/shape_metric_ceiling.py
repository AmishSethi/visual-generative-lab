#!/usr/bin/env python
"""Ground-truth ceiling of the shape metrics on each compositional dataset.

Usage: python -m paper_runs.table3.shape_metric_ceiling DATASET_ROOT [--per N] [--out JSON]
Reports, per dataset, the fraction of ground-truth renders each metric reads correctly.
Any model accuracy below a metric's ceiling is bounded by the metric, not the model.
"""
import argparse, collections, glob, json, os, pathlib, re, sys
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--per", type=int, default=6, help="images per class folder")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()
    from vgl.eval_compositional import evaluate_properties_comprehensive
    from vgl.shape_metric_v2 import classify_shape, NAMES
    S2I = {n: i for i, n in enumerate(NAMES)}
    report = {}
    for d in sorted(p for p in a.root.iterdir() if p.is_dir() and (p / "train").is_dir()):
        if a.datasets and d.name not in a.datasets: continue
        if not any(n in d.name for n in ("shape",)): continue
        tot = collections.Counter(); hit_v2 = collections.Counter(); hit_locked = collections.Counter()
        for cls in sorted(glob.glob(f"{d}/train/*")):
            m = re.search(r"(circle|square|triangle|diamond)", cls)
            if not m: continue
            for f in sorted(glob.glob(f"{cls}/*.png"))[: a.per]:
                img = np.array(Image.open(f).convert("RGB")); tot[m.group(1)] += 1
                hit_v2[m.group(1)] += classify_shape(img) == m.group(1)
                os.environ["VGL_SHAPE_METRIC"] = "locked"
                hit_locked[m.group(1)] += evaluate_properties_comprehensive(img, [S2I[m.group(1)]], ["shape"]).get("shape_accuracy", 0) == 1.0
        T = sum(tot.values())
        report[d.name] = {"n": T, "v2": 100 * sum(hit_v2.values()) / T, "locked": 100 * sum(hit_locked.values()) / T,
                          "per_shape_v2": {k: f"{hit_v2[k]}/{tot[k]}" for k in NAMES}}
        print(f"  {d.name:26s} n={T:3d}  v2 {report[d.name]['v2']:5.1f}%   locked {report[d.name]['locked']:5.1f}%")
    if a.out: a.out.write_text(json.dumps(report, indent=1)); print("wrote", a.out)

if __name__ == "__main__":
    main()
