#!/usr/bin/env python
"""Ground-truth ceiling of the count metric on the shape+count compositional datasets.

Usage: python -m paper_runs.table3.count_metric_ceiling DATASET_ROOT [--per N] [--out JSON]
Runs the locked evaluator's count detector on ground-truth renders of every train and
test_compositional class folder and reports the fraction it counts correctly, per dataset, per
split, and per (shape, count). Any model accuracy below this is bounded by the metric.
"""
import argparse, collections, glob, json, pathlib, re, sys
import numpy as np
from PIL import Image
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--per", type=int, default=6, help="images per class folder")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()
    from vgl.eval_compositional import evaluate_properties_comprehensive
    report = {}
    for d in sorted(p for p in a.root.iterdir() if p.is_dir() and p.name.startswith("shape_count")):
        report[d.name] = {}
        for split in ("train", "test_compositional"):
            tot = collections.Counter(); hit = collections.Counter()
            for cls in sorted(glob.glob(f"{d}/{split}/*")):
                m = re.search(r"(circle|square|triangle|diamond)_cnt(\d+)", cls)
                if not m: continue
                key = f"{m.group(1)}_{m.group(2)}"; n = int(m.group(2))
                for f in sorted(glob.glob(f"{cls}/*.png"))[: a.per]:
                    img = np.array(Image.open(f).convert("RGB")); tot[key] += 1
                    hit[key] += evaluate_properties_comprehensive(img, [n], ["count"]).get("count_accuracy", 0) == 1.0
            T = sum(tot.values())
            report[d.name][split] = {"n": T, "ceiling": 100 * sum(hit.values()) / T if T else None,
                                     "per_class": {k: f"{hit[k]}/{tot[k]}" for k in sorted(tot)}}
            print(f"  {d.name:22s} {split:18s} n={T:3d}  count ceiling {report[d.name][split]['ceiling']:5.1f}%")
            bad = [k for k in tot if hit[k] < tot[k]]
            if bad: print("     misread:", ", ".join(f"{k} {hit[k]}/{tot[k]}" for k in sorted(bad)))
    if a.out: a.out.write_text(json.dumps(report, indent=1)); print("wrote", a.out)

if __name__ == "__main__":
    main()
