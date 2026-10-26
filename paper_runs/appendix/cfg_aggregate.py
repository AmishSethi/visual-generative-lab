#!/usr/bin/env python
"""Figure 6 numbers from the scores of cfg_score.py.

Per guidance scale: accuracy (IoU >= 0.90, the Table 2 size threshold) and mean IoU over all samples of
the interpolation radii and of the extrapolation radii, and the accuracy of each extrapolation radius.
Writes results.json and prints the tables.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.manifest import EVAL_ROOT

IOU_THRESHOLD = 0.90  # evaluate_table2.SIZE_IOU_THRESHOLD


def accuracy(ious):
    return 100.0 * sum(i >= IOU_THRESHOLD for i in ious) / len(ious)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", type=Path, default=EVAL_ROOT / "cfg" / "scores")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "cfg" / "results.json")
    args = ap.parse_args()

    files = sorted(args.scores.glob("scale_*.json"), key=lambda p: float(p.stem.split("_", 1)[1]))
    results = {"iou_threshold": IOU_THRESHOLD, "scales": {}}
    for path in files:
        conditions = json.loads(path.read_text())["conditions"]
        row = {}
        for split in ("interp", "extra"):
            ious = [x["iou"] for c in conditions if c["split"] == split for x in c["per_sample"]]
            row[split] = {"accuracy": accuracy(ious), "mean_iou": st.mean(ious), "num_samples": len(ious)}
        row["extra_accuracy_per_radius"] = {f"{c['value']:g}": accuracy([x["iou"] for x in c["per_sample"]])
                                            for c in conditions if c["split"] == "extra"}
        results["scales"][path.stem.split("_", 1)[1]] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print("| scale | interp acc | interp IoU | extra acc | extra IoU |")
    print("|---|---|---|---|---|")
    for s, row in results["scales"].items():
        i, e = row["interp"], row["extra"]
        print(f"| {s} | {i['accuracy']:.1f} | {i['mean_iou']:.3f} | {e['accuracy']:.1f} | {e['mean_iou']:.3f} |")
    radii = list(next(iter(results["scales"].values()))["extra_accuracy_per_radius"])
    print("\nExtrapolation accuracy per radius:\n")
    print("| scale | " + " | ".join(radii) + " |")
    print("|---|" + "---|" * len(radii))
    for s, row in results["scales"].items():
        print(f"| {s} | " + " | ".join(f"{row['extra_accuracy_per_radius'][r]:.0f}" for r in radii) + " |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
