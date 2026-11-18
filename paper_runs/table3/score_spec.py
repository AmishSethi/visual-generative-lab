#!/usr/bin/env python
"""Score a sample_spec.py output directory with the per-sample rules of score_table3.py.

Every sample is scored on its own against the values it was conditioned on (score_table3.score_sample);
accuracy is the fraction of samples. For each split the output has the summary (detection rate, the
accuracy of each skill, joint accuracy) and the same accuracies for every combination (the cells of
Figure 5). The splits are those of manifest.json, which sample_spec.py writes once every split is saved.

    python -m paper_runs.table3.score_spec --samples <sample_spec.py --out dir> --out scores.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table3 import score_table3 as st


def score_dir(sample_dir: Path) -> dict:
    result = {"samples": str(sample_dir), "splits": {}}
    for split in json.load(open(sample_dir / "manifest.json"))["splits"]:
        meta = json.load(open(sample_dir / f"{split}.json"))
        props = meta["properties"]
        ctx = st.context(props)
        images = np.load(sample_dir / f"{meta['split']}.npz")["images"]
        records, per_combination = [], []
        for ci, combo in enumerate(meta["combinations"]):
            cond = st.cond_from_combo(combo, props)
            recs = []
            for k in range(images.shape[1]):
                r = st.score_sample(images[ci, k], cond, ctx)
                r["combo"] = ci
                recs.append(r)
            s = st.summarize(recs, props)
            per_combination.append({"combination": combo, "accuracy": s["accuracy"], "joint_accuracy": s["joint_accuracy"]})
            records += recs
        result["splits"][meta["split"]] = {**st.summarize(records, props), "per_combination": per_combination}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=Path, required=True, help="sample_spec.py --out dir (manifest.json, <split>.npz, <split>.json)")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = ap.parse_args()
    res = score_dir(args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    for split, v in res["splits"].items():
        print(f"{split:16s} n={v['num_samples']:5d} " + " ".join(f"{p}={a:5.1f}" for p, a in v["accuracy"].items())
              + f" joint={v['joint_accuracy']:5.1f}")


if __name__ == "__main__":
    main()
