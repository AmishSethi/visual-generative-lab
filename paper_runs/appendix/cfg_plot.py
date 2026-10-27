#!/usr/bin/env python
"""Figure 6: size accuracy on the Table 2 interpolation and extrapolation radii against the guidance scale.

Reads the scores of cfg_score.py, checks them against results.json of cfg_aggregate.py, and writes
cfg_scale_size_ablation.pdf and .png, with the plotted numbers in fig6_data.json. Text is set in Arial
when the Arial font files are in --font-dir, and in matplotlib's default sans-serif otherwise.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.manifest import EVAL_ROOT

SCALES = [1, 2, 4, 10, 15, 20, 50, 100]
INTERP = [r + 0.5 for r in range(5, 20)]                     # Table 2 midpoints
EXTRA = [1, 2, 3, 4] + list(range(21, 31))                   # Table 2 extrapolation radii
ARIAL = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")

PAGE_W, PAGE_H = 476.84875, 287.7325       # pt
PRINT_W = 0.50 * 397.48                     # printed width in the paper, 0.50\linewidth
SCALE = PRINT_W / PAGE_W
TEXT_PT = 8.0 / SCALE                       # native size that prints at 8 pt
LEG_PT = 7.0 / SCALE                        # native size that prints at 7 pt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", type=Path, default=EVAL_ROOT / "cfg" / "scores")
    ap.add_argument("--results", type=Path, default=EVAL_ROOT / "cfg" / "results.json")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "cfg")
    ap.add_argument("--font-dir", type=Path, default=Path.home() / ".fonts")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- data
    ref = json.loads(args.results.read_text())["scales"]
    rows = []
    for s in SCALES:
        d = json.loads((args.scores / f"scale_{s}.json").read_text())
        by = {(c["split"], float(c["value"])): c["per_sample"] for c in d["conditions"]}
        res = {}
        for split, radii in (("interp", INTERP), ("extra", EXTRA)):
            ious = []
            per_r = {}
            for r in radii:
                ps = by[(split, float(r))]
                assert len(ps) == 20, (s, split, r, len(ps))
                v = [x["iou"] for x in ps]
                ious += v
                per_r[f"{r:g}"] = 100.0 * sum(i >= 0.90 for i in v) / len(v)
            res[split] = {"acc": 100.0 * sum(i >= 0.90 for i in ious) / len(ious), "mean_iou": st.mean(ious),
                          "n": len(ious), "per_radius_acc": per_r}
        rf = ref[str(s)]
        for split in ("interp", "extra"):
            assert abs(res[split]["acc"] - rf[split]["accuracy"]) < 1e-9, (s, split)
            assert abs(res[split]["mean_iou"] - rf[split]["mean_iou"]) < 1e-9, (s, split)
        for r, a in res["extra"]["per_radius_acc"].items():
            assert abs(a - rf["extra_accuracy_per_radius"][r]) < 1e-9, (s, r)
        rows.append({"cfg_scale": s, **res})
    interp_acc = [r["interp"]["acc"] for r in rows]
    extra_acc = [r["extra"]["acc"] for r in rows]

    # ---- plot
    faces = [args.font_dir / f for f in ARIAL if (args.font_dir / f).exists()]
    for face in faces:
        font_manager.fontManager.addfont(str(face))
    rc = {"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
          "font.size": TEXT_PT, "axes.labelsize": TEXT_PT, "xtick.labelsize": TEXT_PT, "ytick.labelsize": TEXT_PT,
          "legend.fontsize": LEG_PT}
    if faces:
        rc["font.sans-serif"] = ["Arial"]
    plt.rcParams.update(rc)
    fig, ax = plt.subplots(figsize=(PAGE_W / 72, PAGE_H / 72))
    ax.plot(SCALES, interp_acc, marker="o", linewidth=2, markersize=7, color="tab:blue", label="Interpolation", clip_on=False)
    ax.plot(SCALES, extra_acc, marker="D", linewidth=2, markersize=7, color="tab:red", label="Extrapolation", clip_on=False)
    ax.set_xlabel("CFG Scale")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    TICKS = [s for s in SCALES if s != 15]       # a "15" label would run into "20"
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{s:g}" for s in TICKS])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v:g}" for v in (0, 20, 40, 60, 80, 100)])
    ax.grid(True, alpha=0.3)

    # ---- legend: one row above the axes, outside the plot area
    LEG_KW = dict(ncol=2, frameon=False, borderaxespad=0.0, borderpad=0.0, handlelength=1.6, handletextpad=0.4,
                  columnspacing=1.2)
    TOP = 1 - 0.3 * TEXT_PT / PAGE_H                                          # same edge pad as tight_layout(pad=0.3)
    leg = fig.legend(loc="upper center", bbox_to_anchor=(0.5, TOP), **LEG_KW)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    leg_bottom = leg.get_window_extent(r).y0 / fig.dpi / (PAGE_H / 72)       # figure fraction
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, leg_bottom - 0.015))
    fig.canvas.draw()
    ab = ax.get_window_extent(r)
    xc = (ab.x0 + ab.x1) / 2 / fig.dpi / (PAGE_W / 72)                       # centre the legend over the plot frame
    leg.remove()
    leg = fig.legend(loc="upper center", bbox_to_anchor=(xc, TOP), **LEG_KW)
    fig.canvas.draw()
    lb = leg.get_window_extent(r)
    atb = ax.get_tightbbox(r)
    tb = fig.get_tightbbox(r)
    assert tb.x0 >= 0 and tb.y0 >= 0 and tb.x1 * 72 <= PAGE_W + 1e-6 and tb.y1 * 72 <= PAGE_H + 1e-6
    assert lb.y0 > atb.y1, "legend overlaps the axes or its tick labels"
    fig.savefig(out / "cfg_scale_size_ablation.pdf")
    fig.savefig(out / "cfg_scale_size_ablation.png", dpi=150)
    json.dump({"source": str(args.scores / "scale_*.json"),
               "model": "Table 2 size baseline config (DiT-S/2, Table 2 size dataset, batch 128, 1000 epochs) with "
                        "--radius-dropout-prob 0.1, --null-embedding-type none, --null-radius 0.0, seed 0; final_0078000.pt, EMA",
               "evaluator": "size extractor (Table 2)",
               "metric": "accuracy % (IoU >= 0.90); mean_iou also given", "samples_per_condition": 20,
               "interp_radii": INTERP, "extra_radii": EXTRA, "rows": rows},
              open(out / "fig6_data.json", "w"), indent=1)
    for row in rows:
        pr = row["extra"]["per_radius_acc"]
        print("scale %5g  interp acc %5.1f IoU %.3f | extra acc %5.1f IoU %.3f | radii >= 50%%: %s" % (
            row["cfg_scale"], row["interp"]["acc"], row["interp"]["mean_iou"], row["extra"]["acc"], row["extra"]["mean_iou"],
            [k for k, v in pr.items() if v >= 50]))
    print("wrote", out)


if __name__ == "__main__":
    main()
