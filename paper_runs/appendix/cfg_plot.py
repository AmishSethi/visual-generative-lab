#!/usr/bin/env python
"""Figure 6: size accuracy on the Table 2 interpolation and extrapolation radii against the guidance scale.

Reads results.json of cfg_aggregate.py and writes cfg_scale_size_ablation.pdf and .png. Text is set in Arial
when the Arial font files are in --font-dir, and in matplotlib's default sans-serif otherwise.
"""
import argparse
import json
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
ARIAL = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")

PAGE_W, PAGE_H = 476.84875, 287.7325       # pt
PRINT_W = 0.50 * 397.48                     # printed width in the paper, 0.50\linewidth
SCALE = PRINT_W / PAGE_W
TEXT_PT = 8.0 / SCALE                       # native size that prints at 8 pt
LEG_PT = 7.0 / SCALE                        # native size that prints at 7 pt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=EVAL_ROOT / "cfg" / "results.json", help="cfg_aggregate.py --out")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "cfg", help="directory for the PDF and PNG")
    ap.add_argument("--font-dir", type=Path, default=Path.home() / ".fonts",
                    help="directory holding arial*.ttf; matplotlib's default sans-serif is used when they are absent")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- data
    ref = json.loads(args.results.read_text())["scales"]
    interp_acc = [ref[str(s)]["interp"]["accuracy"] for s in SCALES]
    extra_acc = [ref[str(s)]["extra"]["accuracy"] for s in SCALES]

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
    for s in SCALES:
        i, e = ref[str(s)]["interp"], ref[str(s)]["extra"]
        print("scale %5g  interp acc %5.1f IoU %.3f | extra acc %5.1f IoU %.3f | radii >= 50%%: %s" % (
            s, i["accuracy"], i["mean_iou"], e["accuracy"], e["mean_iou"],
            [k for k, v in ref[str(s)]["extra_accuracy_per_radius"].items() if v >= 50]))
    print("wrote", out)


if __name__ == "__main__":
    main()
