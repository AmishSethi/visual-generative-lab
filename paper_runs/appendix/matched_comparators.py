#!/usr/bin/env python
"""Recompute every simple-64 baseline comparator on the *same* condition set as
the experiment it is compared against.

Extrapolation condition sets differ across experiments. Size uses a 14-condition
extended grid ({1..4} u {21..30}); the visually complex runs score 9 conditions
({1..4} u {21..25}). Position's published cell uses a 320-condition extended grid;
the complex runs score 100 evenly sampled conditions. Means taken over different
condition sets have different denominators and are not directly comparable.

This restricts the baseline's per-condition results to the exact conditions the
comparison experiment scored, then averages. For position the two condition sets
are disjoint (the complex runs score positions from the training boundary, Table 2
those at least 4 px beyond it), so --sample-position samples the Table 2 position
baseline on the complex runs' conditions instead (needs a GPU). The result is
written to a JSON file (default: $VGL_ROOT/appendix/eval/matched_comparators.json).
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import os as _os
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
APPENDIX = Path(f"{VGL_ROOT}/appendix")
T2 = Path(f"{VGL_ROOT}/MORE_SEEDS/table2")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Simple-64 runs behind each comparator: the size baseline (Table 11 averages seeds 0-9) and the three
# learnable-null rotation baseline seeds as evaluated by paper_runs/table2 (submit_eval.py), and for
# count the random-layout comparator (radius-8 circles, 1000 epochs, seeds 0-8; eval_family.py
# --family count_random), the layout and schedule of the complex count runs.
BASELINES = {
    "size": T2 / "eval" / "samples20_steps250_cfg1" / "size" / "baseline",
    "rotation": T2 / "eval" / "samples20_steps250_cfg1" / "rotation" / "baseline",
    "count": APPENDIX / "eval" / "count_random" / "count",
}
POSITION_MATCHED = APPENDIX / "eval" / "matched_position_baseline.json"


def baseline_per_condition(base, split):
    """Per-condition accuracies for the simple-64 baseline, one dict per seed."""
    out = []
    for f in sorted(base.glob("seed_*/results.json")):
        d = json.loads(f.read_text())["paper_metrics"]["splits"].get(split, {})
        ca = d.get("condition_accuracies")
        if ca:
            out.append({k: v["accuracy"] for k, v in ca.items()})
    return out


def restricted_mean(per_seed, keys):
    """Mean over the given conditions, per seed, then across seeds."""
    vals = []
    for cond in per_seed:
        sel = [cond[k] for k in keys if k in cond]
        if len(sel) == len(keys):
            vals.append(st.mean(sel))
    return vals


def sample_position_baseline(keys, seeds=(0, 1, 2)):
    """Accuracy of the Table 2 position baseline (seeds 0-2) on exactly these conditions: 20 samples
    each, 250 DDPM steps, the 2 px threshold. Returns {seed: accuracy}."""
    import numpy as np
    import torch
    import paper_runs.table2.evaluate_table2 as t2
    from vgl.eval_position import generate_samples, load_model

    values = [tuple(float(x) for x in key.split(",")) for key in keys]
    meta = t2.load_dataset_metadata("position")
    train_pos = [t2.round_pair(tuple(p)) for p in meta["positions"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for seed in seeds:
        sel = t2.latest_finished_run("position", "baseline", seed)
        cfg = t2.build_position_config(sel, meta)
        model = load_model(cfg, device)
        sampler = t2.create_sampler(cfg, 250)
        samples = generate_samples(model, sampler, values, cfg, device, num_samples=20,
                                   cfg_scale=1.0, batch_size=20, num_sampling_steps=250)
        hit = tot = 0
        for v in values:
            for img in samples[v]:
                m = t2.calculate_position_metrics_enhanced(
                    np.asarray(img, dtype=np.uint8), expected_pos=v,
                    training_positions=train_pos, image_size=cfg["image_size"], debug=False)
                hit += bool(m["detection_success"]) and m["position_error"] <= t2.POSITION_ERROR_THRESHOLD
                tot += 1
        out[seed] = 100.0 * hit / tot
        print(f"  position baseline seed {seed}: {out[seed]:.2f}% on {len(values)} conditions", flush=True)
        del model
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=APPENDIX / "eval" / "matched_comparators.json")
    ap.add_argument("--sample-position", action="store_true",
                    help="Sample the Table 2 position baseline on the complex runs' conditions and write "
                         "matched_position_baseline.json (needs a GPU); without it that file is read.")
    a = ap.parse_args()

    # experiment file -> (skill, split)
    EXPERIMENTS = {
        "realistic": [("size", "extra"), ("position", "extra"), ("rotation", "extra"), ("count", "extra")],
    }

    report = {}
    for family, specs in EXPERIMENTS.items():
        for skill, split in specs:
            # conditions the experiment actually scored
            exp_files = sorted(APPENDIX.glob(f"eval/{family}_{skill}.json")) + \
                        sorted(f for f in APPENDIX.glob(f"eval/{family}_s[0-9]*.json"))   # seed files only; never the per-skill file twice
            keys, exp_vals = None, []
            for f in exp_files:
                try:
                    blob = json.loads(f.read_text())
                except Exception:
                    continue
                sk = blob.get("skills", {}).get(skill)
                if not sk:
                    continue
                sp = sk["splits"].get(split, {})
                ca = sp.get("condition_accuracies")
                if ca:
                    keys = keys or sorted(ca)
                    exp_vals.append(sp["accuracy"])
            if not keys:
                print(f"[skip] {family}/{skill}: no per-condition data"); continue

            if skill == "position":
                if a.sample_position:
                    per_seed = sample_position_baseline(keys)
                    vals = list(per_seed.values())
                    POSITION_MATCHED.parent.mkdir(parents=True, exist_ok=True)
                    POSITION_MATCHED.write_text(json.dumps(
                        {"n_conditions": len(keys), "per_seed": per_seed, "mean": st.mean(vals),
                         "std": st.stdev(vals) if len(vals) > 1 else 0.0}, indent=1))
                saved = json.loads(POSITION_MATCHED.read_text()) if POSITION_MATCHED.exists() else {}
                base_seeds = []
                matched = list(saved["per_seed"].values()) if saved.get("n_conditions") == len(keys) else []
            else:
                base_seeds = baseline_per_condition(BASELINES[skill], split)
                matched = restricted_mean(base_seeds, keys)
            published = restricted_mean(base_seeds, sorted(base_seeds[0])) if base_seeds else []

            report[f"{family}:{skill}:{split}"] = {
                "n_conditions_experiment": len(keys),
                "n_conditions_published": len(base_seeds[0]) if base_seeds else 0,
                "baseline_matched_mean": round(st.mean(matched), 2) if matched else None,
                "baseline_matched_std": round(st.stdev(matched), 2) if len(matched) > 1 else 0.0,
                "baseline_published_grid_mean": round(st.mean(published), 2) if published else None,
                "experiment_mean": round(st.mean(exp_vals), 2) if exp_vals else None,
                "experiment_n_seeds": len(exp_vals),
                "delta_matched": round(st.mean(exp_vals) - st.mean(matched), 2)
                                 if exp_vals and matched else None,
            }
            r = report[f"{family}:{skill}:{split}"]
            f = lambda v: "  n/a" if v is None else f"{v:6.1f}"
            note = "" if r["baseline_matched_mean"] is not None else \
                   "   <-- no seed covers all experiment conditions; grids disjoint"
            print(f"{skill:9s} exp {f(r['experiment_mean'])} (n={r['experiment_n_seeds']})  "
                  f"baseline matched {f(r['baseline_matched_mean'])} on {r['n_conditions_experiment']} conds "
                  f"(published: {r['baseline_published_grid_mean']} on {r['n_conditions_published']})  "
                  f"delta {f(r['delta_matched'])}{note}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
