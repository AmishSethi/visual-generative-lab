#!/usr/bin/env python
"""Guidance sweep of Appendix A.5 (Figure 6): sample the size model trained with conditioning dropout
(submit_training.py --families cfg) at several classifier-free guidance scales.

EMA weights, loaded key for key; 250 DDPM steps; 20 samples per radius in one batch; the Table 2
interpolation radii (5.5-19.5) and extrapolation radii (1-4, 21-30). The unconditional branch is the
null radius 0 (the zero token), and scale 1 samples the conditional model alone. x_T of sample i is
seeded 42 + i (vgl.eval_radius.generate_samples); the per-step noise of condition k at the j-th scale of
SCALES is seeded SEED_BASE + 1000 * j + k. Writes

  <out>/scale_<s>/c<k>.npz        images uint8 [n_samples, 64, 64, 3]
  <out>/scale_<s>/c<k>.json       split, radius, scale, seed
  <out>/scale_<s>/manifest.json   checkpoint, settings, condition files

A condition whose .json already exists is skipped, so a job restarted after a timeout resumes.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.diffusion import create_diffusion
from vgl.eval_radius import generate_samples, load_model
from vgl.reproducibility_utils import git_state
from paper_runs.appendix.manifest import EVAL_ROOT, RESULTS_ROOT
from paper_runs.table2.evaluate_table2 import RunSelection, build_size_config, build_size_queries, load_json
from paper_runs.table2.manifest import DATASET_SPECS

SCALES = (1.0, 2.0, 4.0, 10.0, 15.0, 20.0, 50.0, 100.0)
SEED_BASE = 7_000_000


def newest_finished_run(results_dir):
    """The newest run under a train.py --results-dir that holds a final checkpoint."""
    runs = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        prefix = run_dir.name.split("-", 1)[0]
        finals = sorted((run_dir / "checkpoints").glob("final_*.pt")) if run_dir.is_dir() else []
        if prefix.isdigit() and finals:
            runs.append((int(prefix), run_dir, finals[-1]))
    if not runs:
        raise SystemExit(f"no finished run under {results_dir}")
    _, run_dir, checkpoint = max(runs, key=lambda item: item[0])
    return RunSelection(run_dir=run_dir, checkpoint=checkpoint, run_config=load_json(run_dir / "run_config.json"))


def main():
    ap = argparse.ArgumentParser(description="Sample the guidance sweep of Appendix A.5 (Figure 6).")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_ROOT / "cfg" / "size" / "seed_0",
                    help="train.py --results-dir of the model; its newest finished run is sampled")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="sample this checkpoint instead (its run_config.json must sit beside it)")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "cfg" / "samples")
    ap.add_argument("--scales", nargs="+", type=float, default=list(SCALES), choices=SCALES)
    ap.add_argument("--radii", nargs="+", type=float, default=None,
                    help="only these radii (default: every interpolation and extrapolation radius)")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    if args.checkpoint:
        ckpt = args.checkpoint.resolve()
        selection = RunSelection(run_dir=ckpt.parent, checkpoint=ckpt, run_config=load_json(ckpt.parent / "run_config.json"))
    else:
        selection = newest_finished_run(args.results_dir)
    config = build_size_config(selection)
    config["use_cfg"] = True  # generate_samples guides only when cfg_scale > 1

    queries = build_size_queries(DATASET_SPECS["size"])
    conditions = [(split, value) for split in ("interp", "extra") for value in queries[split]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    weights = torch.load(selection.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(weights["ema"], strict=True)
    train_steps = weights.get("train_steps")
    del weights
    print(f"{selection.checkpoint}: EMA weights, strict load ok, step {train_steps}", flush=True)
    diffusion = create_diffusion(str(args.num_sampling_steps))

    for scale in args.scales:
        out = args.out / f"scale_{scale:g}"
        out.mkdir(parents=True, exist_ok=True)
        for k, (split, value) in enumerate(conditions):
            stem = out / f"c{k:05d}"
            if (args.radii and value not in args.radii) or stem.with_suffix(".json").exists():
                continue
            seed = SEED_BASE + 1000 * SCALES.index(scale) + k
            torch.manual_seed(seed)  # per-step noise inside p_sample comes from the global RNG
            samples = generate_samples(model, diffusion, [value], config, device, num_samples=args.num_samples,
                                       cfg_scale=scale, batch_size=args.batch_size,
                                       num_sampling_steps=args.num_sampling_steps)
            images = np.stack(samples[value])
            tmp = out / f".c{k:05d}.npz.tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, images=images)
            os.replace(tmp, stem.with_suffix(".npz"))
            record = {"index": k, "split": split, "value": value, "key": f"{value:.6f}", "cfg_scale": scale,
                      "seed": seed, "num_samples": int(images.shape[0])}
            stem.with_suffix(".json").write_text(json.dumps(record))
            print(f"scale {scale:g}: {split} r={value:g} done", flush=True)

        manifest = {
            "checkpoint": str(selection.checkpoint), "train_steps": train_steps, "weights": "checkpoint['ema']",
            "cfg_scale": scale, "null_radius": config["null_radius"],
            "settings": {"num_samples": args.num_samples, "batch_size": args.batch_size,
                         "diffusion": f"create_diffusion('{args.num_sampling_steps}')",
                         "x_T_seed": "42 + sample index",
                         "step_seed": f"{SEED_BASE} + 1000 * {SCALES.index(scale)} + condition index",
                         "device": str(device), "torch": torch.__version__},
            "repo_git": git_state(),
            "files": sorted(p.stem for p in out.glob("c*.json")),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
