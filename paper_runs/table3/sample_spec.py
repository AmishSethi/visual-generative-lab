#!/usr/bin/env python
"""Sample a compositional checkpoint on a fixed set of combinations and save every image for score_spec.py.

Used for the fixed-budget color + shape models of Figures 4(b) and 5 (--dataset: the held-out
combinations of a scripts/generate_color_shape_dataset.py dataset) and for the three- and four-skill
models of Appendix G (--splits: a file written by paper_runs.appendix.appendix_g_splits).
EMA weights by default, loaded key-for-key; 20 samples per combination; 250 DDPM steps, no guidance;
seeded noise. Writes

  <out>/<split>.npz    images uint8 [n_combo, n_samples, 64, 64, 3]
  <out>/<split>.json   split, properties, combinations
  <out>/manifest.json  checkpoint, sha256, weights, settings

A split whose .npz and .json already exist is skipped, so a rerun after a timeout resumes at the next split.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.diffusion import create_diffusion
from vgl.models_compositional import DiT_models_compositional
from paper_runs.table3.manifest import REPO_ROOT
from paper_runs.table3.sample_table3 import build_kwargs, git_state, to_uint8

SHAPES = ("circle", "square", "triangle", "diamond")
COLORS = ("red", "blue", "green", "yellow", "magenta", "cyan", "orange", "purple")


def color_shape_spec(dataset_dir):
    """The held-out combinations of a color + shape dataset, as (color id, shape id)."""
    meta = json.load(open(Path(dataset_dir) / "dataset_metadata.json"))
    combos = [[COLORS.index(c), SHAPES.index(s)] for s, c in meta["test_combinations"]]
    return {"properties": ["color", "shape"], "splits": {"test_composition": combos}}


def load_model(ckpt_path, properties, weights, model_name, device):
    model = DiT_models_compositional[model_name](input_size=64, in_channels=3, conditioning_method="concat",
                                                 property_dropout_prob=0.0, active_properties=properties)
    raw = Path(ckpt_path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    del raw
    if weights not in ckpt:
        raise SystemExit(f"no '{weights}' weights in {ckpt_path}; keys={list(ckpt)}")
    model.load_state_dict(ckpt[weights], strict=True)
    return model.to(device).eval(), sha, ckpt.get("train_steps")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path, help="color + shape dataset: sample its held-out combinations")
    src.add_argument("--splits", type=Path, help="JSON {properties, splits: {name: [combination, ...]}}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--model", default="DiT-S/2")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--sampling-seed", type=int, default=0, help="offset mixed into every batch seed")
    args = ap.parse_args()

    spec = color_shape_spec(args.dataset) if args.dataset else json.load(open(args.splits))
    props = spec["properties"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    t0 = time.time()
    model, sha, train_steps = load_model(args.checkpoint, props, args.weights, args.model, device)
    print(f"{args.checkpoint} [{args.weights}] strict load ok, sha256 {sha}, step {train_steps}", flush=True)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    args.out.mkdir(parents=True, exist_ok=True)

    n = args.num_samples
    for split, combos in spec["splits"].items():
        npz, meta_path = args.out / f"{split}.npz", args.out / f"{split}.json"
        if npz.exists() and meta_path.exists():
            print(f"{split}: done", flush=True)
            continue
        jobs = [(ci, k) for ci in range(len(combos)) for k in range(n)]
        images = np.zeros((len(combos), n, 64, 64, 3), dtype=np.uint8)
        base = (int(hashlib.sha256(split.encode()).hexdigest()[:8], 16) + args.sampling_seed * 1_000_003) % (2**31)
        nb = -(-len(jobs) // args.batch_size)
        for b in range(nb):
            batch = jobs[b * args.batch_size:(b + 1) * args.batch_size]
            bseed = (base + b) % (2**31)
            g = torch.Generator(device=device).manual_seed(bseed)
            z = torch.randn(len(batch), 3, 64, 64, device=device, generator=g)
            torch.manual_seed(bseed)  # per-step noise inside p_sample comes from the global RNG
            kw = build_kwargs([combos[ci] for ci, _ in batch], props, device)
            with torch.no_grad():
                s = diffusion.p_sample_loop(model, z.shape, z, clip_denoised=True, model_kwargs=kw, progress=False)
            u8 = to_uint8(s)
            for j, (ci, k) in enumerate(batch):
                images[ci, k] = u8[j]
            print(f"{split} batch {b + 1}/{nb}: {len(batch)} samples", flush=True)
        tmp = args.out / f".{split}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, images=images)
        os.replace(tmp, npz)
        json.dump({"split": split, "properties": props, "num_samples": n, "combinations": combos},
                  open(meta_path, "w"), indent=1)

    manifest = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha, "train_steps": train_steps,
        "weights": f"checkpoint['{args.weights}']", "properties": props,
        "source": str(args.dataset or args.splits),
        "splits": {s: len(v) for s, v in spec["splits"].items()},
        "settings": {"model": f"{args.model} concat", "num_samples_per_combination": n,
                     "diffusion": f"create_diffusion('{args.num_sampling_steps}')",
                     "sampler": "p_sample_loop, clip_denoised=True, no guidance",
                     "batch_size": args.batch_size, "sampling_seed": args.sampling_seed,
                     "base_seed": "(int(sha256(split)[:8], 16) + sampling_seed * 1000003) % 2**31, plus the batch index",
                     "device": str(device), "torch": torch.__version__},
        "repo_git": git_state(REPO_ROOT), "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(manifest, open(args.out / "manifest.json", "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
