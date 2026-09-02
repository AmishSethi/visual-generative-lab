#!/usr/bin/env python
"""Nearest-neighbour memorisation probe (Reviewer HeVs, weakness 3).

HeVs notes the paper's memorisation discussion is indirect, and asks whether
poor interpolation/extrapolation reflects memorisation or nearest-support
behaviour rather than learned skill structure.

Two measurements, in raw pixels and in DINOv2 features:

  copy rate      d_NN / d_2NN < 1/3, the usual replication statistic.  On its
                 own this is weak here: VGL training sets contain hundreds of
                 near-duplicates per skill value, so d_NN ~ d_2NN by
                 construction and the ratio is near 1 whatever the model does.

  held-out control  the distance from each *generation* to its nearest training
                 image, compared against the same distance computed for fresh
                 ground-truth renders that were never trained on.  If
                 generations sit no closer to the training set than genuine
                 held-out samples do, there is no replication beyond what the
                 data distribution itself implies.  This is the measurement the
                 conclusion rests on.
"""

import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{VGL_ROOT}/hf_cache")

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.manifest import DATASET_ROOT, EVAL_ROOT, TABLE2_DATASETS
from paper_runs.table2.generate_canonical_datasets import (


    background_color, finalize, foreground_color, render_circle,
    sample_non_overlapping_centers,
)

BATCH = 256


def load_images(paths, size=64):
    out = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        out.append(np.asarray(img, dtype=np.uint8))
    return np.stack(out) if out else np.zeros((0, size, size, 3), dtype=np.uint8)


def pixel_features(images, device):
    x = torch.from_numpy(images).to(device).float().div_(255.0)
    return x.reshape(len(images), -1)


def dinov2_features(images, device):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    feats = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            chunk = [Image.fromarray(im) for im in images[start:start + 64]]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            out = model(**inputs).last_hidden_state[:, 0]  # CLS token
            feats.append(torch.nn.functional.normalize(out, dim=-1))
    return torch.cat(feats)


def nearest_two(query, reference):
    """Smallest and second-smallest L2 distance from each query to reference."""
    d_nn = torch.empty(len(query), device=query.device)
    d_2nn = torch.empty(len(query), device=query.device)
    for start in range(0, len(query), BATCH):
        block = query[start:start + BATCH]
        dists = torch.cdist(block, reference)
        vals, _ = torch.topk(dists, k=2, dim=1, largest=False)
        d_nn[start:start + BATCH] = vals[:, 0]
        d_2nn[start:start + BATCH] = vals[:, 1]
    return d_nn, d_2nn


def fresh_heldout_renders(skill, n_per_value, seed=98765):
    """Ground-truth renders from the training distribution, never trained on."""
    rng = np.random.default_rng(seed)
    images = []
    if skill == "size":
        for radius in range(5, 21):
            for _ in range(n_per_value):
                img = render_circle(radius, 0, 0, 64, foreground_color(rng), background_color(rng))
                images.append(np.asarray(finalize(img, rng), dtype=np.uint8))
    elif skill == "count":
        from PIL import ImageDraw

        for count in range(2, 8):
            for _ in range(n_per_value):
                img = Image.new("RGB", (64, 64), background_color(rng))
                draw = ImageDraw.Draw(img)
                for cx, cy in sample_non_overlapping_centers(count, 8, 64, rng):
                    draw.ellipse([32 + cx - 8, 32 + cy - 8, 32 + cx + 8, 32 + cy + 8],
                                 fill=foreground_color(rng))
                images.append(np.asarray(finalize(img, rng), dtype=np.uint8))
    else:
        raise ValueError(skill)
    return np.stack(images)


def summarise(d_nn, d_2nn, label):
    ratio = (d_nn / d_2nn.clamp_min(1e-8)).cpu().numpy()
    d = d_nn.cpu().numpy()
    return {
        "label": label,
        "n": int(len(d)),
        "mean_nn_distance": float(d.mean()),
        "median_nn_distance": float(np.median(d)),
        "p5_nn_distance": float(np.percentile(d, 5)),
        "copy_rate_pct": float(100.0 * (ratio < 1 / 3).mean()),
        "mean_nn_over_2nn": float(ratio.mean()),
    }


def run(skill, samples_root, device, use_dino, max_train):
    train_dir = TABLE2_DATASETS / skill
    train_paths = sorted(train_dir.rglob("*.png"))
    rng = np.random.default_rng(0)
    if max_train and len(train_paths) > max_train:
        idx = rng.choice(len(train_paths), max_train, replace=False)
        train_paths = [train_paths[i] for i in sorted(idx)]

    gen_paths = sorted((samples_root / "train").rglob("*.png"))
    if not gen_paths:
        raise SystemExit(f"no generations under {samples_root / 'train'}")

    train_images = load_images(train_paths)
    gen_images = load_images(gen_paths)
    held_images = fresh_heldout_renders(skill, n_per_value=max(1, len(gen_images) // 16))

    results = {}
    encoders = {"pixel": pixel_features}
    if use_dino:
        encoders["dinov2"] = dinov2_features

    for space, encode in encoders.items():
        ref = encode(train_images, device)
        entries = []
        for images, label in ((gen_images, "generations"), (held_images, "held_out_real")):
            q = encode(images, device)
            d_nn, d_2nn = nearest_two(q, ref)
            entries.append(summarise(d_nn, d_2nn, label))
            del q
        gen_stats, held_stats = entries
        results[space] = {
            "generations": gen_stats,
            "held_out_real": held_stats,
            "gen_over_heldout_nn_ratio": gen_stats["mean_nn_distance"] / held_stats["mean_nn_distance"],
            "num_train_reference": len(train_images),
        }
        del ref
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Nearest-neighbour memorisation probe.")
    parser.add_argument("--skills", nargs="+", default=["size", "count"])
    parser.add_argument("--samples-root", type=Path, default=EVAL_ROOT / "samples")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--no-dino", action="store_true")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "memorization_probe.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {}
    for skill in args.skills:
        root = args.samples_root / skill / args.variant / f"seed_{args.seed}"
        if not root.exists():
            print(f"[skip] {root} missing")
            continue
        payload[skill] = run(skill, root, device, not args.no_dino, args.max_train)
        print(f"\n=== {skill} ===")
        for space, res in payload[skill].items():
            g, h = res["generations"], res["held_out_real"]
            print(f"  [{space}] mean NN distance to training set:")
            print(f"      generations   {g['mean_nn_distance']:.4f}   copy rate {g['copy_rate_pct']:.1f}%")
            print(f"      held-out real {h['mean_nn_distance']:.4f}   copy rate {h['copy_rate_pct']:.1f}%")
            print(f"      ratio gen/held-out = {res['gen_over_heldout_nn_ratio']:.3f}"
                  "  (<1 would indicate generations hug the training set)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
