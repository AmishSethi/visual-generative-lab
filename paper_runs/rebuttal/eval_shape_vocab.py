#!/usr/bin/env python
"""Evaluate the shape-vocabulary coverage runs (Reviewer XDWw).

For every trained (S, coverage) model we generate samples for both the seen
combinations (category A) and the held-out ones (category B, compositional
interpolation) and score shape and colour with the validated extractors in
shapes8.py.

Category B is the number the experiment turns on: it is what "compositional
generalisation" means in the paper's Table 3, and comparing it across the
S x coverage grid separates coverage fraction from the absolute number of
training combinations K.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.diffusion import create_diffusion
from vgl.models_compositional import DiT_models_compositional
from paper_runs.rebuttal.manifest import DATASET_ROOT, EVAL_ROOT, RESULTS_ROOT
from paper_runs.rebuttal.shapes8 import (
    COLORS, COLORS16, SHAPES4, SHAPES8, classify_color, classify_shape,
)

ACTIVE_PROPERTIES = ["shape", "color"]


def find_checkpoint(results_dir):
    candidates = sorted(results_dir.rglob("final_*.pt"))
    if not candidates:
        candidates = sorted(results_dir.rglob("*.pt"),
                            key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {results_dir}")
    return candidates[-1]


def load_model(checkpoint_path, num_shapes, num_colors, device):
    model = DiT_models_compositional["DiT-S/2"](
        input_size=64,
        in_channels=3,
        conditioning_method="concat",
        property_dropout_prob=0.0,
        num_shapes=num_shapes,
        num_colors=num_colors,
        null_embedding_type="learnable",
        active_properties=ACTIVE_PROPERTIES,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} e.g. {unexpected[:3]}")
    model.eval()
    return model


@torch.no_grad()
def sample_combo(model, diffusion, shape_id, color_id, n, device, batch=20):
    images = []
    while len(images) < n:
        k = min(batch, n - len(images))
        z = torch.randn(k, 3, 64, 64, device=device)
        kwargs = {
            "shape": torch.full((k,), shape_id, device=device, dtype=torch.long),
            "color": torch.full((k,), color_id, device=device, dtype=torch.long),
        }
        out = diffusion.p_sample_loop(model, z.shape, z, clip_denoised=True,
                                      model_kwargs=kwargs, progress=False)
        arr = ((out.clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        images.extend(arr)
    return images[:n]


def evaluate_run(name, results_dir, dataset_dir, num_samples, steps, device, dump_dir=None):
    meta = json.loads((dataset_dir / "dataset_metadata.json").read_text())
    vocabulary = SHAPES8[:meta["num_shapes"]] if meta["num_shapes"] == 8 else SHAPES4
    # The colour vocabulary is 8 or 16 depending on the cell; the embedding table,
    # the id mapping and the nearest-colour palette must all agree with it.
    num_colors = int(meta.get("num_colors", len(COLORS)))
    palette = COLORS16[:num_colors]
    shape_ids = {s: i for i, s in enumerate(vocabulary)}
    color_ids = {c: i for i, (c, _) in enumerate(palette)}

    checkpoint = find_checkpoint(results_dir)
    model = load_model(checkpoint, meta["num_shapes"], num_colors, device)
    diffusion = create_diffusion(timestep_respacing=str(steps))

    per_category = {}
    for category, combos in (("A_train", meta["train_combos"]), ("B_heldout", meta["held_out_combos"])):
        shape_hits = color_hits = joint_hits = total = 0
        per_combo = {}
        for combo in combos:
            shape_name, color_name = combo.rsplit("_", 1)
            images = sample_combo(model, diffusion, shape_ids[shape_name], color_ids[color_name],
                                  num_samples, device)
            s_ok = c_ok = j_ok = 0
            for idx, img in enumerate(images):
                pred_shape, _ = classify_shape(img, vocabulary)
                pred_color = classify_color(img, palette=palette)
                s_ok += pred_shape == shape_name
                c_ok += pred_color == color_name
                j_ok += (pred_shape == shape_name) and (pred_color == color_name)
                if dump_dir is not None and idx < 4:
                    from PIL import Image
                    d = dump_dir / name / category / combo
                    d.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img).save(d / f"{idx:02d}.png")
            n = len(images)
            per_combo[combo] = {"shape": 100.0 * s_ok / n, "color": 100.0 * c_ok / n,
                                "joint": 100.0 * j_ok / n, "n": n}
            shape_hits += s_ok
            color_hits += c_ok
            joint_hits += j_ok
            total += n
        per_category[category] = {
            "shape_accuracy": 100.0 * shape_hits / total,
            "color_accuracy": 100.0 * color_hits / total,
            "joint_accuracy": 100.0 * joint_hits / total,
            "num_combinations": len(combos),
            "total_samples": total,
            "per_combination": per_combo,
        }

    del model
    torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint),
        "num_shapes": meta["num_shapes"],
        "coverage": meta["coverage"],
        "K_train_combinations": meta["train_combinations"],
        "samples_per_combination": meta["samples_per_combination"],
        "categories": per_category,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate shape-vocabulary coverage runs.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run names such as s8_cov25; default is the full 2x3 grid.")
    parser.add_argument("--family", default="shape_vocab",
                        choices=["shape_vocab", "shape_vocab_masks"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--dump-samples", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.runs is None:
        if args.family == "shape_vocab":
            args.runs = [f"s{s}_cov{c}" for s in (4, 8) for c in (25, 50, 75)]
        else:
            args.runs = [f"s4_cov{c}_mask{m}" for c in (50, 25) for m in (42, 1, 2, 3, 4)]
    # One file per seed so multiple seeds accumulate instead of overwriting.
    out_path = args.out or EVAL_ROOT / f"{args.family}_results_seed{args.seed}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {}
    for name in args.runs:
        results_dir = RESULTS_ROOT / args.family / name / f"seed_{args.seed}"
        dataset_dir = DATASET_ROOT / args.family / name
        if not results_dir.exists() or not list(results_dir.rglob("*.pt")):
            print(f"[skip] {name}: no checkpoint yet")
            continue
        print(f"\n=== {name} ===", flush=True)
        payload[name] = evaluate_run(
            name, results_dir, dataset_dir, args.num_samples, args.steps, device,
            dump_dir=(EVAL_ROOT / f"{args.family}_samples") if args.dump_samples else None,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        for category, res in payload[name]["categories"].items():
            print(f"  {category:>9}: shape {res['shape_accuracy']:5.1f}%  "
                  f"colour {res['color_accuracy']:5.1f}%  joint {res['joint_accuracy']:5.1f}%")

    if payload:
        print(f"\n{'run':>16} {'S':>2} {'cov':>5} {'K':>4} {'smp/K':>6} "
              f"{'B shape':>8} {'B colour':>9} {'B joint':>8}")
        for name, res in payload.items():
            b = res["categories"]["B_heldout"]
            print(f"{name:>16} {res['num_shapes']:>2} {res['coverage']:>5.0%} "
                  f"{res['K_train_combinations']:>4} {res['samples_per_combination']:>6} "
                  f"{b['shape_accuracy']:>7.1f}% {b['color_accuracy']:>8.1f}% {b['joint_accuracy']:>7.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
