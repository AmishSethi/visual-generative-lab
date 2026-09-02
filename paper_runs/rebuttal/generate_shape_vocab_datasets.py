#!/usr/bin/env python
"""Shape x colour datasets for the coverage-vs-K experiment (Reviewer XDWw).

XDWw's objection: our "coverage matters more than dataset size" result may
really be a "number of unique combinations" result, since raising the coverage
fraction also raises the absolute number of combinations K.  With only 4 shapes
the two are locked together.  Doubling the shape vocabulary unlocks them:

    S=4 (32 combos):  cov 25% -> K=8    cov 50% -> K=16   cov 75% -> K=24
    S=8 (64 combos):  cov 25% -> K=16   cov 50% -> K=32   cov 75% -> K=48

The decisive cell pair is S=4@50% and S=8@25%: identical K (16) and identical
samples per combination, but 50% vs 25% coverage.  If accuracy tracks K the two
are equal; if it tracks coverage fraction, S=4@50% wins.

Total image budget is held at BUDGET for every run, so samples-per-combination
falls as coverage rises exactly as in the paper's setup.
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.manifest import DATASET_ROOT
from paper_runs.rebuttal.shapes8 import COLORS, COLORS16, SHAPES4, SHAPES8, render_shape

BUDGET = 15000
IMAGE_SIZE = 64
RADIUS = 12.0
NOISE_LEVEL = 0.02
SPLIT_SEED = 42
TEST_SAMPLES_PER_COMBO = 20


def add_noise(img, rng):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0, NOISE_LEVEL, arr.shape), 0, 1)
    return (arr * 255).astype(np.uint8)


def select_train_combinations(shapes, colors, coverage, seed=SPLIT_SEED):
    """Sample `coverage` of the shape x colour grid, guaranteeing every shape
    and every colour is seen at least once.

    Both properties use lookup embeddings, so a value that never appears in
    training cannot be generated at all -- that would be a missing-token
    failure rather than a compositional one, and would confound the comparison.
    """
    rng = random.Random(seed)
    color_names = [name for name, _ in colors]
    all_combos = [(s, c) for s in shapes for c in color_names]
    target = int(round(len(all_combos) * coverage))

    # Latin-square cover: walking (shapes[i % S], colors[i % C]) for
    # i < lcm-free max(S, C) touches every shape and every colour in exactly
    # max(S, C) combinations -- the minimum possible.  S=4 at 25% coverage
    # allows only K=8, so any looser construction would overshoot the budget.
    shuffled_shapes = rng.sample(shapes, len(shapes))
    shuffled_colors = rng.sample(color_names, len(color_names))
    cover = []
    for i in range(max(len(shapes), len(color_names))):
        combo = (shuffled_shapes[i % len(shapes)], shuffled_colors[i % len(color_names)])
        if combo not in cover:
            cover.append(combo)

    if len(cover) > target:
        raise ValueError(
            f"coverage {coverage:.0%} gives K={target} but {len(cover)} combinations "
            f"are needed to cover every shape and colour at least once"
        )

    remaining = [c for c in all_combos if c not in set(cover)]
    rng.shuffle(remaining)
    chosen = cover + remaining[: target - len(cover)]

    held_out = [c for c in all_combos if c not in set(chosen)]
    assert len({s for s, _ in chosen}) == len(shapes), "a shape is missing from training"
    assert len({c for _, c in chosen}) == len(color_names), "a colour is missing from training"
    return sorted(chosen), sorted(held_out)


def render_combo_dir(out_dir, shape, color_rgb, n_samples, rng):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_samples):
        img = render_shape(shape, RADIUS, color_rgb, image_size=IMAGE_SIZE)
        from PIL import Image

        Image.fromarray(add_noise(img, rng)).save(out_dir / f"img_{idx:05d}.png")


def build(n_shapes, coverage, output_dir, force=False, split_seed=SPLIT_SEED, n_colors=8,
          budget=BUDGET):
    shapes = SHAPES8[:n_shapes] if n_shapes == 8 else SHAPES4
    colors = COLORS16[:n_colors]
    color_map = dict(colors)

    if output_dir.exists() and not force:
        print(f"[skip] {output_dir} already exists")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    train_combos, held_out = select_train_combinations(shapes, colors, coverage, seed=split_seed)
    per_combo = budget // len(train_combos)
    rng = np.random.default_rng(split_seed)

    print(
        f"S={n_shapes} C={len(colors)} cov={coverage:.0%}: K={len(train_combos)} train combos, "
        f"{per_combo} samples each ({per_combo * len(train_combos)} images), "
        f"{len(held_out)} held-out combos"
    )

    for shape, color_name in train_combos:
        render_combo_dir(output_dir / "train" / f"{shape}_{color_name}",
                         shape, color_map[color_name], per_combo, rng)

    for shape, color_name in held_out:
        render_combo_dir(output_dir / "test_compositional" / f"{shape}_{color_name}",
                         shape, color_map[color_name], TEST_SAMPLES_PER_COMBO, rng)

    metadata = {
        "experiment": "shape_vocabulary_coverage_vs_K",
        "num_shapes": n_shapes,
        "shapes": shapes,
        "colors": [name for name, _ in colors],
        "num_colors": len(colors),
        "coverage": coverage,
        "total_combinations": len(shapes) * len(colors),
        "train_combinations": len(train_combos),
        "samples_per_combination": per_combo,
        "total_train_images": per_combo * len(train_combos),
        "budget": budget,
        "image_size": IMAGE_SIZE,
        "radius": RADIUS,
        "split_seed": split_seed,
        "train_combos": [f"{s}_{c}" for s, c in train_combos],
        "held_out_combos": [f"{s}_{c}" for s, c in held_out],
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Generate shape-vocabulary coverage datasets.")
    parser.add_argument("--shapes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--coverages", nargs="+", type=int, default=[25, 50, 75])
    parser.add_argument("--colors", nargs="+", type=int, default=[8])
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help="Total training images per cell. 15000 saturates the task; "
                             "5000 sits off the ceiling where differences are visible.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mask-seeds", nargs="+", type=int, default=None,
                        help="Emit one dataset per seed into shape_vocab_masks/ to measure "
                             "variance across random coverage masks (Reviewer yLdq, question 2).")
    args = parser.parse_args()

    if args.mask_seeds:
        for n_shapes in args.shapes:
            for cov in args.coverages:
                for seed in args.mask_seeds:
                    out = DATASET_ROOT / "shape_vocab_masks" / f"s{n_shapes}_cov{cov}_mask{seed}"
                    build(n_shapes, cov / 100.0, out, force=args.force, split_seed=seed)
        return

    for n_shapes in args.shapes:
        for n_colors in args.colors:
            for cov in args.coverages:
                suffix = "" if n_colors == 8 else f"_c{n_colors}"
                budget_tag = "" if args.budget == BUDGET else f"_b{args.budget // 1000}k"
                out = DATASET_ROOT / "shape_vocab" / f"s{n_shapes}{suffix}_cov{cov}{budget_tag}"
                build(n_shapes, cov / 100.0, out, force=args.force, n_colors=n_colors,
                      budget=args.budget)


if __name__ == "__main__":
    main()
