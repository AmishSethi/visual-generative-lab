#!/usr/bin/env python
"""Evaluation splits of the Appendix F three- and four-skill models, built from their training grid.

The grid is the one scripts/generate_compositional_dataset_coverage.py draws at 75% coverage for radius,
position, shape (three-skill) and radius, position, shape, color (four-skill): radii 6-14 in training and
16, 18 held out; positions on a 7-px grid from -14 to 14, with x = 14 held out. The splits are

  A_train          training combinations, 5 per (radius, shape)
  B_inrange        in-range combinations not in training
  Cs_size_ood      radius 16 or 18 at a training position
  Cp_position_ood  radius 6-14 at x = 14
  D_both_ood       radius 16 or 18 at x = 14

in full for the three-skill model, and for the four-skill model a seeded sample of 5 (B, Cp) or 10 (Cs, D)
combinations per (radius, shape). Writes <out-dir>/3skill.json and <out-dir>/4skill.json for
`sample_spec.py --splits`; score the samples with paper_runs/table3/score_spec.py.

    python -m paper_runs.appendix.multi_skill_splits --out-dir $VGL_ROOT/appendix/multi_skill
"""
import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generate_compositional_dataset_coverage import create_coverage_based_splits, generate_property_values_coverage

SHAPES = ["circle", "square", "triangle", "diamond"]
COLORS = ["red", "blue", "green", "yellow", "magenta", "cyan", "orange", "purple"]
MODELS = {"3skill": ["radius", "position", "shape"], "4skill": ["radius", "position", "shape", "color"]}
SPLIT_SEED = 20261019


def build_splits(props):
    pv, _ = generate_property_values_coverage(list(props))
    train = create_coverage_based_splits(pv, list(props), 0.75)["train"]
    r_tr, r_ood = pv["radius"][:-2], pv["radius"][-2:]
    p_tr, p_ood = pv["position"][:-5], pv["position"][-5:]
    other = [pv[p] for p in props[2:]]
    in_train = set(train)
    rest = [c for c in itertools.product(r_tr, p_tr, *other) if c not in in_train]
    ood = {"Cs_size_ood": list(itertools.product(r_ood, p_tr, *other)),
           "Cp_position_ood": list(itertools.product(r_tr, p_ood, *other)),
           "D_both_ood": list(itertools.product(r_ood, p_ood, *other))}
    rng = random.Random(SPLIT_SEED)

    def per_cell(pool, k):
        """A seeded sample of k combinations per (radius, shape), all of them if fewer."""
        cells = defaultdict(list)
        for c in pool:
            cells[(c[0], c[2])].append(c)
        out = []
        for key in sorted(cells):
            out += cells[key] if len(cells[key]) <= k else rng.sample(cells[key], k)
        return out

    if len(props) == 3:
        splits = {"A_train": per_cell(train, 5), "B_inrange": rest, **ood}
    else:
        splits = {"A_train": per_cell(train, 5), "B_inrange": per_cell(rest, 5),
                  "Cs_size_ood": per_cell(ood["Cs_size_ood"], 10), "Cp_position_ood": per_cell(ood["Cp_position_ood"], 5),
                  "D_both_ood": per_cell(ood["D_both_ood"], 10)}
    # model conditioning: radius, [x, y], shape id, color id
    return {s: [[float(c[0]), [float(c[1][0]), float(c[1][1])], SHAPES.index(c[2])]
                + ([COLORS.index(c[3][0])] if len(c) > 3 else []) for c in v] for s, v in splits.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, props in MODELS.items():
        splits = build_splits(props)
        json.dump({"properties": props, "splits": splits}, open(args.out_dir / f"{name}.json", "w"), indent=1)
        print(name, {s: len(v) for s, v in splits.items()})


if __name__ == "__main__":
    main()
