#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.manifest import EVAL_NUM_SAMPLES, EVAL_NUM_SAMPLING_STEPS, EVAL_ROOT, TABLE_ROWS


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate canonical Table 2 evaluation outputs.")
    parser.add_argument("--num-samples-per-condition", type=int, default=EVAL_NUM_SAMPLES)
    parser.add_argument("--num-sampling-steps", type=int, default=EVAL_NUM_SAMPLING_STEPS)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_payloads(eval_root: Path):
    payloads = {}
    for path in sorted(eval_root.glob("*/*/seed_*/results.json")):
        with open(path, "r") as handle:
            payload = json.load(handle)
        payloads[(payload["skill"], payload["variant"], int(payload["seed"]))] = payload
    return payloads


def aggregate_variant(payloads, skill, variant):
    per_seed = []
    for seed in (0, 1, 2):
        payload = payloads.get((skill, variant, seed))
        if payload is None:
            return None
        per_seed.append(
            {
                split: float(payload["paper_metrics"]["splits"][split]["accuracy"])
                for split in ("train", "interp", "extra")
            }
        )
    return {
        split: float(np.mean([seed_result[split] for seed_result in per_seed]))
        for split in ("train", "interp", "extra")
    }


def build_table(payloads):
    table = {}
    for row_label, variant in TABLE_ROWS.items():
        row = {}
        for skill in ("size", "position", "rotation", "count"):
            row[skill] = aggregate_variant(payloads, skill, variant)
        table[row_label] = row
    return table


def render_csv(table):
    lines = ["row,skill,train,interp,extra"]
    for row_label, skill_data in table.items():
        for skill, metrics in skill_data.items():
            if metrics is None:
                lines.append(f"{row_label},{skill},,,")
                continue
            lines.append(
                f"{row_label},{skill},{metrics['train']:.1f},{metrics['interp']:.1f},{metrics['extra']:.1f}"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    eval_root = args.eval_root or (
        EVAL_ROOT / f"samples{args.num_samples_per_condition}_steps{args.num_sampling_steps}_cfg1"
    )
    payloads = load_payloads(eval_root)
    table = build_table(payloads)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(table, handle, indent=2)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.write_text(render_csv(table))

    for row_label, skill_data in table.items():
        print(row_label)
        for skill in ("size", "position", "rotation", "count"):
            metrics = skill_data[skill]
            if metrics is None:
                print(f"  {skill}: missing")
            else:
                print(
                    f"  {skill}: {metrics['train']:.1f} / {metrics['interp']:.1f} / {metrics['extra']:.1f}"
                )


if __name__ == "__main__":
    main()
