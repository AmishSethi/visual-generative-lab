#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table2.manifest import CANONICAL_SEEDS, EVAL_ROOT, SKILLS, TABLE_ROWS, VARIANTS


SPLIT_COLUMN_NAMES = {
    "train": "train",
    "interp": "interp.",
    "extra": "extra.",
}


def default_eval_root() -> Path:
    return EVAL_ROOT / "samples20_steps250_cfg1"


def load_json(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def result_path(eval_root: Path, skill: str, variant: str, seed: int) -> Path:
    return eval_root / skill / variant / f"seed_{seed}" / "results.json"


def collect_variant_means(eval_root: Path) -> tuple[dict, list[dict]]:
    means = defaultdict(dict)
    missing = []

    for skill in SKILLS:
        for variant in VARIANTS:
            seed_results = []
            missing_seeds = []
            for seed in CANONICAL_SEEDS:
                path = result_path(eval_root, skill, variant, seed)
                if not path.exists():
                    missing_seeds.append(seed)
                    continue
                payload = load_json(path)
                seed_results.append(payload["paper_metrics"]["splits"])

            if missing_seeds:
                missing.append({"skill": skill, "variant": variant, "missing_seeds": missing_seeds})
                continue

            for split_name in ("train", "interp", "extra"):
                split_values = [seed_result[split_name]["accuracy"] for seed_result in seed_results]
                means[(skill, variant)][split_name] = float(np.mean(split_values))
                means[(skill, variant)][f"{split_name}_std"] = float(np.std(split_values))

    return means, missing


def render_table_rows(means: dict) -> list[dict]:
    rendered = []
    for row_name, variant in TABLE_ROWS.items():
        row = {"model": row_name}
        for skill in SKILLS:
            stats = means.get((skill, variant))
            for split_name, column_name in SPLIT_COLUMN_NAMES.items():
                key = f"{skill}_{split_name}"
                row[key] = None if stats is None else round(stats[split_name], 1)
        rendered.append(row)
    return rendered


def print_markdown_table(rows: list[dict]):
    headers = [
        "model",
        "size_train", "size_interp", "size_extra",
        "position_train", "position_interp", "position_extra",
        "rotation_train", "rotation_interp", "rotation_extra",
        "count_train", "count_interp", "count_extra",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                str(
                    row.get(header.replace("_interp", "_interp.").replace("_extra", "_extra."))
                    if header == "model"
                    else row[
                        header
                        .replace("_interp", "_interp")
                        .replace("_extra", "_extra")
                    ]
                )
                for header in headers
            )
            + " |"
        )


def write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "model",
        "size_train", "size_interp", "size_extra",
        "position_train", "position_interp", "position_extra",
        "rotation_train", "rotation_interp", "rotation_extra",
        "count_train", "count_interp", "count_extra",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "size_train": row["size_train"],
                    "size_interp": row["size_interp"],
                    "size_extra": row["size_extra"],
                    "position_train": row["position_train"],
                    "position_interp": row["position_interp"],
                    "position_extra": row["position_extra"],
                    "rotation_train": row["rotation_train"],
                    "rotation_interp": row["rotation_interp"],
                    "rotation_extra": row["rotation_extra"],
                    "count_train": row["count_train"],
                    "count_interp": row["count_interp"],
                    "count_extra": row["count_extra"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Aggregate paper-locked Table 2 evaluations.")
    parser.add_argument("--eval-root", type=Path, default=default_eval_root())
    args = parser.parse_args()

    means, missing = collect_variant_means(args.eval_root)
    rows = render_table_rows(means)

    summary_payload = {
        "eval_root": str(args.eval_root),
        "rows": rows,
        "missing": missing,
    }
    summary_path = args.eval_root / "table2_summary.json"
    csv_path = args.eval_root / "table2_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
    write_csv(csv_path, rows)

    print(f"Saved {summary_path}")
    print(f"Saved {csv_path}")
    if missing:
        print("Missing evaluations:")
        for item in missing:
            print(f"  {item['skill']}/{item['variant']}: seeds {item['missing_seeds']}")

    print()
    for row in rows:
        print(
            row["model"],
            row["size_train"], row["size_interp"], row["size_extra"],
            row["position_train"], row["position_interp"], row["position_extra"],
            row["rotation_train"], row["rotation_interp"], row["rotation_extra"],
            row["count_train"], row["count_interp"], row["count_extra"],
        )


if __name__ == "__main__":
    main()
