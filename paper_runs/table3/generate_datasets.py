#!/usr/bin/env python
"""Generate the Table 3 compositional datasets with the settings used in the paper.

One dataset per (skill pair, coverage), written to DATASET_ROOT/<prefix>_cov<N> with the
same samples per training combination as the paper's runs.
"""
import argparse
import os as _os
import subprocess
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.table3.manifest import DATASET_ROOT, SKILL_PAIRS

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "scripts" / "generate_compositional_dataset_coverage.py"
COVERAGES = (25, 50, 75)
IMAGE_SIZE = 64
SAMPLES_PER_COMBINATION = {
    "color_shape": 312,
    "color_count": 312,
    "shape_count": 312,
    "color_position": 50,
    "position_size": 57,
    "position_shape": 100,
    "size_shape": 357,
    "position_rotation": 50,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=list(SKILL_PAIRS), choices=list(SKILL_PAIRS))
    parser.add_argument("--coverages", nargs="+", type=int, default=list(COVERAGES), choices=list(COVERAGES))
    parser.add_argument("--force", action="store_true", help="Regenerate datasets that already exist.")
    args = parser.parse_args()

    for pair in args.pairs:
        spec = SKILL_PAIRS[pair]
        for coverage in args.coverages:
            out = DATASET_ROOT / f"{spec['dataset_prefix']}_cov{coverage}"
            if (out / "dataset_metadata.json").exists() and not args.force:
                print(f"exists, skipping: {out}")
                continue
            command = [
                sys.executable, str(GENERATOR),
                "--output-dir", str(out),
                "--coverage", str(coverage / 100),
                "--samples-per-combination", str(SAMPLES_PER_COMBINATION[pair]),
                "--image-size", str(IMAGE_SIZE),
                "--include-properties", *spec["properties"],
            ]
            print(" ".join(command))
            subprocess.run(command, check=True, cwd=REPO)


if __name__ == "__main__":
    main()
