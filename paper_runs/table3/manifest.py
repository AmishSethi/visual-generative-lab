from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
import os
VGL_ROOT = os.environ.get("VGL_ROOT", os.path.expanduser("~/vgl-data"))
ARTIFACT_ROOT = Path(os.environ.get("TABLE3_ARTIFACT_ROOT", f"{VGL_ROOT}/MORE_SEEDS/table3"))
DATASET_ROOT = Path(os.environ.get("TABLE3_DATASET_ROOT", f"{VGL_ROOT}/datasets/compositional_experiments_2prop/datasets"))
RESULTS_ROOT = ARTIFACT_ROOT / "results"
SLURM_ROOT = ARTIFACT_ROOT / "slurm"
LOG_ROOT = ARTIFACT_ROOT / "logs"
EVAL_ROOT = ARTIFACT_ROOT / "eval"

CANONICAL_SEEDS = (0, 1, 2)

COVERAGES = (25, 50, 75)

# Skill pairs from Table 3 of the paper
SKILL_PAIRS = {
    "color_shape": {
        "properties": ["shape", "color"],
        "dataset_prefix": "shape_color",
    },
    "color_count": {
        "properties": ["count", "color"],
        "dataset_prefix": "count_color",
    },
    "shape_count": {
        "properties": ["shape", "count"],
        "dataset_prefix": "shape_count",
    },
    "color_position": {
        "properties": ["position", "color"],
        "dataset_prefix": "position_color",
    },
    "position_size": {
        "properties": ["radius", "position"],
        "dataset_prefix": "radius_position",
    },
    "position_shape": {
        "properties": ["position", "shape"],
        "dataset_prefix": "position_shape",
    },
    "size_shape": {
        "properties": ["radius", "shape"],
        "dataset_prefix": "radius_shape",
    },
    "position_rotation": {
        "properties": ["position", "rotation"],
        "dataset_prefix": "position_rotation",
    },
}
