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
# Table 3 reports each cell as the mean over six training seeds.
TABLE3_SEEDS = (0, 1, 2, 3, 4, 5)

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

# Color+Count and Position+Rotation were retrained for Table 3 with the training command of
# submit_training.py, unchanged (1000 epochs at global batch 64), on the datasets of
# generate_datasets.py, for TABLE3_SEEDS:
#   color_count        with the folder-name fix in train_compositional.py; seeds 0-2 and 3-5 train
#                      on two separately generated copies of each dataset (same arguments).
#   position_rotation  trains on 45/135/225/315 deg and holds out 0/15/30/330/345 deg, outside the
#                      training range; one dataset for all six seeds.
# sample_table3.py samples each run's final checkpoint, or the checkpoint at exactly the step below
# (1000 epochs), which resumed Color+Count runs train past.
#
# This pipeline renders each dataset once and trains all six seeds on it. In the paper's Table 3,
# 14 settings pool two renderings of the same combinations, seeds 0-2 on the first and 3-5 on
# the second: color_shape, color_count and the 75% coverage of every other pair except
# position_rotation, plus the 50% coverage of color_position, position_shape and shape_count. The
# other 10 settings (position_rotation, and the 25% coverage of color_position, position_shape,
# shape_count, and the 25% and 50% coverage of position_size and size_shape) use one rendering
# for all six seeds.
SAMPLE_STEPS = {
    ("color_count", 25): 48000,
    ("color_count", 50): 58000,
    ("color_count", 75): 87000,
}
