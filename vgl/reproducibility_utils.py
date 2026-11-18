import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import numpy as np
import torch


def configure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_worker_init_fn(base_seed):
    def _seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2 ** 32))
        torch.manual_seed(worker_seed)

    return _seed_worker


def make_data_loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# The repository this code is in, so the recorded commit is the code's whatever the working directory.
_REPO = Path(__file__).resolve().parents[1]


def _git(*command):
    try:
        return subprocess.check_output(["git", *command], cwd=_REPO, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _git_commit():
    return _git("rev-parse", "HEAD")


def git_state():
    """Commit of this code and whether its tracked files have uncommitted changes."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": _git_commit(), "git_dirty": None if status is None else bool(status)}


def write_run_config(experiment_dir, args, extra=None):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        **git_state(),
        "args": vars(args),
        "extra": extra or {},
    }
    path = os.path.join(experiment_dir, "run_config.json")
    if os.path.exists(path):
        # A resumed run keeps the first config and writes its own beside it.
        resumes = len(glob(os.path.join(experiment_dir, "run_config_resume_*.json")))
        path = os.path.join(experiment_dir, f"run_config_resume_{resumes + 1}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
