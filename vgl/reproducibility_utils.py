import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

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


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def write_run_config(experiment_dir, args, extra=None):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
        "args": vars(args),
        "extra": extra or {},
    }
    with open(os.path.join(experiment_dir, "run_config.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
