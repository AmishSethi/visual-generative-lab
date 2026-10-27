#!/usr/bin/env python
"""Sample one Table 3 run and save every generated image for score_table3.py (no scoring).

Online (non-EMA) weights, loaded key-for-key; 20 samples for every combination of every split,
train included; 250 DDPM steps, no guidance. Samples are batched across combinations with
per-sample conditioning tensors and seeded noise, and written to

  <out>/<pair>/cov<c>/seed_<s>/<split>.npz   images uint8 [n_combo, n_samples, 64, 64, 3]
  <out>/<pair>/cov<c>/seed_<s>/<split>.json  split, properties, combinations, old_eval mask
  <out>/<pair>/cov<c>/seed_<s>/manifest.json checkpoint, sha256, git commit, settings, runtime

The run is looked up under RESULTS_ROOT and its dataset under DATASET_ROOT (manifest.py; set
TABLE3_ARTIFACT_ROOT and TABLE3_DATASET_ROOT). The checkpoint is the newest final_*.pt, or the one
at exactly --step (default: manifest.SAMPLE_STEPS). old_eval marks the combinations that
evaluate_table3.py's folder parser reads; the others (e.g. red_cnt2) are sampled too.
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.diffusion import create_diffusion
from vgl.eval_compositional import load_test_combinations_comprehensive, parse_folder_name_comprehensive
from paper_runs.table3 import evaluate_table3 as et
from paper_runs.table3.manifest import ARTIFACT_ROOT, DATASET_ROOT, REPO_ROOT, RESULTS_ROOT, SAMPLE_STEPS, SKILL_PAIRS

SHAPES = {"circle": 0, "square": 1, "triangle": 2, "diamond": 3}
COLORS = {"red": 0, "blue": 1, "green": 2, "yellow": 3, "magenta": 4, "cyan": 5, "orange": 6, "purple": 7}


def _num(s):
    return float(s.replace("n", "-").replace("p", "."))


def parse_folder_fixed(name, properties):
    """Folder-name parser without the old parser's bug (a leading 'red' token is read as a radius)."""
    vals, toks = {}, name.split("_")
    for i, t in enumerate(toks):
        if t in SHAPES:
            vals["shape"] = SHAPES[t]
        elif t in COLORS:
            vals["color"] = COLORS[t]
        elif re.fullmatch(r"cnt\d+", t):
            vals["count"] = int(t[3:])
        elif re.fullmatch(r"rot[n\d.p]+", t):
            vals["rotation"] = _num(t[3:])
        elif re.fullmatch(r"r[n\d]+(p\d+)?", t):
            vals["radius"] = _num(t[1:])
        elif t.startswith("x") and i + 1 < len(toks) and toks[i + 1].startswith("y"):
            vals["position"] = (_num(t[1:]), _num(toks[i + 1][1:]))
    if any(p not in vals for p in properties):
        return None
    return tuple(vals[p] for p in properties)


def dropped_combinations(dataset_path, properties, parse_old):
    """Folders that the old parser rejects (returns None), parsed with the fixed parser, per split."""
    out = {}
    for split in sorted(e.name for e in os.scandir(dataset_path) if e.is_dir()):
        if not (split.startswith("test_") or split == "train"):
            continue
        for folder in sorted(e.name for e in os.scandir(Path(dataset_path) / split) if e.is_dir()):
            if parse_old(folder, properties) is None:
                combo = parse_folder_fixed(folder, properties)
                if combo is None:
                    raise SystemExit(f"cannot parse folder {dataset_path}/{split}/{folder}")
                out.setdefault(split, []).append(combo)
    return out


def build_kwargs(combos, properties, device):
    """Per-sample conditioning tensors; dtypes as in generate_samples_comprehensive."""
    kw = {}
    for i, prop in enumerate(properties):
        vals = [c[i] for c in combos]
        if prop == "position":
            kw[prop] = torch.tensor([[float(v[0]), float(v[1])] for v in vals], device=device, dtype=torch.float32)
        elif prop in ("radius", "rotation"):
            kw[prop] = torch.tensor([float(v) for v in vals], device=device, dtype=torch.float32)
        elif prop in ("shape", "color", "count"):
            kw[prop] = torch.tensor([int(v) for v in vals], device=device, dtype=torch.long)
        else:
            raise ValueError(prop)
    return kw


def to_uint8(samples):
    """Exactly the conversion in evaluate_properties_comprehensive: (x+1)/2, clamp, *255, astype(uint8)."""
    x = torch.clamp((samples.permute(0, 2, 3, 1) + 1) / 2, 0, 1)
    return (x.cpu().numpy() * 255).astype(np.uint8)


def load_model_checked(ckpt_path, properties, device, weights="model"):
    """_load_compositional_model, but the load must match the checkpoint key-for-key."""
    from vgl.models_compositional import DiT_models_compositional as DiT_models
    model = DiT_models["DiT-S/2"](input_size=64, in_channels=3, conditioning_method="concat",
                                  property_dropout_prob=0.0, active_properties=properties)
    raw = Path(ckpt_path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    del raw
    if weights not in ckpt:
        raise SystemExit(f"FATAL: no '{weights}' weights in {ckpt_path}; keys={list(ckpt)}")
    missing, unexpected = model.load_state_dict(ckpt[weights], strict=False)
    if missing or unexpected:
        raise SystemExit(f"FATAL load mismatch for {ckpt_path}: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), sha, {"missing": list(missing), "unexpected": list(unexpected),
                                          "checkpoint_top_level_keys": sorted(ckpt.keys()),
                                          "train_steps": ckpt.get("train_steps")}


def find_checkpoint(pair, coverage, seed, step):
    """(run_dir, checkpoint, run_config): the newest final_*.pt, or the checkpoint at exactly `step`."""
    if step is None:
        return et.latest_finished_run(pair, coverage, seed)
    seed_root = RESULTS_ROOT / pair / f"cov{coverage}" / f"seed_{seed}"
    found = (sorted(seed_root.glob(f"[0-9][0-9][0-9]-*/checkpoints/{step:07d}.pt"))
             or sorted(seed_root.glob(f"[0-9][0-9][0-9]-*/checkpoints/final_{step:07d}.pt")))
    if len(found) != 1:
        raise SystemExit(f"FATAL: need exactly one checkpoint at step {step} under {seed_root}, found {found}")
    run_dir = found[0].parent.parent
    config_path = run_dir / "run_config.json"
    run_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return run_dir, found[0], run_config


def git_state(repo):
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return {"commit": head, "branch": branch, "tracked_changes": dirty.splitlines()}


def main():
    ap = argparse.ArgumentParser(description="Sample one Table 3 run for score_table3.py.")
    ap.add_argument("--pair", required=True, choices=sorted(SKILL_PAIRS))
    ap.add_argument("--coverage", required=True, type=int, choices=[25, 50, 75])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--step", type=int, default=None,
                    help="sample the checkpoint at exactly this training step (default: manifest.SAMPLE_STEPS, "
                         "else the newest final_*.pt)")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--sampling-seed", type=int, default=0, help="offset mixed into every batch seed")
    ap.add_argument("--out-root", type=Path, default=ARTIFACT_ROOT / "samples")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    step = args.step if args.step is not None else SAMPLE_STEPS.get((args.pair, args.coverage))

    out_dir = args.out_root / args.pair / f"cov{args.coverage}" / f"seed_{args.seed}"
    if (out_dir / "manifest.json").exists() and not args.force:
        m = json.load(open(out_dir / "manifest.json"))
        if m.get("complete"):
            print(f"already complete: {out_dir}")
            return
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: no CUDA device")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    t_start = time.time()
    spec = SKILL_PAIRS[args.pair]
    properties = spec["properties"]
    run_dir, checkpoint, run_config = find_checkpoint(args.pair, args.coverage, args.seed, step)
    dataset_path = DATASET_ROOT / f"{spec['dataset_prefix']}_cov{args.coverage}"
    # the run must have trained on this dataset (one "Dataset contains" line per training launch)
    logged = re.findall(r"Dataset contains [\d,]+ images \((.+?)\)", (run_dir / "log.txt").read_text(errors="replace"))
    trained_on = sorted({str(Path(x).parent) for x in logged})
    if trained_on != [str(dataset_path)]:
        raise SystemExit(f"FATAL: {run_dir} trained on {trained_on}, expected {dataset_path}")
    print(f"checkpoint {checkpoint}\ndataset {dataset_path}", flush=True)

    model, sha, load_report = load_model_checked(checkpoint, properties, device)
    if step is not None and load_report["train_steps"] != step:
        raise SystemExit(f"FATAL: {checkpoint} has train_steps {load_report['train_steps']} != {step}")
    print(f"strict load ok (sha256 {sha}, step {load_report['train_steps']})", flush=True)
    diffusion = create_diffusion(str(args.num_sampling_steps))

    combos = load_test_combinations_comprehensive(str(dataset_path), properties, include_train_split=True)
    combos = {k: [tuple(c) for c in v] for k, v in combos.items() if v}
    extra = dropped_combinations(dataset_path, properties, parse_folder_name_comprehensive)
    splits = {}
    for split in sorted(set(combos) | set(extra)):
        old = combos.get(split, [])
        add = [c for c in extra.get(split, []) if c not in set(old)]
        splits[split] = {"combinations": old + add, "old_eval": [True] * len(old) + [False] * len(add)}
        print(f"  {split}: {len(old)} combos in old eval, {len(add)} dropped by old parser (sampled too)", flush=True)

    n = args.num_samples
    jobs = [(s, ci, k) for s in splits for ci in range(len(splits[s]["combinations"])) for k in range(n)]
    images = {s: np.zeros((len(v["combinations"]), n, 64, 64, 3), dtype=np.uint8) for s, v in splits.items()}
    base_seed = int(hashlib.sha256(f"{args.pair}/cov{args.coverage}/seed_{args.seed}".encode()).hexdigest()[:8], 16)
    base_seed = (base_seed + args.sampling_seed * 1_000_003) % (2**31)
    batch_seeds = []
    t_sample = time.time()
    for b0 in range(0, len(jobs), args.batch_size):
        batch = jobs[b0:b0 + args.batch_size]
        bseed = (base_seed + b0 // args.batch_size) % (2**31)
        batch_seeds.append(bseed)
        g = torch.Generator(device=device).manual_seed(bseed)
        z = torch.randn(len(batch), 3, 64, 64, device=device, generator=g)
        torch.manual_seed(bseed)  # per-step noise inside p_sample comes from the global CUDA RNG
        kw = build_kwargs([splits[s]["combinations"][ci] for s, ci, _ in batch], properties, device)
        tb = time.time()
        with torch.no_grad():
            samples = diffusion.p_sample_loop(model, z.shape, z, clip_denoised=True, model_kwargs=kw, progress=False)
        u8 = to_uint8(samples)
        for j, (s, ci, k) in enumerate(batch):
            images[s][ci, k] = u8[j]
        print(f"batch {b0 // args.batch_size + 1}/{-(-len(jobs) // args.batch_size)}: {len(batch)} samples "
              f"in {time.time() - tb:.1f}s", flush=True)
    sampling_seconds = time.time() - t_sample

    for s, v in splits.items():
        tmp = out_dir / f".{s}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, images=images[s])
        os.replace(tmp, out_dir / f"{s}.npz")
        meta = {"split": s, "properties": properties, "num_samples": n,
                "combinations": [[list(x) if isinstance(x, tuple) else x for x in c] for c in v["combinations"]],
                "old_eval": v["old_eval"]}
        json.dump(meta, open(out_dir / f"{s}.json", "w"), indent=1)

    st = Path(checkpoint).stat()
    manifest = {
        "complete": True,
        "pair": args.pair, "coverage": args.coverage, "seed": args.seed,
        "properties": properties,
        "results_root": str(RESULTS_ROOT),
        "run_dir": str(run_dir), "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
        "checkpoint_size": st.st_size,
        "checkpoint_mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "step": step, "train_steps": load_report["train_steps"],
        "weights": "checkpoint['model'] (non-EMA), as _load_compositional_model",
        "load_check": load_report,
        "dataset_path": str(dataset_path), "run_config": run_config,
        "splits": {s: {"n_combinations": len(v["combinations"]), "n_old_eval": sum(v["old_eval"])} for s, v in splits.items()},
        "repo": str(REPO_ROOT), "repo_git": git_state(REPO_ROOT),
        "settings": {"num_samples_per_combination": n, "diffusion": f"create_diffusion('{args.num_sampling_steps}')",
                     "sampler": "p_sample_loop, clip_denoised=True, no guidance (cfg 1)",
                     "model": "DiT-S/2 concat, property_dropout_prob=0.0, model defaults otherwise",
                     "batch_size": args.batch_size, "batching": "flattened over (split, combination, sample) in that order",
                     "noise": "z from torch.Generator(cuda).manual_seed(batch_seed); per-step noise from global RNG after torch.manual_seed(batch_seed)",
                     "base_seed": base_seed, "batch_seeds": batch_seeds, "sampling_seed": args.sampling_seed,
                     "uint8": "clamp((x+1)/2,0,1)*255 then astype(uint8), as evaluate_properties_comprehensive",
                     "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
                     "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": True},
        "n_samples_total": len(jobs),
        "runtime_seconds": {"sampling": round(sampling_seconds, 1), "total": round(time.time() - t_start, 1)},
        "slurm_job": f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', ''))}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}",
        "host": os.uname().nodename,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = out_dir / ".manifest.json.tmp"
    json.dump(manifest, open(tmp, "w"), indent=1)
    os.replace(tmp, out_dir / "manifest.json")
    print(f"saved {out_dir} ({len(jobs)} samples, sampling {sampling_seconds:.0f}s)")


if __name__ == "__main__":
    main()
