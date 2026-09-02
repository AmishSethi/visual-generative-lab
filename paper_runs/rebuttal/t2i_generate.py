#!/usr/bin/env python
"""Generate VGL skill queries with pretrained text-to-image models.

Runs in the `lora` conda env (torch 2.7 / diffusers 0.39 / transformers 4.54),
not the DiT env, which has no transformers.
"""
import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{VGL_ROOT}/hf_cache")

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.rebuttal.t2i_prompts import NEGATIVE_PROMPT, build_all



MODELS = {
    "sd15": {
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionPipeline",
        "resolution": 512,
        "steps": 50,
        "guidance": 7.5,
        "supports_negative": True,
    },
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLPipeline",
        "resolution": 1024,
        "steps": 50,
        "guidance": 7.5,
        "supports_negative": True,
    },
    # DiT-backboned T2I, the closest public architecture to the paper's own
    # model.  The Sigma-512 repo ships weights only (no model_index.json), so
    # we use the Alpha-512 repo, which is a complete diffusers pipeline.
    "pixart": {
        "repo": "PixArt-alpha/PixArt-XL-2-512x512",
        "pipeline": "PixArtAlphaPipeline",
        "resolution": 512,
        "steps": 40,
        "guidance": 4.5,
        "supports_negative": True,
    },
    # NOT USED in the rebuttal.  Every prompt phrasing and sampling setting we
    # tried returned a dense circle texture rather than the requested handful of
    # objects, including for "one red dot".  That is a prompt-following failure
    # we could not separate from our own invocation of the checkpoint, so
    # reporting a score for it would not be defensible.  Kept for reference.
    "sana": {
        "repo": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "pipeline": "SanaPipeline",
        "resolution": 1024,
        "steps": 20,
        "guidance": 5.0,
        "supports_negative": True,
    },
    # PixArt-alpha's successor: same DiT backbone, stronger text encoder and a
    # 1024px training set.  Paired with "pixart" it isolates one generation of
    # improvement within a fixed architecture family.
    "pixart_sigma": {
        "repo": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "pipeline": "PixArtSigmaPipeline",
        "resolution": 1024,
        "steps": 20,
        "guidance": 4.5,
        "supports_negative": True,
    },
    "hunyuandit": {
        "repo": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        "pipeline": "HunyuanDiTPipeline",
        "resolution": 1024,
        "steps": 50,
        "guidance": 5.0,
        "supports_negative": True,
    },
    # An Apache-2.0 distillation of FLUX.1-schnell.  Both black-forest-labs
    # repos are gated to an approval list this account is not on, so this is how
    # the FLUX architecture enters the comparison at all.
    "shuttle3": {
        "repo": "shuttleai/shuttle-3-diffusion",
        "pipeline": "FluxPipeline",
        "resolution": 1024,
        "steps": 4,
        "guidance": 0.0,
        "supports_negative": False,
    },
    # 2025 frontier models.  The size result is the one claim that holds across
    # every model tested, so it is worth testing against the strongest and most
    # recent systems we can actually obtain rather than only 2022-24 ones.
    # FLUX.1-Krea-dev is the FLUX dev lineage without the approval list that
    # blocks FLUX.1-dev itself.
    "flux_krea": {
        "repo": "black-forest-labs/FLUX.1-Krea-dev",
        "pipeline": "FluxPipeline",
        "resolution": 1024,
        "steps": 28,
        "guidance": 3.5,
        "supports_negative": False,
    },
    # 2026 FLUX.2 family.  Distilled for 4-step sampling at guidance 1.0, per the
    # model card, so it is far cheaper than the other frontier entries here.
    # FLUX.2-dev is gated; this one is not.
    "flux2_klein": {
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline": "Flux2KleinPipeline",
        "resolution": 1024,
        "steps": 4,
        "guidance": 1.0,
        "supports_negative": False,
    },
    # NOT RUN for the rebuttal.  Configured and verified reachable, but at 20B it
    # is the most expensive entry here and our cluster fairshare was exhausted
    # before it could be scheduled, so it was dropped in favour of flux_krea.
    # Nothing is known to be wrong with it -- rerun when capacity allows.
    "qwen_image": {
        "repo": "Qwen/Qwen-Image",
        "pipeline": "QwenImagePipeline",
        "resolution": 1024,
        "steps": 30,
        "guidance": 4.0,
        "supports_negative": True,
    },
    # The three entries below are gated to an approval list; they run only if
    # the account accepts each model's licence on huggingface.co.
    "sd35m": {
        "repo": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline": "StableDiffusion3Pipeline",
        "resolution": 1024,
        "steps": 40,
        "guidance": 4.5,
        "supports_negative": True,
    },
    "flux_schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "resolution": 1024,
        "steps": 4,
        "guidance": 0.0,
        "supports_negative": False,
    },
    "flux_dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "pipeline": "FluxPipeline",
        "resolution": 1024,
        "steps": 30,
        "guidance": 3.5,
        "supports_negative": False,
    },
}


def load_pipeline(model_key, device="cuda"):
    import diffusers

    spec = MODELS[model_key]
    cls = getattr(diffusers, spec["pipeline"])
    pipe = cls.from_pretrained(spec["repo"], torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    return pipe, spec


def generate(model_key, out_root, num_samples, skills, batch_size=4):
    pipe, spec = load_pipeline(model_key)
    all_prompts = build_all()

    for skill in skills:
        for condition, prompt in all_prompts[skill]:
            cond_dir = out_root / model_key / skill / str(condition).replace(" ", "_").replace("%", "pct")
            cond_dir.mkdir(parents=True, exist_ok=True)
            existing = len(list(cond_dir.glob("*.png")))
            if existing >= num_samples:
                continue

            (cond_dir / "prompt.txt").write_text(prompt)
            todo = num_samples - existing
            made = existing
            while todo > 0:
                n = min(batch_size, todo)
                kwargs = dict(
                    prompt=[prompt] * n,
                    num_inference_steps=spec["steps"],
                    guidance_scale=spec["guidance"],
                    height=spec["resolution"],
                    width=spec["resolution"],
                    generator=[torch.Generator("cuda").manual_seed(1000 + made + i) for i in range(n)],
                )
                if spec["supports_negative"]:
                    kwargs["negative_prompt"] = [NEGATIVE_PROMPT] * n
                images = pipe(**kwargs).images
                for img in images:
                    img.save(cond_dir / f"sample_{made:03d}.png")
                    made += 1
                todo -= n
            print(f"{model_key}/{skill}/{condition}: {made} images", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate VGL prompts with pretrained T2I models.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--out-root", type=Path,
                        default=Path(f"{VGL_ROOT}/REBUTTAL/eval/t2i/images"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--skills", nargs="+", default=["count", "position", "size", "rotation"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    generate(args.model, args.out_root, args.num_samples, args.skills, args.batch_size)
    print("DONE", args.model)


if __name__ == "__main__":
    main()
