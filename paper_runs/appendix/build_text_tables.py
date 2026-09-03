#!/usr/bin/env python
"""Precompute frozen CLIP text embeddings for VGL skill queries.

Text-encoder conditioning ablation.  The paper's numeric conditioning differs
from most real-world generation settings, so this family conditions VGL the way
a text-to-image model does -- on a frozen text encoder's output for a natural
language description of the skill value -- and checks whether the
generalisation picture changes.

Embeddings are precomputed here (in an environment with ``transformers``
installed) and saved as a lookup table, so training and evaluation never need to
load the text encoder.  Out-of-range values get embeddings the same way in-range
ones do, by encoding their sentence, so extrapolation is a property of the
language representation rather than of an extrapolating projection layer.
"""
import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{VGL_ROOT}/hf_cache")

import numpy as np
import torch



TEXT_MODEL = "openai/clip-vit-base-patch32"

PROMPTS = {
    "size": lambda v: f"a photo of a single circle with a radius of {v:g} pixels",
    "count": lambda v: f"a photo of exactly {int(round(v))} circles",
    "rotation": lambda v: f"a photo of an arrow rotated {v:g} degrees",
}

GRIDS = {
    # 0.5-px steps out to 32 px covers train (5-20), interpolation midpoints and
    # the extended extrapolation set (1-4 and 21-30).
    "size": np.arange(0.0, 32.5, 0.5),
    "count": np.arange(0.0, 11.0, 1.0),
    "rotation": np.arange(0.0, 360.0, 1.0),
}


def build(skill, out_path, device):
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(TEXT_MODEL)
    encoder = CLIPTextModel.from_pretrained(TEXT_MODEL).to(device).eval()

    values = GRIDS[skill]
    prompts = [PROMPTS[skill](float(v)) for v in values]

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(prompts), 64):
            batch = prompts[start:start + 64]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = encoder(**tokens).pooler_output
            embeddings.append(out.cpu())
    embeddings = torch.cat(embeddings).float()

    table = {
        "skill": skill,
        "text_model": TEXT_MODEL,
        "values": torch.tensor(values, dtype=torch.float32),
        "embeddings": embeddings,
        "example_prompt": prompts[len(prompts) // 2],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, out_path)

    # A quick geometry report: is the language representation of these numerals
    # even monotone?  If it is not, that predicts worse extrapolation than the
    # linear embedder, and is itself part of what this ablation measures.
    normed = torch.nn.functional.normalize(embeddings, dim=-1)
    sim = normed @ normed.T
    neighbour = torch.diagonal(sim, offset=1)
    print(f"[{skill}] {len(values)} prompts -> {tuple(embeddings.shape)}")
    print(f"    example: {table['example_prompt']!r}")
    print(f"    adjacent-value cosine similarity: mean {neighbour.mean():.4f}, "
          f"min {neighbour.min():.4f}")
    print(f"    wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build CLIP text tables for VGL skills.")
    parser.add_argument("--skills", nargs="+", default=["size", "count"])
    parser.add_argument("--out-dir", type=Path,
                        default=Path(f"{VGL_ROOT}/appendix/text_tables"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for skill in args.skills:
        build(skill, args.out_dir / f"{skill}.pt", device)


if __name__ == "__main__":
    main()
