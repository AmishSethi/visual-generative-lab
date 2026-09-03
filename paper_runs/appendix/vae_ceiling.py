#!/usr/bin/env python
"""Measure the VAE reconstruction ceiling on VGL renders at 64 and 128 px.

The VAE-latent ablation underperforms the pixel-space baseline.  One
candidate explanation is that on low-resolution renders the VAE bottleneck
discards fine-grained geometric information, so a latent model never sees the
skill it is asked to learn.

This tests that hypothesis directly and needs no trained model.  Ground-truth
renders are pushed through the same pretrained VAE the latent runs use
(stabilityai/sd-vae-ft-ema), encoded then decoded, and the reconstructions are
scored with the paper's own metrics.  The result is an upper bound on what any
latent-space model could achieve:

  * a low ceiling means the autoencoder itself discards the skill information,
    so the latent row is bounded by the VAE rather than by the diffusion model;
  * a high ceiling shows latent diffusion genuinely failed to learn the skill.

Reporting both resolutions separates "the VAE cannot represent small shapes"
from "the VAE cannot represent these shapes at all".
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.manifest import EVAL_ROOT
from paper_runs.table2.generate_canonical_datasets import render_arrow, render_circle

SAMPLES_PER_CONDITION = 8


def load_vae(device):
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    return vae


@torch.no_grad()
def roundtrip(images, vae, device):
    """Encode then decode a batch of PIL images; returns uint8 arrays."""
    array = np.stack([np.asarray(img, dtype=np.float32) for img in images])
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(device) / 127.5 - 1.0
    latent = vae.encode(tensor).latent_dist.sample().mul_(0.18215)
    decoded = vae.decode(latent / 0.18215).sample
    out = ((decoded.clamp(-1, 1) + 1.0) * 127.5).permute(0, 2, 3, 1).cpu().numpy()
    return out.round().astype(np.uint8)


def circle_mask(image_np, threshold=30):
    white = np.array([255, 255, 255], dtype=np.float32)
    diff = np.linalg.norm(image_np.astype(np.float32) - white, axis=2)
    return diff > threshold


def iou(mask_a, mask_b):
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.0


def eval_size(vae, device, image_size, scale):
    """Mean IoU between the render and its VAE round-trip, per radius."""
    radii = [int(r * scale) for r in range(5, 21)]
    per_radius = {}
    for radius in radii:
        ious = []
        rng = np.random.default_rng(radius)
        originals = [
            render_circle(radius, 0, 0, image_size, (240, 90, 90), (255, 255, 255))
            for _ in range(SAMPLES_PER_CONDITION)
        ]
        recon = roundtrip(originals, vae, device)
        for original, rec in zip(originals, recon):
            ious.append(iou(circle_mask(np.asarray(original)), circle_mask(rec)))
        per_radius[radius] = float(np.mean(ious))
    passing = 100.0 * np.mean([v >= 0.90 for v in per_radius.values()])
    return {"per_radius_mean_iou": per_radius,
            "mean_iou": float(np.mean(list(per_radius.values()))),
            "pct_radii_above_0.90_iou": passing}


def eval_position(vae, device, image_size, scale):
    """Centroid displacement introduced by the VAE round-trip."""
    radius = int(8 * scale)
    limit = 18.0 * scale
    coords = np.linspace(-limit, limit, 8)
    errors = []
    for x in coords:
        originals = [
            render_circle(radius, int(x), int(y), image_size, (240, 90, 90), (255, 255, 255))
            for y in coords
        ]
        recon = roundtrip(originals, vae, device)
        for original, rec in zip(originals, recon):
            m0, m1 = circle_mask(np.asarray(original)), circle_mask(rec)
            if m0.sum() == 0 or m1.sum() == 0:
                errors.append(float("inf"))
                continue
            y0, x0 = np.nonzero(m0)
            y1, x1 = np.nonzero(m1)
            errors.append(float(np.hypot(x1.mean() - x0.mean(), y1.mean() - y0.mean())))
    finite = [e for e in errors if np.isfinite(e)]
    threshold = 2.0 * scale
    return {"mean_position_error_px": float(np.mean(finite)) if finite else None,
            "pct_within_threshold": 100.0 * np.mean([e <= threshold for e in errors]),
            "threshold_px": threshold}


def eval_rotation(vae, device, image_size, scale):
    """Angle error introduced by the VAE round-trip, via the paper's detector."""
    from vgl.eval_rotation import extract_arrow_angle_template_matching

    shape_size = int(20 * scale)
    angles = list(range(0, 360, 15))
    errors = []
    for angle in angles:
        originals = [
            render_arrow(angle, image_size, shape_size, (240, 90, 90), (255, 255, 255))
            for _ in range(2)
        ]
        recon = roundtrip(originals, vae, device)
        for rec in recon:
            resized = rec
            if image_size != 64:
                from PIL import Image

                resized = np.asarray(Image.fromarray(rec).resize((64, 64), Image.Resampling.LANCZOS))
            detected, ok = extract_arrow_angle_template_matching(resized)
            if not ok:
                errors.append(180.0)
                continue
            diff = abs((float(detected) - angle + 180.0) % 360.0 - 180.0)
            errors.append(diff)
    return {"mean_angle_error_deg": float(np.mean(errors)),
            "pct_within_5deg": 100.0 * np.mean([e <= 5.0 for e in errors])}


def main():
    parser = argparse.ArgumentParser(description="VAE reconstruction ceiling on VGL renders.")
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "vae_ceiling.json")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[64, 128, 256])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(device)

    payload = {"vae": "stabilityai/sd-vae-ft-ema", "resolutions": {}}
    for image_size in args.resolutions:
        scale = image_size / 64.0
        print(f"\n=== {image_size}x{image_size} (geometry scaled {scale:g}x) ===", flush=True)
        entry = {
            "size": eval_size(vae, device, image_size, scale),
            "position": eval_position(vae, device, image_size, scale),
            "rotation": eval_rotation(vae, device, image_size, scale),
        }
        payload["resolutions"][str(image_size)] = entry
        print(f"  size     : mean IoU {entry['size']['mean_iou']:.3f}, "
              f"{entry['size']['pct_radii_above_0.90_iou']:.0f}% of radii clear the 0.90 bar")
        print(f"  position : mean error {entry['position']['mean_position_error_px']:.2f} px, "
              f"{entry['position']['pct_within_threshold']:.0f}% within {entry['position']['threshold_px']:.0f} px")
        print(f"  rotation : mean error {entry['rotation']['mean_angle_error_deg']:.2f} deg, "
              f"{entry['rotation']['pct_within_5deg']:.0f}% within 5 deg")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
