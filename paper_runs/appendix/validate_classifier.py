#!/usr/bin/env python
"""Validate the 8-shape renderer and classifier on ground truth before use.

Checks two things:
  1. render_shape reproduces the paper's generate_shape_image exactly on the
     original four shapes (so S=4 numbers stay comparable to the paper).
  2. classify_shape recovers the true shape on clean renders, for both the
     4-shape and 8-shape vocabularies, across radii and colours and with the
     same pixel noise the datasets carry.

Prints a confusion matrix so any shape pair the matcher cannot separate is
visible rather than silently deflating the experiment.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.shapes8 import (
    COLORS,
    SHAPES4,
    SHAPES8,
    classify_color,
    classify_shape,
    render_shape,
)

RADII = [8, 10, 12, 14, 16, 18]
NOISE_LEVEL = 0.02


def add_noise(img, rng):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0, NOISE_LEVEL, arr.shape), 0, 1)
    return (arr * 255).astype(np.uint8)


def check_renderer_parity():
    from scripts.generate_compositional_dataset_coverage import generate_shape_image

    worst = 0.0
    for shape in SHAPES4:
        for radius in RADII:
            a = np.array(render_shape(shape, radius, (255, 0, 0), image_size=64))
            b = np.array(
                generate_shape_image(radius=radius, position=(0, 0), shape=shape,
                                     color_rgb=(255, 0, 0), image_size=64, rotation=0, count=1)
            )
            worst = max(worst, float(np.abs(a.astype(int) - b.astype(int)).max()))
    print(f"renderer parity vs paper generate_shape_image: max abs pixel diff = {worst:.0f}")
    return worst


def check_classifier(vocabulary, label, rng):
    correct = 0
    total = 0
    confusion = Counter()
    color_correct = 0
    for shape in vocabulary:
        for radius in RADII:
            for color_name, color_rgb in COLORS:
                img = render_shape(shape, radius, color_rgb, image_size=64)
                noisy = add_noise(img, rng)
                pred, _ = classify_shape(noisy, vocabulary)
                total += 1
                correct += pred == shape
                if pred != shape:
                    confusion[(shape, pred)] += 1
                color_correct += classify_color(noisy) == color_name

    acc = 100.0 * correct / total
    col = 100.0 * color_correct / total
    print(f"\n[{label}] shape accuracy on ground truth: {acc:.1f}%  ({correct}/{total})")
    print(f"[{label}] colour accuracy on ground truth: {col:.1f}%")
    if confusion:
        print(f"[{label}] confusions (true -> pred: n):")
        for (true_shape, pred_shape), n in confusion.most_common():
            print(f"    {true_shape:>9} -> {str(pred_shape):<9}: {n}")
    return acc, col


def degrade(img, rng, noise, blur, scale_jitter, shift_jitter):
    """Approximate the imperfections of a diffusion sample."""
    import cv2

    arr = np.array(img).astype(np.float32)
    if scale_jitter or shift_jitter:
        s = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
        dx = rng.uniform(-shift_jitter, shift_jitter)
        dy = rng.uniform(-shift_jitter, shift_jitter)
        c = (arr.shape[1] - 1) / 2.0
        M = np.float32([[s, 0, c - s * c + dx], [0, s, c - s * c + dy]])
        arr = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if blur:
        arr = cv2.GaussianBlur(arr, (0, 0), blur)
    arr = np.clip(arr / 255.0 + rng.normal(0, noise, arr.shape), 0, 1)
    return (arr * 255).astype(np.uint8)


def check_robustness(vocabulary, label):
    """Shape accuracy under progressively harsher degradation."""
    settings = [
        ("clean (dataset noise)", dict(noise=0.02, blur=0.0, scale_jitter=0.0, shift_jitter=0.0)),
        ("mild  blur+jitter", dict(noise=0.05, blur=0.6, scale_jitter=0.06, shift_jitter=1.5)),
        ("heavy blur+jitter", dict(noise=0.10, blur=1.2, scale_jitter=0.12, shift_jitter=3.0)),
    ]
    print(f"\n[{label}] robustness sweep:")
    for name, kwargs in settings:
        rng = np.random.default_rng(1)
        correct = total = 0
        confusion = Counter()
        for shape in vocabulary:
            for radius in RADII:
                for _, color_rgb in COLORS:
                    img = render_shape(shape, radius, color_rgb, image_size=64)
                    pred, _ = classify_shape(degrade(img, rng, **kwargs), vocabulary)
                    total += 1
                    correct += pred == shape
                    if pred != shape:
                        confusion[(shape, pred)] += 1
        top = ", ".join(f"{a}->{b}:{n}" for (a, b), n in confusion.most_common(3))
        print(f"    {name:<22} {100.0 * correct / total:5.1f}%   {top}")


def main():
    rng = np.random.default_rng(0)
    check_renderer_parity()
    check_classifier(SHAPES4, "S=4", rng)
    check_classifier(SHAPES8, "S=8", rng)
    check_robustness(SHAPES4, "S=4")
    check_robustness(SHAPES8, "S=8")


if __name__ == "__main__":
    main()
