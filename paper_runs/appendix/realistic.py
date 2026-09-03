"""Visually complex VGL renders.

Textured backgrounds, shading, cast shadows and varied colours on the same
skill grids as the flat renders, to test whether the single-skill findings
survive appearance closer to natural images.  The 128x128 runs cover
resolution; this covers appearance.

Every skill keeps its exact rule-based ground truth (a circle of radius r really
is at radius r), but the render gains the things that make natural images hard:

  * shaded objects -- radial falloff plus a specular highlight, not flat fill
  * surface texture -- multi-octave value noise modulating the object
  * cast shadows -- soft, offset, semi-transparent
  * textured backgrounds -- smooth colour gradients plus noise, never white
  * global photometric jitter -- brightness, contrast and blur

The object stays strongly saturated while backgrounds stay desaturated, so a
saturation key recovers the object mask exactly.  That keeps the geometry
measurable, which is the whole point: we want to vary appearance complexity
without making the metric itself the confound.  `validate_extraction` checks the
extractor against known geometry before any of it is used.
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OBJECT_HUES = [0, 15, 30, 120, 150, 200, 220, 260, 290, 330]
SATURATION_KEY = 60          # HSV saturation above which a pixel is object
MIN_OBJECT_SATURATION = 150   # objects are rendered well above the key
MAX_BACKGROUND_SATURATION = 45


def _value_noise(shape, rng, octaves=4):
    """Cheap multi-octave value noise in [0, 1]."""
    h, w = shape
    total = np.zeros((h, w), np.float32)
    amplitude = 1.0
    norm = 0.0
    for octave in range(octaves):
        size = max(2, int(2 ** (octave + 2)))
        low = rng.random((size, size)).astype(np.float32)
        total += amplitude * cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        norm += amplitude
        amplitude *= 0.5
    return np.clip(total / norm, 0, 1)


def _background(size, rng):
    """Desaturated gradient plus noise -- never a flat white canvas."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / max(1, size - 1)
    angle = rng.uniform(0, 2 * math.pi)
    ramp = xs * math.cos(angle) + ys * math.sin(angle)
    ramp = (ramp - ramp.min()) / max(1e-6, ramp.max() - ramp.min())

    v0, v1 = rng.uniform(0.35, 0.95), rng.uniform(0.35, 0.95)
    value = v0 + (v1 - v0) * ramp
    value = np.clip(value + 0.12 * (_value_noise((size, size), rng) - 0.5), 0.05, 1.0)

    hue = rng.integers(0, 180)
    saturation = rng.uniform(0, MAX_BACKGROUND_SATURATION)
    hsv = np.stack([
        np.full((size, size), hue, np.float32),
        np.full((size, size), saturation, np.float32) * _value_noise((size, size), rng),
        value * 255.0,
    ], axis=2).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _shaded_object_color(size, rng):
    """Per-pixel object colour: base hue, radial shading, texture, highlight."""
    hue = int(rng.choice(OBJECT_HUES))
    saturation = rng.uniform(MIN_OBJECT_SATURATION, 255)
    texture = _value_noise((size, size), rng)
    value = np.clip(0.55 + 0.35 * texture, 0, 1) * 255.0
    hsv = np.stack([
        np.full((size, size), hue, np.float32),
        np.full((size, size), saturation, np.float32),
        value,
    ], axis=2).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _composite_one(out, shape_mask, size, rng):
    """Shade one object and composite it, with its own colour and highlight."""
    obj = _shaded_object_color(size, rng).astype(np.float32)
    ys, xs = np.nonzero(shape_mask)
    if len(ys):
        cy, cx = ys.mean(), xs.mean()
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius /= max(1e-6, radius[shape_mask > 0].max())
        obj *= (1.0 - 0.35 * np.clip(radius, 0, 1))[..., None]
        hy, hx = cy - size * 0.12, cx - size * 0.12
        highlight = np.exp(-(((yy - hy) ** 2 + (xx - hx) ** 2) / (2 * (size * 0.13) ** 2)))
        obj = obj + 70.0 * highlight[..., None]
    alpha = shape_mask.astype(np.float32)[..., None]
    return out * (1 - alpha) + np.clip(obj, 0, 255) * alpha


def _apply_shape(background, shape_masks, size, rng):
    """Composite shaded, shadowed objects into the background.

    Objects are composited one at a time, each with its own hue and highlight.
    A single colour field shared by all objects lets adjacent objects merge into
    one blob under the saturation key, which caps the count metric at 45% even
    on ground-truth renders -- a metric failure rather than a model failure.
    """
    if isinstance(shape_masks, np.ndarray):
        shape_masks = [shape_masks]

    union = np.clip(np.sum(shape_masks, axis=0), 0, 1).astype(np.float32)
    shadow = cv2.GaussianBlur(union, (0, 0), max(1.0, size * 0.03))
    offset = int(round(size * 0.035))
    shadow = np.roll(np.roll(shadow, offset, axis=0), offset, axis=1)
    out = background.astype(np.float32) * (1.0 - 0.45 * shadow[..., None])

    for mask in shape_masks:
        out = _composite_one(out, mask, size, rng)

    # Blur kept modest: heavier blur bridges the 2 px gaps between neighbouring
    # objects and merges them before the geometry is ever measured.
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.6))
    out = np.clip(out * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12), 0, 255)
    out = np.clip(out + rng.normal(0, 3.0, out.shape), 0, 255)
    return out.astype(np.uint8)


def _mask_from_draw(size, draw_fn, antialias=4):
    """Rasterise a shape at high resolution and downsample to a soft mask."""
    big = Image.new("L", (size * antialias, size * antialias), 0)
    draw_fn(ImageDraw.Draw(big), antialias)
    small = big.resize((size, size), Image.Resampling.LANCZOS)
    return (np.asarray(small, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)


def render_circle(radius, cx, cy, size, rng):
    def draw(d, a):
        r, x, y = radius * a, size * a / 2 + cx * a, size * a / 2 + cy * a
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return _apply_shape(_background(size, rng), _mask_from_draw(size, draw), size, rng)


def render_circles(centers, radius, size, rng):
    masks = []
    for cx, cy in centers:
        def draw(d, a, cx=cx, cy=cy):
            r, x, y = radius * a, size * a / 2 + cx * a, size * a / 2 + cy * a
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        masks.append(_mask_from_draw(size, draw))
    return _apply_shape(_background(size, rng), masks, size, rng)


def circle_mask(radius, cx, cy, size):
    """The exact mask the renderer rasterises, for use as evaluation ground truth."""
    def draw(d, a):
        r, x, y = radius * a, size * a / 2 + cx * a, size * a / 2 + cy * a
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return _mask_from_draw(size, draw)


ARROW_POINTS = [
    (0.0, -0.5), (-0.25, -0.2), (-0.15, -0.2), (-0.10, 0.4),
    (0.0, 0.5), (0.10, 0.4), (0.15, -0.2), (0.25, -0.2),
]


def render_arrow(angle_degrees, size, shape_size, rng):
    def draw(d, a):
        rad = math.radians(-angle_degrees + 90.0)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cx = cy = size * a / 2
        pts = []
        for px, py in ARROW_POINTS:
            x, y = px * shape_size * a, py * shape_size * a
            pts.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
        d.polygon(pts, fill=255)
    return _apply_shape(_background(size, rng), _mask_from_draw(size, draw), size, rng)


def object_mask(image_rgb):
    """Recover the object mask from a complex render by saturation keying.

    OPEN removes specks; there is deliberately no CLOSE.  Neighbouring objects
    sit only ~2 px apart, and a 3x3 closing bridges that gap and merges them,
    dropping the count ceiling on ground-truth renders to 43%.  Without it the
    ceiling is 100% and single-object IoU is unchanged.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] > SATURATION_KEY).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def validate_extraction(n=200, size=64, seed=0):
    """Check the extractor recovers known geometry from the complex renders."""
    rng = np.random.default_rng(seed)
    ious, pos_errors, radius_errors = [], [], []
    for _ in range(n):
        radius = float(rng.integers(5, 21))
        cx, cy = float(rng.integers(-12, 13)), float(rng.integers(-12, 13))

        def draw(d, a):
            r, x, y = radius * a, size * a / 2 + cx * a, size * a / 2 + cy * a
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)

        truth = _mask_from_draw(size, draw)
        image = _apply_shape(_background(size, rng), truth, size, rng)
        found = object_mask(image)

        inter = np.logical_and(truth > 0, found > 0).sum()
        union = np.logical_or(truth > 0, found > 0).sum()
        ious.append(inter / union if union else 0.0)

        ys, xs = np.nonzero(found)
        if len(ys):
            pos_errors.append(float(np.hypot(xs.mean() - (size / 2 + cx),
                                             ys.mean() - (size / 2 + cy))))
            radius_errors.append(abs(math.sqrt(found.sum() / math.pi) - radius))
    return {
        "mean_iou": float(np.mean(ious)),
        "pct_iou_above_0.90": 100.0 * float(np.mean([i >= 0.90 for i in ious])),
        "mean_centroid_error_px": float(np.mean(pos_errors)),
        "mean_radius_error_px": float(np.mean(radius_errors)),
    }
