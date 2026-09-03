"""Eight-shape vocabulary for the coverage-vs-K experiment.

The paper uses four shapes (circle, square, triangle, diamond).  With a fixed
vocabulary the coverage fraction and the absolute number of training
combinations K are locked together, so a coverage effect cannot be told apart
from a number-of-unique-combinations effect.  Separating the two needs a
larger vocabulary, so pentagon, hexagon, star and cross are added.

`render_shape` reproduces `generate_compositional_dataset_coverage.generate_shape_image`
bit-for-bit on the original four shapes (verified in validate_classifier.py) and
extends the same drawing conventions to the new four.

`classify_shape` is the multi-scale template matcher already used by
eval_compositional.py, generalised over whichever vocabulary is passed in.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

SHAPES4 = ["circle", "square", "triangle", "diamond"]
SHAPES8 = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon", "star", "cross"]

COLORS = [
    ("red", (255, 0, 0)),
    ("blue", (0, 0, 255)),
    ("green", (0, 255, 0)),
    ("yellow", (255, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("orange", (255, 128, 0)),
    ("purple", (128, 0, 255)),
]

# Eight further colours, so the colour axis can be doubled the way the shape
# axis is.  With both vocabularies variable, coverage fraction and the absolute
# number of training combinations K stop being locked together and their
# separate contributions can be estimated.  Minimum pairwise separation is 64
# in RGB (brown vs maroon), far above the noise floor of the mean-colour
# readout used by classify_color.
COLORS16 = COLORS + [
    ("lime", (128, 255, 0)),
    ("teal", (0, 128, 128)),
    ("pink", (255, 128, 192)),
    ("brown", (128, 64, 0)),
    ("navy", (0, 0, 128)),
    ("maroon", (128, 0, 0)),
    ("olive", (128, 128, 0)),
    ("azure", (0, 128, 255)),
]


def _regular_polygon(cx, cy, r, n_sides, start_angle=-math.pi / 2):
    return [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]


def _star(cx, cy, r, n_points=5, inner_ratio=0.42, start_angle=-math.pi / 2):
    points = []
    for i in range(2 * n_points):
        radius = r if i % 2 == 0 else r * inner_ratio
        angle = start_angle + math.pi * i / n_points
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _cross(cx, cy, r, arm_ratio=0.36):
    w = r * arm_ratio
    return [
        (cx - w, cy - r), (cx + w, cy - r), (cx + w, cy - w), (cx + r, cy - w),
        (cx + r, cy + w), (cx + w, cy + w), (cx + w, cy + r), (cx - w, cy + r),
        (cx - w, cy + w), (cx - r, cy + w), (cx - r, cy - w), (cx - w, cy - w),
    ]


def render_shape(shape, radius, color_rgb, image_size=64, position=(0, 0),
                 background_color=(255, 255, 255), antialiasing=4):
    """Render one filled shape centred at `position` on a blank canvas."""
    large_size = image_size * antialiasing
    img = Image.new("RGB", (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)

    r = radius * antialiasing
    cx = large_size // 2 + position[0] * antialiasing
    cy = large_size // 2 + position[1] * antialiasing

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color_rgb, outline=color_rgb)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color_rgb, outline=color_rgb)
    elif shape == "triangle":
        height = r * 1.732
        draw.polygon(
            [(cx, cy - height * 2 / 3), (cx - r, cy + height / 3), (cx + r, cy + height / 3)],
            fill=color_rgb, outline=color_rgb,
        )
    elif shape == "diamond":
        draw.polygon(
            [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
            fill=color_rgb, outline=color_rgb,
        )
    elif shape == "pentagon":
        draw.polygon(_regular_polygon(cx, cy, r, 5), fill=color_rgb, outline=color_rgb)
    elif shape == "hexagon":
        draw.polygon(_regular_polygon(cx, cy, r, 6), fill=color_rgb, outline=color_rgb)
    elif shape == "star":
        draw.polygon(_star(cx, cy, r), fill=color_rgb, outline=color_rgb)
    elif shape == "cross":
        draw.polygon(_cross(cx, cy, r), fill=color_rgb, outline=color_rgb)
    else:
        raise ValueError(f"unknown shape {shape!r}")

    return img.resize((image_size, image_size), Image.Resampling.LANCZOS)


def foreground_mask(image_np, threshold=30):
    """Binary mask of non-background pixels (background is near-white)."""
    white = np.array([255, 255, 255], dtype=np.float32)
    diff = np.linalg.norm(image_np.astype(np.float32) - white, axis=2)
    return (diff > threshold).astype(np.uint8)


_TEMPLATE_CACHE = {}

# Template radii.  Fine and wide so a generation whose size drifted still finds
# a same-shape template rather than a different-shape one at a better scale.
TEMPLATE_SIZES = tuple(float(s) for s in np.arange(5.0, 22.0, 0.5))


def _center_subpixel(mask, image_size):
    """Translate a binary mask so its centroid sits at the exact image centre.

    Integer-pixel centring leaves up to ~1 px of residual offset, which costs
    ~0.13 IoU on a 64x64 blob -- enough to make a circle template outscore the
    correct pentagon/hexagon one.  Sub-pixel translation lifts self-match IoU
    from ~0.87 to ~0.99 and makes the vocabulary separable.
    """
    import cv2

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    target = (image_size - 1) / 2.0
    shift = np.float32([[1, 0, target - xs.mean()], [0, 1, target - ys.mean()]])
    warped = cv2.warpAffine(
        mask.astype(np.float32), shift, (image_size, image_size), flags=cv2.INTER_LINEAR
    )
    return (warped > 0.5).astype(np.uint8)


def _templates(vocabulary, image_size, sizes):
    key = (tuple(vocabulary), image_size, tuple(sizes))
    if key not in _TEMPLATE_CACHE:
        built = []
        for shape in vocabulary:
            for size in sizes:
                img = render_shape(shape, size, (255, 0, 0), image_size=image_size)
                mask = foreground_mask(np.array(img))
                built.append((shape, _center_subpixel(mask, image_size)))
        _TEMPLATE_CACHE[key] = built
    return _TEMPLATE_CACHE[key]


def classify_shape(image_np, vocabulary, image_size=64, sizes=TEMPLATE_SIZES):
    """Multi-scale template matching, sub-pixel centroid-aligned.

    Returns (shape, iou).  100% on ground-truth renders for both the 4-shape
    and 8-shape vocabularies (see validate_classifier.py).
    """
    mask = foreground_mask(image_np)
    if mask.sum() < 10:
        return None, 0.0

    centered = _center_subpixel(mask, image_size)

    best_shape, best_iou = None, -1.0
    for shape, template in _templates(vocabulary, image_size, sizes):
        inter = np.logical_and(centered > 0, template > 0).sum()
        union = np.logical_or(centered > 0, template > 0).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_shape = iou, shape
    return best_shape, float(best_iou)


def classify_color(image_np, palette=COLORS):
    """Nearest-colour assignment over the mean *interior* foreground colour.

    Antialiased boundary pixels are blends of the fill and the white ground, so
    averaging over the whole mask drags every colour toward white and makes
    near-neighbours in the 16-colour palette (maroon vs brown) collide.  Eroding
    the mask first keeps only interior pixels.
    """
    import cv2

    mask = foreground_mask(image_np)
    if mask.sum() < 10:
        return None
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    if interior.sum() >= 10:
        mask = interior
    mean_rgb = image_np[mask > 0].astype(np.float32).mean(axis=0)
    best_name, best_dist = None, float("inf")
    for name, rgb in palette:
        dist = float(np.linalg.norm(mean_rgb - np.array(rgb, dtype=np.float32)))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name
