#!/usr/bin/env python
"""Per-sample scorer for the two-skill compositional experiments (paper Table 3), version 2.

Every generated image is scored on its own against the skill values it was conditioned on.
A sample with no object fails every skill, and accuracy is the fraction of samples (not of
combinations). Ground-truth masks come from the dataset renderer
(scripts/generate_compositional_dataset_coverage.generate_shape_image in the release repo).

Detection and the object
  An image has an object when more than DETECT_MIN_PIXELS pixels differ from the background colour
  (median of the border pixels) by more than DETECT_MIN_DIST in RGB distance; otherwise it is a
  detection failure. The object mask is the Otsu threshold of that RGB distance. In pairs without
  count, "the object" is the largest 8-connected component of the mask (stray specks are ignored);
  in count pairs every component of at least COUNT_MIN_AREA px is an object.

  size      IoU >= 0.90 between the object mask and the mask of the REQUESTED shape at the requested
            size, rendered by the dataset generator at the object's centroid (best of a 1/4-px
            translation grid within +-0.5 px), so size is scored independently of position. Masks get
            a 3x3 close and open, analogous to the single-skill size metric (vgl/eval_radius), which
            compares against a circle at the canvas centre without alignment.
  position  Euclidean distance <= 2 px between the centre of mass of the object and the centre of
            mass of the ground-truth render of the requested position (requested shape and size; for
            the rotated object, at the detected angle, because the arrow's centroid is not its rotation
            centre). The Appendix A.4 / single-skill rule (centroid minus the canvas centre (32, 32))
            is returned as position_ok_a4; it carries a renderer offset of 0.5-0.6 px for the four
            shapes and up to 1.35 px for the arrow.
  rotation  the sample is translated so the object's centroid sits at a fixed point and correlated with
            a 1-degree bank of templates drawn by the dataset renderer (template centroids at the same
            point); circular error <= 5 deg (the threshold of Appendix A.4). The error is taken modulo
            the object's symmetry period: 360 deg for the arrow, 120 deg for the equilateral triangle
            of the older datasets. A 2 deg variant is also returned. The rotated shape comes from the
            dataset metadata (or is passed explicitly); there is no default.
  count     exact match. Objects are the connected components of the mask with >= 30 px (the
            single-skill area cut). Touching objects are split at necks: flooding the distance transform
            (the inscribed radius) from its peaks, two parts are separate objects where the radius at the
            saddle between them is at most COUNT_PEAK_REL (0.75) of the smaller part's peak, and that peak
            is >= COUNT_MIN_PEAK. A convex object (one peak) is never split; a fused bar with no neck is
            one object (and then fails shape). The ratio was chosen on synthetic rows, bars and single
            objects (studies.py count_rule) before any Table 3 cell was scored; the single-skill
            watershed (detect_count_with_watershed, returned as count_ok_single_skill) blurs and closes
            the mask first, which merges objects 2.5 px apart.
  shape     each object (the object, or every counted component in count pairs) is compared, centroid
            aligned and at the size implied by its area (x0.94..1.06), with the circle and with the
            square and triangle rendered at every orientation (5-deg search refined to 1 deg). An object
            whose second-moment elongation exceeds ELONG_MAX is none of the four (all four shapes are
            isotropic), as is one whose best IoU is below 0.75. Otherwise the best fit decides: circle;
            a square-like polygon within ANGLE_TOL of axis-aligned is a square and within ANGLE_TOL of
            45 deg a diamond; a triangle within ANGLE_TOL of upright is a triangle; any other
            orientation is none. Success if every object is the requested shape. "vertex" (A.4's
            description: Douglas-Peucker vertex count) is kept as a diagnostic.
  color     mean RGB inside the object (in count pairs, the whole mask), nearest of the 8 palette
            colours by L2 distance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

RELEASE = Path(__file__).resolve().parents[2]  # repo root
if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))
from scripts.generate_compositional_dataset_coverage import generate_shape_image  # noqa: E402

IMAGE_SIZE = 64
SIZE_IOU_THRESHOLD = 0.90
POSITION_THRESHOLD_PX = 2.0
ROTATION_THRESHOLD_DEG = 5.0
ROTATION_STRICT_DEG = 2.0
DETECT_MIN_DIST = 60.0     # RGB distance from the background colour
DETECT_MIN_PIXELS = 10     # more than this many pixels above DETECT_MIN_DIST
MIN_OBJECT_PIXELS = 10     # eval_position.extract_circle_center_debug: valid if mask sum > 10
MIN_COMPONENT_PIXELS = 10  # the object, pairs without count
COUNT_MIN_AREA = 30        # paper_runs/table2/evaluate_table2.COUNT_MIN_AREA
COUNT_PEAK_REL = 0.75      # neck ratio, see _count_necks (studies.py count_rule)
COUNT_MIN_PEAK = 2.0       # px; a part narrower than ~4 px is smaller than any object the 30-px cut keeps
TEMPLATE_SCALES = (0.94, 0.97, 1.0, 1.03, 1.06)
SHAPE_MIN_IOU = 0.75
ELONG_MAX = 1.25           # GT max 1.115 (studies.py shape_rot)
ANGLE_TOL = 15.0           # deg from a shape's canonical orientation
ANGLE_COARSE = 5           # deg, then refined in 1-deg steps around the best coarse angle
DP_EPSILON = 0.04          # Douglas-Peucker tolerance (fraction of perimeter), vertex diagnostic
UPSAMPLE = 4

MASK_MODE = os.environ.get("NEW_EVAL_MASK_MODE", "rgb")           # "rgb" (default) or "gray" (single-skill)
SHAPE_METHOD = os.environ.get("NEW_EVAL_SHAPE_METHOD", "template")  # "template" or "vertex"

SHAPES = ("circle", "square", "triangle", "diamond")
COLORS = (("red", (255, 0, 0)), ("blue", (0, 0, 255)), ("green", (0, 255, 0)), ("yellow", (255, 255, 0)),
          ("magenta", (255, 0, 255)), ("cyan", (0, 255, 255)), ("orange", (255, 128, 0)), ("purple", (128, 0, 255)))
COLOR_NAMES = tuple(n for n, _ in COLORS)
COLOR_RGB = dict(COLORS)
PALETTE = np.array([rgb for _, rgb in COLORS], dtype=np.float64)
AREA_PER_R2 = {"circle": math.pi, "square": 4.0, "triangle": math.sqrt(3.0), "diamond": 2.0}

# Table 3 pairs: property order as in paper_runs/table3/manifest.SKILL_PAIRS (test_prop_a_ood = first).
PAIRS = {
    "color_shape": (["shape", "color"], "shape_color"),
    "color_count": (["count", "color"], "count_color"),
    "shape_count": (["shape", "count"], "shape_count"),
    "color_position": (["position", "color"], "position_color"),
    "position_size": (["radius", "position"], "radius_position"),
    "position_shape": (["position", "shape"], "position_shape"),
    "size_shape": (["radius", "shape"], "radius_shape"),
    "position_rotation": (["position", "rotation"], "position_rotation"),
}
SPLITS = ("train", "test_compositional", "test_prop_a_ood", "test_prop_b_ood", "test_both_ood")


# ----------------------------------------------------------------------------------------- masks
def bg_distance(img: np.ndarray) -> np.ndarray:
    """Per-pixel RGB distance to the background colour (median of the border pixels)."""
    f = img.astype(np.float32)
    border = np.concatenate([f[0], f[-1], f[1:-1, 0], f[1:-1, -1]])
    return np.linalg.norm(f - np.median(border, axis=0), axis=2)


def has_object(img: np.ndarray) -> bool:
    return int((bg_distance(img) > DETECT_MIN_DIST).sum()) > DETECT_MIN_PIXELS


def otsu(img: np.ndarray, mode: str | None = None):
    """Object mask (uint8 0/1) plus what is needed to re-threshold: (mask, channel, threshold, object_is_low).

    mode "gray" is exactly the single-skill rule (grayscale Otsu, invert when more than half the
    pixels come out white). mode "rgb" thresholds the RGB distance to the background colour with
    Otsu instead, which does not depend on the object being darker than the background in grayscale.
    """
    mode = mode or MASK_MODE
    if mode == "gray":
        ch = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        d = bg_distance(img)
        ch = np.clip(255.0 * d / max(float(d.max()), 1.0), 0, 255).astype(np.uint8)
    t, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = bool(np.mean(m) > 127)
    if inverted:
        m = 255 - m
    # object pixels are ch <= t when the white class was inverted away, ch > t otherwise
    return (m > 127).astype(np.uint8), ch, float(t), inverted


def clean(mask: np.ndarray) -> np.ndarray:
    """3x3 close then open, as vgl/eval_radius.extract_circle_mask."""
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)


def components(mask: np.ndarray, min_area: int = MIN_COMPONENT_PIXELS) -> list[np.ndarray]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [(lab == i).astype(np.uint8) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return sorted(comps, key=lambda c: -int(c.sum()))


def largest(mask: np.ndarray, min_area: int = MIN_COMPONENT_PIXELS):
    comps = components(mask, min_area)
    return comps[0] if comps else None


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def elongation(mask: np.ndarray) -> float:
    """sqrt of the ratio of the second-moment eigenvalues (1 for the four dataset shapes at any angle)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return float("inf")
    cov = np.cov(np.stack([xs, ys]).astype(np.float64)) + np.eye(2) / 12.0  # pixel-area term
    ev = np.linalg.eigvalsh(cov)
    return float(math.sqrt(ev[1] / max(ev[0], 1e-9)))


# ------------------------------------------------------------------------------- GT rendering
def render(shape, radius, pos, rotation=0.0, color=(255, 0, 0), count=1, count_active=False):
    img = generate_shape_image(float(radius), (float(pos[0]), float(pos[1])), shape, tuple(int(c) for c in color),
                               image_size=IMAGE_SIZE, background_color=(255, 255, 255), rotation=float(rotation),
                               count=int(count), count_is_active_property=count_active)
    return np.asarray(img)


_OFFSET: dict = {}


def _offset(shape):
    """Mask centroid (pixel-index coordinates) of an object rendered at position (0, 0)."""
    if shape not in _OFFSET:
        _OFFSET[shape] = centroid(otsu(render(shape, 10.0, (0.0, 0.0)), "gray")[0])
    return _OFFSET[shape]


def render_mask_at(shape, radius, target_xy, rotation=0.0, color=(255, 0, 0)):
    """GT mask of `shape` placed so that its mask centroid sits at target_xy (calibrated offset; the
    square and triangle rotate about their centroid, so the offset holds at every angle)."""
    ox, oy = _offset(shape)
    return otsu(render(shape, radius, (target_xy[0] - ox, target_xy[1] - oy), rotation, color))[0]


ALIGN_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)  # px, around the centroid-matched placement


def aligned_iou(mask: np.ndarray, shape, radius, rotation=0.0, color=(255, 0, 0)) -> float:
    """Size IoU: max over a 1/4-px translation grid (+-0.5 px) around the placement that matches the
    mask centroids, of IoU(mask, cleaned mask of the GT render). Centroid matching alone is unstable
    for small polygons (a sub-pixel shift adds or removes a whole row at an apex)."""
    c = centroid(mask)
    ox, oy = _offset(shape)
    best = 0.0
    for dy in ALIGN_GRID:
        for dx in ALIGN_GRID:
            g = clean(otsu(render(shape, radius, (c[0] - ox + dx, c[1] - oy + dy), rotation, color))[0])
            best = max(best, iou(mask, g))
    return best


# ------------------------------------------------------------------------------------- shape
def _polygon_fit(obj, c, area, shape, period, color):
    """Best IoU over orientation (coarse ANGLE_COARSE grid, then 1-deg refinement with the scale grid)."""
    r0 = math.sqrt(area / AREA_PER_R2[shape])
    best_s, best_a = -1.0, 0
    for a in range(0, period, ANGLE_COARSE):
        s = iou(obj, render_mask_at(shape, r0, c, rotation=a, color=color))
        if s > best_s:
            best_s, best_a = s, a
    a0 = best_a
    for a in range(a0 - ANGLE_COARSE + 1, a0 + ANGLE_COARSE):
        for f in TEMPLATE_SCALES:
            s = iou(obj, render_mask_at(shape, r0 * f, c, rotation=a % period, color=color))
            if s > best_s:
                best_s, best_a = s, a % period
    return best_s, best_a


def shape_fit(obj: np.ndarray, color=(255, 0, 0)) -> dict:
    """Label of one object plus the fit details (IoUs, best angles, elongation)."""
    area, c = float(obj.sum()), centroid(obj)
    circ = max(iou(obj, render_mask_at("circle", math.sqrt(area / math.pi) * f, c, color=color)) for f in TEMPLATE_SCALES)
    sq, sq_a = _polygon_fit(obj, c, area, "square", 90, color)
    tri, tri_a = _polygon_fit(obj, c, area, "triangle", 120, color)
    el = elongation(obj)
    info = {"iou_circle": circ, "iou_square_family": sq, "angle_square_family": sq_a,
            "iou_triangle": tri, "angle_triangle": tri_a, "elongation": el}
    best = max(circ, sq, tri)
    if el > ELONG_MAX or best < SHAPE_MIN_IOU:
        label = "none"
    elif best == circ:
        label = "circle"
    elif best == sq:
        d0 = min(sq_a % 90, 90 - sq_a % 90)
        label = "square" if d0 <= ANGLE_TOL else "diamond" if abs(sq_a % 90 - 45) <= ANGLE_TOL else "none"
    else:
        d0 = min(tri_a % 120, 120 - tri_a % 120)
        label = "triangle" if d0 <= ANGLE_TOL else "none"
    info["label"] = label
    info["best_iou"] = best
    return info


def classify_shape_template(obj: np.ndarray, color=(255, 0, 0)):
    """(label, IoU per template family)."""
    f = shape_fit(obj, color)
    return f["label"], {"circle": f["iou_circle"], "square_family": f["iou_square_family"], "triangle": f["iou_triangle"]}


def classify_shape_vertex(obj: np.ndarray, ch: np.ndarray, thr: float, object_is_low: bool, eps: float | None = None):
    ys, xs = np.nonzero(obj)
    y0, y1 = max(ys.min() - 3, 0), min(ys.max() + 4, ch.shape[0])
    x0, x1 = max(xs.min() - 3, 0), min(xs.max() + 4, ch.shape[1])
    patch = cv2.resize(ch[y0:y1, x0:x1].astype(np.float32), None, fx=UPSAMPLE, fy=UPSAMPLE,
                       interpolation=cv2.INTER_CUBIC)
    up = ((patch <= thr) if object_is_low else (patch > thr)).astype(np.uint8)
    keep = cv2.resize(obj[y0:y1, x0:x1], None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_NEAREST)
    keep = cv2.dilate(keep, np.ones((2 * UPSAMPLE + 1, 2 * UPSAMPLE + 1), np.uint8))
    up = up & keep
    contours, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return "none", {"vertices": 0}
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, (eps or DP_EPSILON) * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(float)
    n = len(approx)
    info = {"vertices": n}
    if n == 3:
        return "triangle", info
    if n == 4:
        e = np.roll(approx, -1, axis=0) - approx
        th = np.arctan2(e[:, 1], e[:, 0])
        phi = math.degrees(math.atan2(np.sin(4 * th).sum(), np.cos(4 * th).sum()) / 4.0)  # (-45, 45]
        info["edge_angle"] = phi
        return ("square" if abs(phi) < 22.5 else "diamond"), info
    return "circle", info


# ---------------------------------------------------------------------------------- rotation
class RotationBank:
    """1-degree template bank for one rotated object; templates and samples are centroid aligned."""

    def __init__(self, shape: str, radius: float = 10.0):
        self.shape = shape
        self.period = 120 if shape == "triangle" else 360
        self.anchor = (31.5, 31.5)
        feats = []
        for a in range(self.period):
            ox, oy = _offset(shape)
            pos = [self.anchor[0] - ox, self.anchor[1] - oy]
            for _ in range(3):  # the arrow's centroid is not its rotation centre: re-render until aligned
                img = render(shape, radius, pos, rotation=a)
                c = centroid(otsu(img, "gray")[0])
                dx, dy = self.anchor[0] - c[0], self.anchor[1] - c[1]
                if abs(dx) < 0.03 and abs(dy) < 0.03:
                    break
                pos = [pos[0] + dx, pos[1] + dy]
            feats.append(self._feature(255.0 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)))
        self.bank = np.stack(feats)

    @staticmethod
    def _feature(fg: np.ndarray) -> np.ndarray:
        v = fg.reshape(-1).astype(np.float64)
        v = v - v.mean()
        return v / max(np.linalg.norm(v), 1e-9)

    def detect(self, mask: np.ndarray, ch: np.ndarray, low: bool):
        c = centroid(mask)  # object centroid, as for the templates
        if c is None:
            return None
        fg = (255.0 - ch.astype(np.float32)) if low else ch.astype(np.float32)
        fg = fg - np.median(fg)  # background ~0 so the warp border does not create an edge
        M = np.float32([[1, 0, self.anchor[0] - c[0]], [0, 1, self.anchor[1] - c[1]]])
        fg = cv2.warpAffine(fg, M, (fg.shape[1], fg.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        scores = self.bank @ self._feature(fg)
        return float(int(np.argmax(scores)))

    def error(self, detected: float, expected: float) -> float:
        p = self.period
        return abs(((detected - expected + p / 2.0) % p) - p / 2.0)


_BANKS: dict = {}


def rotation_bank(shape):
    if shape not in _BANKS:
        _BANKS[shape] = RotationBank(shape)
    return _BANKS[shape]


# ------------------------------------------------------------------------------------- count
def detect_count_with_watershed(image: np.ndarray, min_area: int = COUNT_MIN_AREA) -> int:
    """Verbatim copy of paper_runs/table2/evaluate_table2.detect_count_with_watershed (single-skill count)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if binary.max() == 0:
        return 0

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return 0

    markers = markers + 1
    markers[unknown == 255] = 0
    watershed_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_input, markers)

    count = 0
    for label in np.unique(markers):
        if label <= 1:
            continue
        if int(np.count_nonzero(markers == label)) > min_area:
            count += 1
    return count


def _count_necks(dist: np.ndarray, comp: np.ndarray, rel: float) -> int:
    """Parts of one component separated by necks. The distance transform (inscribed radius) is flooded from
    its peaks downwards (union-find, 8-connectivity); when two parts meet at level v, the lower part (peak p)
    is a separate object if v <= rel * p, i.e. the neck is at most rel times that part's inscribed radius,
    and p >= COUNT_MIN_PEAK. A convex object has one peak and is never split."""
    h, w = dist.shape
    flat = dist.ravel()
    idx = np.flatnonzero(comp.ravel())
    order = idx[np.argsort(-flat[idx], kind="stable")]
    parent = np.full(h * w, -1, dtype=np.int64)
    peak = np.zeros(h * w)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    extra = 0
    for p in order:
        v = flat[p]
        parent[p], peak[p] = p, v
        y, x = divmod(int(p), w)
        roots = {find(yy * w + xx) for yy in range(max(y - 1, 0), min(y + 2, h)) for xx in range(max(x - 1, 0), min(x + 2, w))
                 if (yy != y or xx != x) and parent[yy * w + xx] >= 0}
        if not roots:
            continue
        roots = sorted(roots, key=lambda r: -peak[r])
        top = roots[0]
        parent[p] = top
        for r in roots[1:]:
            if peak[r] >= COUNT_MIN_PEAK and v <= rel * peak[r]:
                extra += 1
            parent[r] = top
    return 1 + extra


def count_objects(mask: np.ndarray, rel: float | None = None, min_area: int = COUNT_MIN_AREA,
                  rule: str = "neck", per_component: bool = True) -> int:
    """Connected components of the mask with >= min_area px, each split at necks (rule "neck", the
    evaluator's rule, see _count_necks). rule "threshold" counts, within each component, the regions where
    the distance transform exceeds rel x the component's maximum (per_component) or x the image maximum;
    rule "threshold", rel=0.35, per_component=False is version 1."""
    rel = COUNT_PEAK_REL if rel is None else rel
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0
    dist = cv2.distanceTransform(m * 255, cv2.DIST_L2, 5)  # components are separated by background
    gmax = float(dist.max())
    total = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = lab == i
        if rule == "neck":
            total += _count_necks(dist, comp, rel)
            continue
        t = rel * (float(dist[comp].max()) if per_component else gmax)
        k, _ = cv2.connectedComponents(((dist > t) & comp).astype(np.uint8))
        total += max(1, k - 1)
    return total


_REF: dict = {}


def reference_centroid(shape, radius, pos, rotation):
    """Centre of mass of the ground-truth render of the requested position (pixel-index coordinates)."""
    key = (shape, round(float(radius), 3), round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(rotation), 1))
    if key not in _REF:
        m = otsu(render(shape, radius, pos, rotation))[0]
        cy, cx = ndimage.center_of_mass(m.astype(np.float32))
        _REF[key] = (float(cx), float(cy))
    return _REF[key]


# ------------------------------------------------------------------------------------ scoring
def context(props, metadata=None, rotation_shape=None):
    """What the renderer draws for skills that are not varied (generate_dataset_split defaults). The rotated
    shape must come from the dataset metadata or be passed explicitly; the two must agree."""
    metadata = metadata or {}
    meta_shape = metadata.get("rotation_shape")
    if rotation_shape and meta_shape and rotation_shape != meta_shape:
        raise ValueError(f"rotation shape {rotation_shape!r} disagrees with dataset metadata {meta_shape!r}")
    rot_shape = rotation_shape or meta_shape
    if "rotation" in props and "shape" not in props and rot_shape not in ("arrow", "triangle"):
        raise ValueError("rotation pair needs the rotated shape (dataset metadata 'rotation_shape' or rotation_shape=)")
    return {"props": list(props), "rotation_shape": rot_shape, "count_active": "count" in props}


def score_sample(img: np.ndarray, cond: dict, ctx: dict, shape_method: str | None = None) -> dict:
    """Score one uint8 RGB image. cond maps each skill to its requested value
    (radius float, position (x, y) px, shape name, color name, count int, rotation deg)."""
    shape_method = shape_method or SHAPE_METHOD
    props = ctx["props"]
    rec = {"detected": False}
    for p in props:
        rec[f"{p}_ok"] = False
    if "count" in props:
        rec["count_det"] = 0

    if not has_object(img):
        return rec
    raw, ch, thr, low = otsu(img)
    if int(raw.sum()) <= MIN_OBJECT_PIXELS:
        return rec
    if "count" in props:
        objs = components(raw, COUNT_MIN_AREA)
        whole = raw
    else:
        obj = largest(raw)
        objs = [obj] if obj is not None else []
        whole = obj
    if not objs:
        return rec
    rec["detected"] = True
    shape_req = cond.get("shape", ctx["rotation_shape"] or "circle")
    color_req = COLOR_RGB[cond.get("color", "red")]

    if "radius" in props:
        m = largest(clean(raw))
        if m is not None:
            rec["radius_iou"] = aligned_iou(m, shape_req, cond["radius"], color=color_req)
            rec["radius_ok"] = rec["radius_iou"] >= SIZE_IOU_THRESHOLD

    if "rotation" in props:
        bank = rotation_bank(ctx["rotation_shape"])
        det = bank.detect(whole, ch, low)
        if det is not None:
            err = bank.error(det, float(cond["rotation"]))
            rec["rotation_det"], rec["rotation_err"] = det, err
            rec["rotation_ok"] = err <= ROTATION_THRESHOLD_DEG
            rec["rotation_ok_2deg"] = err <= ROTATION_STRICT_DEG

    if "position" in props:
        cy, cx = ndimage.center_of_mass(whole.astype(np.float32))
        dx, dy = cx - IMAGE_SIZE / 2.0, cy - IMAGE_SIZE / 2.0
        rec["position_det"] = (float(dx), float(dy))
        rec["position_err_a4"] = float(math.hypot(dx - cond["position"][0], dy - cond["position"][1]))
        rec["position_ok_a4"] = rec["position_err_a4"] <= POSITION_THRESHOLD_PX
        ref_rot = rec.get("rotation_det", cond.get("rotation", 0.0)) if "rotation" in props else 0.0
        rx, ry = reference_centroid(shape_req, cond.get("radius", 5.0 if ctx["count_active"] else 10.0),
                                    cond["position"], ref_rot)
        rec["position_err"] = float(math.hypot(cx - rx, cy - ry))
        rec["position_ok"] = rec["position_err"] <= POSITION_THRESHOLD_PX

    if "shape" in props:
        if shape_method in ("template", "both"):
            fits = [shape_fit(o, color_req) for o in objs]
            det_t = [f["label"] for f in fits]
            rec["shape_det_template"] = det_t
            rec["shape_iou_template"] = [float(f["best_iou"]) for f in fits]
            rec["shape_elongation"] = [float(f["elongation"]) for f in fits]
            rec["shape_ok_template"] = bool(det_t) and all(s == shape_req for s in det_t)
        if shape_method in ("vertex", "both"):
            det_v = [classify_shape_vertex(o, ch, thr, low)[0] for o in objs]
            rec["shape_det_vertex"] = det_v
            rec["shape_ok_vertex"] = bool(det_v) and all(s == shape_req for s in det_v)
        rec["shape_ok"] = rec["shape_ok_template"] if "shape_ok_template" in rec else rec["shape_ok_vertex"]

    if "color" in props:
        mean = img[whole > 0].reshape(-1, 3).astype(np.float64).mean(axis=0)
        cid = int(np.argmin(np.linalg.norm(PALETTE - mean, axis=1)))
        rec["color_det"] = COLOR_NAMES[cid]
        rec["color_mean"] = [float(v) for v in mean]
        rec["color_ok"] = COLOR_NAMES[cid] == cond["color"]

    if "count" in props:
        rec["count_det"] = count_objects(raw)
        rec["count_ok"] = rec["count_det"] == int(cond["count"])
        rec["count_det_single_skill"] = int(detect_count_with_watershed(img))
        rec["count_ok_single_skill"] = rec["count_det_single_skill"] == int(cond["count"])
    return rec


def summarize(records: list[dict], props: list[str]) -> dict:
    """records: dicts with 'combo' (index) plus score_sample output. Accuracy = fraction of samples."""
    n = len(records)
    if n == 0:
        return {"num_samples": 0}
    out = {"num_samples": n, "num_combinations": len({r["combo"] for r in records}),
           "detection_rate": 100.0 * np.mean([r["detected"] for r in records]),
           "accuracy": {p: 100.0 * np.mean([bool(r[f"{p}_ok"]) for r in records]) for p in props},
           "joint_accuracy": 100.0 * np.mean([all(bool(r[f"{p}_ok"]) for p in props) for r in records])}
    if "rotation" in props:
        out["accuracy_rotation_2deg"] = 100.0 * np.mean([bool(r.get("rotation_ok_2deg", False)) for r in records])
    return out


# ------------------------------------------------------------------------------------- inputs
def _num(s):
    return float(s.replace("n", "-").replace("p", "."))


def parse_folder(name: str) -> dict:
    vals, toks, i = {}, name.split("_"), 0
    while i < len(toks):
        t = toks[i]
        if t in SHAPES:
            vals["shape"] = t
        elif t in COLOR_NAMES:
            vals["color"] = t
        elif re.fullmatch(r"cnt\d+", t):
            vals["count"] = int(t[3:])
        elif re.fullmatch(r"rot\d+", t):
            vals["rotation"] = float(t[3:])
        elif re.fullmatch(r"r\d+p\d+", t):
            vals["radius"] = _num(t[1:])
        elif re.fullmatch(r"xn?\d+p\d+", t) and i + 1 < len(toks) and re.fullmatch(r"yn?\d+p\d+", toks[i + 1]):
            vals["position"] = (_num(t[1:]), _num(toks[i + 1][1:]))
            i += 1
        else:
            raise ValueError(f"unparsed token {t!r} in folder {name!r}")
        i += 1
    return vals


def cond_from_combo(combo, props) -> dict:
    """Combination tuple as stored by the old evaluator / sampler (shape and color as integer ids)."""
    cond = {}
    for p, v in zip(props, combo):
        if p == "shape":
            cond[p] = SHAPES[int(v)] if not isinstance(v, str) else v
        elif p == "color":
            cond[p] = COLOR_NAMES[int(v)] if not isinstance(v, str) else v
        elif p == "position":
            cond[p] = (float(v[0]), float(v[1]))
        elif p == "count":
            cond[p] = int(v)
        else:
            cond[p] = float(v)
    return cond


def run_context(sample_dir: Path, dataset_dir: Path | None = None, rotation_shape: str | None = None):
    """(pair, props, ctx) for a sample dir written by AUDIT_OCT19/sampling/sample_table3.py, from its
    manifest.json (pair, dataset_path) and the dataset metadata. For a rotation pair the manifest's
    rotation_scoring_shape (what the old scorer used) must agree with the dataset metadata."""
    man = json.load(open(sample_dir / "manifest.json"))
    if not man.get("complete"):
        raise RuntimeError(f"{sample_dir}: manifest not complete")
    pair = man["pair"]
    props = PAIRS[pair][0]
    if list(man["properties"]) != props:
        raise ValueError(f"{sample_dir}: properties {man['properties']} != {props}")
    ddir = Path(dataset_dir or man["dataset_path"])
    meta = json.load(open(ddir / "dataset_metadata.json"))
    ctx = context(props, meta, rotation_shape)
    if "rotation" in props:
        old = str(man.get("settings", {}).get("rotation_scoring_shape", "")).split(" ")[0]
        if old and old != ctx["rotation_shape"]:
            raise ValueError(f"{sample_dir}: manifest rotation_scoring_shape {old!r} != {ctx['rotation_shape']!r}")
    return pair, props, ctx


def score_npz_dir(sample_dir: Path, dataset_dir: Path | None = None, rotation_shape: str | None = None,
                  keep_records: bool = False) -> dict:
    """Score <split>.npz / <split>.json written by AUDIT_OCT19/sampling/sample_table3.py."""
    pair, props, ctx = run_context(sample_dir, dataset_dir, rotation_shape)
    result = {"pair": pair, "properties": props, "rotation_shape": ctx["rotation_shape"], "splits": {}}
    for split in SPLITS:
        if not (sample_dir / f"{split}.npz").exists():
            continue
        images = np.load(sample_dir / f"{split}.npz")["images"]
        info = json.load(open(sample_dir / f"{split}.json"))
        records = []
        for ci, combo in enumerate(info["combinations"]):
            cond = cond_from_combo(combo, props)
            for k in range(images.shape[1]):
                r = score_sample(images[ci, k], cond, ctx)
                r["combo"] = ci
                records.append(r)
        result["splits"][split] = summarize(records, props)
        if keep_records:
            result["splits"][split]["records"] = records
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=Path, required=True, help="dir with manifest.json, <split>.npz and <split>.json")
    ap.add_argument("--dataset", type=Path, default=None, help="override the manifest's dataset_path")
    ap.add_argument("--rotation-shape", choices=("arrow", "triangle"), default=None,
                    help="only for datasets whose metadata has no rotation_shape")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = ap.parse_args()
    res = score_npz_dir(args.samples, args.dataset, args.rotation_shape)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(json.dumps({s: v.get("accuracy") for s, v in res["splits"].items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
