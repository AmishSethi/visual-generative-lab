#!/usr/bin/env python
"""Score pretrained-T2I outputs with rule-based skill extractors adapted from VGL's metrics.

Extraction is built for real T2I renders, which have soft edges, off-white
backgrounds and occasional drop shadows: the foreground test keys on the
redness of the requested subject rather than the plain near-white threshold
VGL uses on its own clean renders.

`--validate` runs the identical scorer over VGL ground-truth renders first; the
validation numbers are reported alongside the T2I numbers.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_runs.appendix.t2i_prompts import POSITION_CELLS, ROTATION_DIRECTIONS, SIZE_LEVELS

import os as _os
# Storage root. Override for your own cluster:
#   export VGL_ROOT=/path/to/your/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))
# Conda activation line injected into generated SLURM scripts.
VGL_CONDA_ACTIVATE = _os.environ.get("VGL_CONDA_ACTIVATE", "conda activate vgl")


SIZE_REL_TOLERANCE = 0.25      # |measured - requested| / requested
ROTATION_TOLERANCE_DEG = 22.5  # half a compass sector
MIN_BLOB_AREA_FRAC = 0.0008    # ignore specks below this fraction of the image


BORDER_FRAC = 0.06          # width of the ring used to estimate the background
BACKGROUND_UNIFORM_STD = 26  # per-channel std above which the background is not flat
BACKGROUND_MIN_BRIGHTNESS = 170
MAX_FOREGROUND_FRACTION = 0.92


def background_stats(img_rgb):
    """Estimate the background colour from a border ring."""
    h, w = img_rgb.shape[:2]
    b = max(2, int(round(BORDER_FRAC * min(h, w))))
    ring = np.concatenate([
        img_rgb[:b].reshape(-1, 3), img_rgb[-b:].reshape(-1, 3),
        img_rgb[:, :b].reshape(-1, 3), img_rgb[:, -b:].reshape(-1, 3),
    ]).astype(np.float32)
    return np.median(ring, axis=0), float(ring.std(axis=0).mean())


REDNESS_THRESHOLD = 45      # R - max(G, B), on 0-255
MIN_SUBJECT_FRACTION = 0.0008


def redness(img_rgb):
    rgb = img_rgb.astype(np.int16)
    return rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])


def foreground_mask(img_rgb):
    """Foreground = the red subject, isolated by colour rather than by contrast.

    Every prompt asks for a red subject, and the models frequently honour the
    subject while ignoring the "plain white background" instruction, returning
    striped or halftone backdrops.  A background-contrast rule throws those
    images away even though the red geometry is perfectly measurable, and a
    "saturated or dark" rule marks the whole frame as subject.  Keying on
    redness segments the thing the query is actually about and is immune to the
    achromatic textures the models like to add.
    """
    mask = (redness(img_rgb) > REDNESS_THRESHOLD).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def protocol_compliant(img_rgb):
    """Is there a measurable red subject at all?

    This is a much weaker gate than requiring a plain background: it only asks
    whether the query's subject exists and does not swallow the frame.  Images
    that fail it carry no extractable geometry, so scoring them as skill
    failures would confound "drew the wrong thing entirely" with "drew the right
    thing in the wrong place"; we count them separately and report the rate.
    """
    mask = foreground_mask(img_rgb)
    fraction = float(mask.mean())
    if fraction < MIN_SUBJECT_FRACTION:
        return False, "no red subject found"
    if fraction > MAX_FOREGROUND_FRACTION:
        return False, "subject fills frame"
    return True, "ok"


def blobs(mask, min_area):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append({"area": int(stats[i, cv2.CC_STAT_AREA]),
                        "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                        "label": i, "labels": labels})
    return out


FRAME_MARGIN = 2           # px; a blob this close to the image edge touches the frame
HOLLOW_FILL_RATIO = 1.3    # filled outline / blob area above which a blob is a ring or frame


def _touches_frame(component):
    m = FRAME_MARGIN
    return bool(component[:m].any() or component[-m:].any()
                or component[:, :m].any() or component[:, -m:].any())


def _is_hollow(component):
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(component.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled.sum() > HOLLOW_FILL_RATIO * component.sum()


def main_blob(img_rgb):
    """The measured object: the largest red blob clear of the frame and not hollow.

    Models often wrap the requested dot or arrow in a red vignette, frame,
    border or corner gradient, and the largest red blob is then the decoration.
    Those decorations touch the image edge or enclose a hole, so the largest
    blob that does neither is the object.  If no blob qualifies, the largest
    blob is used, so a dot that fills the frame is still measured.
    """
    mask = foreground_mask(img_rgb)
    h, w = mask.shape
    found = sorted(blobs(mask, MIN_BLOB_AREA_FRAC * h * w), key=lambda b: -b["area"])
    if not found:
        return None
    for b in found:
        component = b["labels"] == b["label"]
        if not _touches_frame(component) and not _is_hollow(component):
            return b
    return found[0]


# --------------------------------------------------------------------------
# per-skill scorers
# --------------------------------------------------------------------------
COUNT_MARKER_FRAC = 0.6    # watershed markers: above this fraction of each blob's own distance maximum
COUNT_SPECK_FRAC = 0.2     # regions below this fraction of the median region area are fragments


def score_count(img_rgb):
    """Count distinct objects, splitting ones that touch.

    Connected components alone merge touching objects, so a model that
    correctly draws two adjacent spheres would score 1, charging the model for
    the metric's failure.  A distance-transform watershed separates convex blobs
    that touch.  Markers are taken per blob, relative to that blob's own
    distance maximum, so a small dot next to a large one still gets a marker,
    and the regions are counted one each: dividing areas by the median area
    counts a dot with 1.5x the median area as two.  Regions under a fifth of
    the median area are highlight or shadow fragments and are not counted.
    """
    mask = foreground_mask(img_rgb)
    h, w = mask.shape
    min_area = MIN_BLOB_AREA_FRAC * h * w
    if mask.max() == 0:
        return 0

    scaled = (mask * 255).astype(np.uint8)
    dist = cv2.distanceTransform(scaled, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return 0
    n_blobs, blob = cv2.connectedComponents(scaled)
    blob_max = np.zeros(n_blobs)
    np.maximum.at(blob_max, blob.ravel(), dist.ravel())
    sure_fg = np.uint8((dist > COUNT_MARKER_FRAC * blob_max[blob]) & (blob > 0)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    unknown = cv2.subtract(cv2.dilate(scaled, kernel, iterations=2), sure_fg)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 1:
        return 0
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR), markers)

    _, areas = np.unique(markers[markers > 1], return_counts=True)
    areas = areas[areas >= min_area].astype(np.float64)
    if len(areas) == 0:
        return 0
    areas = areas[areas >= COUNT_SPECK_FRAC * np.median(areas)]
    return int(len(areas))


def score_position(img_rgb):
    blob = main_blob(img_rgb)
    if blob is None:
        return None
    h, w = blob["labels"].shape
    cx, cy = blob["centroid"]
    return (min(2, int(cx / (w / 3))), min(2, int(cy / (h / 3))))


def score_size(img_rgb):
    """Diameter of the main blob as a fraction of image width."""
    blob = main_blob(img_rgb)
    if blob is None:
        return None
    w = blob["labels"].shape[1]
    diameter = 2.0 * np.sqrt(blob["area"] / np.pi)
    return float(diameter / w)


ROTATION_SYM_MARGIN = 0.08  # a mirror axis must beat the principal axis by this much
ROTATION_TRIM = 0.005       # fraction of pixels ignored at each end of the axis (stubs, specks)
ROTATION_END = 0.04         # depth of an end slice, as a fraction of the arrow length
ROTATION_BINS = 40


def _mirror_score(component, points, centre, direction):
    """Fraction of pixels whose reflection across the line (centre, direction) is also in the blob."""
    normal = np.array([-direction[1], direction[0]])
    mirrored = points - 2.0 * ((points - centre) @ normal)[:, None] * normal[None, :]
    xi = np.round(mirrored[:, 0]).astype(int)
    yi = np.round(mirrored[:, 1]).astype(int)
    h, w = component.shape
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    hit = np.zeros(len(points), bool)
    hit[inside] = component[yi[inside], xi[inside]]
    return float(hit.mean())


def score_rotation(img_rgb):
    """Arrow direction in VGL convention (0 deg = right, counter-clockwise).

    Line of action: an arrow is mirror-symmetric about its direction.  That is
    usually the principal axis, but for short wide shapes (chevrons, heads wider
    than the arrow is long) it is the cross axis, so the best mirror axis over
    5-degree steps replaces the principal axis when it is clearly more symmetric.

    Sign: the head is the end that comes to a point.  When one end is a flat cut
    (the tail of a block arrow, the back of a triangle or chevron) at least 0.3x
    the arrow's width, and the other end is at most half as wide, the narrow end
    is the tip.  This holds for heads of any length, where comparing the spread
    of the two axial quartiles flips long or fat heads.  Otherwise (thin shafts,
    where both ends are narrow) the quartile with the larger spread is the head,
    as for VGL's own arrows.
    """
    blob = main_blob(img_rgb)
    if blob is None:
        return None
    component = blob["labels"] == blob["label"]
    ys, xs = np.nonzero(component)
    if len(xs) < 20:
        return None
    step = max(1, len(xs) // 40000)
    points = np.stack([xs, ys], axis=1).astype(np.float64)[::step]
    centre = points.mean(axis=0)
    centred = points - centre

    principal = np.linalg.svd(centred, full_matrices=False)[2][0]
    axis = principal
    candidates = [np.array([np.cos(t), np.sin(t)]) for t in np.radians(np.arange(0, 180, 5))]
    scores = [_mirror_score(component, points, centre, d) for d in candidates]
    best = int(np.argmax(scores))
    if scores[best] > _mirror_score(component, points, centre, principal) + ROTATION_SYM_MARGIN:
        axis = candidates[best]

    axial = centred @ axis
    keep = (axial >= np.quantile(axial, ROTATION_TRIM)) & (axial <= np.quantile(axial, 1 - ROTATION_TRIM))
    centred = centred[keep]
    axial = centred @ axis
    across = centred @ np.array([-axis[1], axis[0]])
    lo, hi = axial.min(), axial.max()
    length = max(hi - lo, 1e-6)

    def extent(sel):
        return across[sel].max() - across[sel].min() + 1 if sel.any() else 0.0

    bins = np.clip(((axial - lo) / length * ROTATION_BINS).astype(int), 0, ROTATION_BINS - 1)
    width = max(extent(bins == b) for b in range(ROTATION_BINS))

    perpendicular = np.abs(across)
    head_side = axial >= np.quantile(axial, 0.75)
    tail_side = axial <= np.quantile(axial, 0.25)
    sign = 1.0 if perpendicular[head_side].mean() > perpendicular[tail_side].mean() else -1.0

    d = ROTATION_END * length
    end_hi, end_lo = extent(axial >= hi - d), extent(axial <= lo + d)
    wide, narrow = max(end_hi, end_lo), min(end_hi, end_lo)
    if end_hi >= end_lo:
        inner = extent((axial >= hi - 3 * d) & (axial <= hi - 2 * d))
    else:
        inner = extent((axial <= lo + 3 * d) & (axial >= lo + 2 * d))
    # the wide end must be a cut (as wide at 4% depth as 0.6x its width at 8-12%), not a wide-angle apex
    if wide >= 0.3 * width and narrow <= 0.5 * wide and wide >= 0.6 * inner:
        sign = 1.0 if end_hi < end_lo else -1.0

    direction = sign * axis
    # image y grows downward, so flip it to get standard counter-clockwise angles
    return float(np.degrees(np.arctan2(-direction[1], direction[0])) % 360.0)


def angle_error(pred, target):
    diff = abs((pred - target + 180.0) % 360.0 - 180.0)
    return diff


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def score_condition_dir(skill, condition, cond_dir):
    results = []
    for path in sorted(cond_dir.glob("sample_*.png")):
        img = np.array(Image.open(path).convert("RGB"))
        compliant, reason = protocol_compliant(img)
        if not compliant:
            results.append({"pred": None, "correct": False,
                            "compliant": False, "reason": reason})
            continue
        if skill == "count":
            pred = score_count(img)
            results.append({"pred": pred, "correct": pred == int(condition), "compliant": True})
        elif skill == "position":
            target = dict(POSITION_CELLS)[condition]
            pred = score_position(img)
            results.append({"pred": pred, "correct": pred == target, "compliant": True})
        elif skill == "size":
            target = dict(SIZE_LEVELS)[condition]
            pred = score_size(img)
            ok = pred is not None and abs(pred - target) / target <= SIZE_REL_TOLERANCE
            results.append({"pred": pred, "correct": bool(ok), "compliant": True})
        elif skill == "rotation":
            target = dict(ROTATION_DIRECTIONS)[condition]
            pred = score_rotation(img)
            ok = pred is not None and angle_error(pred, target) <= ROTATION_TOLERANCE_DEG
            results.append({"pred": pred, "correct": bool(ok), "compliant": True,
                            "error": None if pred is None else angle_error(pred, target)})
    return results


def condition_key_to_dirname(condition):
    return str(condition).replace(" ", "_").replace("%", "pct")


def dirname_to_condition(skill, name):
    if skill == "count":
        return int(name)
    if skill == "position":
        return name.replace("_", " ")
    if skill == "size":
        return name.replace("pct", "%")
    return name.replace("_", " ")


def score_model(model_dir, skills):
    summary = {}
    for skill in skills:
        skill_dir = model_dir / skill
        if not skill_dir.exists():
            continue
        per_condition = {}
        for cond_dir in sorted(skill_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            condition = dirname_to_condition(skill, cond_dir.name)
            res = score_condition_dir(skill, condition, cond_dir)
            if not res:
                continue
            compliant = [r for r in res if r["compliant"]]
            acc = 100.0 * sum(r["correct"] for r in res) / len(res)
            entry = {
                "n": len(res),
                "accuracy": acc,
                "n_compliant": len(compliant),
                "compliance_rate": 100.0 * len(compliant) / len(res),
                "accuracy_given_compliant": (
                    100.0 * sum(r["correct"] for r in compliant) / len(compliant)
                    if compliant else None
                ),
            }
            if skill == "size":
                vals = [r["pred"] for r in res if r.get("pred") is not None]
                entry["mean_measured_fraction"] = float(np.mean(vals)) if vals else None
                entry["requested_fraction"] = dict(SIZE_LEVELS)[condition]
            if skill == "rotation":
                errs = [r["error"] for r in res if r.get("error") is not None]
                entry["mean_angle_error"] = float(np.mean(errs)) if errs else None
            if skill == "count":
                preds = [r["pred"] for r in res if r.get("pred") is not None]
                entry["mean_predicted"] = float(np.mean(preds)) if preds else None
            per_condition[str(condition)] = entry
        overall = float(np.mean([v["accuracy"] for v in per_condition.values()])) if per_condition else None
        compliance = float(np.mean([v["compliance_rate"] for v in per_condition.values()])) if per_condition else None
        given = [v["accuracy_given_compliant"] for v in per_condition.values()
                 if v["accuracy_given_compliant"] is not None]
        summary[skill] = {
            "overall_accuracy": overall,
            "compliance_rate": compliance,
            "accuracy_given_compliant": float(np.mean(given)) if given else None,
            "per_condition": per_condition,
        }
    return summary


# --------------------------------------------------------------------------
# validation on VGL ground truth
# --------------------------------------------------------------------------
def validate():
    """Run the identical extractors on VGL-style synthetic renders."""
    from paper_runs.table2.generate_canonical_datasets import (
        render_arrow, render_circle, sample_non_overlapping_centers,
    )
    from PIL import ImageDraw

    rng = np.random.default_rng(0)
    out = {}

    # count: 1..10 circles on a 512 canvas (T2I-like resolution)
    correct = total = 0
    for n in range(1, 11):
        for _ in range(10):
            size = 512
            radius = 28
            img = Image.new("RGB", (size, size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            centers = sample_non_overlapping_centers(n, radius, size, rng)
            for cx, cy in centers:
                draw.ellipse([size / 2 + cx - radius, size / 2 + cy - radius,
                              size / 2 + cx + radius, size / 2 + cy + radius],
                             fill=(220, 40, 40))
            total += 1
            correct += score_count(np.array(img)) == n
    out["count"] = 100.0 * correct / total

    # position: circle placed in each of the 9 cells
    correct = total = 0
    for name, target in POSITION_CELLS:
        for _ in range(5):
            size = 512
            img = Image.new("RGB", (size, size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            cx = (target[0] + 0.5) * size / 3
            cy = (target[1] + 0.5) * size / 3
            r = 40
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 40, 40))
            total += 1
            correct += score_position(np.array(img)) == target
    out["position"] = 100.0 * correct / total

    # size: circle at each requested diameter fraction
    errors = []
    for _, frac in SIZE_LEVELS:
        size = 512
        r = frac * size / 2
        img = Image.new("RGB", (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r], fill=(220, 40, 40))
        measured = score_size(np.array(img))
        errors.append(abs(measured - frac) / frac)
    out["size_mean_rel_error"] = float(np.mean(errors))

    # rotation: VGL arrows at the eight compass angles
    correct = total = 0
    errs = []
    for _, angle in ROTATION_DIRECTIONS:
        img = render_arrow(angle_degrees=angle, image_size=512, shape_size=240,
                           fill_rgb=(220, 40, 40), background_rgb=(255, 255, 255))
        pred = score_rotation(np.array(img))
        err = angle_error(pred, angle) if pred is not None else 180.0
        errs.append(err)
        total += 1
        correct += err <= ROTATION_TOLERANCE_DEG
    out["rotation"] = 100.0 * correct / total
    out["rotation_mean_error"] = float(np.mean(errs))
    return out


def main():
    parser = argparse.ArgumentParser(description="Score T2I outputs with VGL metrics.")
    parser.add_argument("--images-root", type=Path,
                        default=Path(f"{VGL_ROOT}/appendix/eval/t2i/images"))
    parser.add_argument("--out", type=Path,
                        default=Path(f"{VGL_ROOT}/appendix/eval/t2i/scores.json"))
    parser.add_argument("--skills", nargs="+", default=["count", "position", "size", "rotation"])
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    payload = {}
    if args.validate:
        payload["validation_on_vgl_ground_truth"] = validate()
        print("validation on VGL ground truth:")
        for k, v in payload["validation_on_vgl_ground_truth"].items():
            print(f"    {k:>24}: {v:.2f}")

    if args.images_root.exists():
        for model_dir in sorted(p for p in args.images_root.iterdir() if p.is_dir()):
            payload[model_dir.name] = score_model(model_dir, args.skills)
            print(f"\n=== {model_dir.name} ===")
            for skill, res in payload[model_dir.name].items():
                if res["overall_accuracy"] is not None:
                    given = res["accuracy_given_compliant"]
                    given_str = f"{given:.1f}%" if given is not None else "n/a"
                    print(f"  {skill:>9}: {res['overall_accuracy']:5.1f}% raw   "
                          f"{given_str:>6} on-protocol   "
                          f"(protocol compliance {res['compliance_rate']:.0f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
