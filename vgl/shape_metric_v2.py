"""Shape classification that reads its own ground truth on every compositional dataset.

The paper-locked classifier in eval_compositional.py matches a centred union mask against
templates at six sizes (8-18 px). On datasets whose objects fall outside that range, or that
contain several objects, it cannot read ground-truth renders (32% on shape_count). This version
classifies each connected component separately, matches templates of every size IN PLACE so the
frame clips template and object identically, and votes across components weighted by IoU.
Measured ceilings on ground truth: shape_count 94.8%, radius_shape 98.3%, position_shape 95.8%,
shape_color 94.8% (see paper_runs/table3/shape_metric_ceiling.py).

It is NOT the metric behind the published tables; select it with VGL_SHAPE_METRIC=v2.
"""
import collections
import numpy as np
import cv2

NAMES = ["circle", "square", "triangle", "diamond"]
_TPL = {}


def _template(name, r):
    if (name, r) not in _TPL:
        from scripts.generate_compositional_dataset_coverage import generate_shape_image
        t = np.array(generate_shape_image(radius=r, position=(0, 0), shape=name, color_rgb=(255, 0, 0),
                                          image_size=64, rotation=0, count=1))
        m = (np.linalg.norm(t.astype(float) - 255, axis=2) > 40).astype(np.uint8)
        ys, xs = np.nonzero(m)
        _TPL[(name, r)] = (m, ys.mean(), xs.mean(), int(m.sum()))
    return _TPL[(name, r)]


def foreground_mask(img):
    m = (np.linalg.norm(img.astype(float) - 255, axis=2) > 40).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def classify_component(comp):
    """(best IoU, shape name) for one binary component in a 64x64 frame."""
    ys, xs = np.nonzero(comp); cy, cx = ys.mean(), xs.mean(); area = int(comp.sum())
    best = (-1.0, None)
    for name in NAMES:
        for r in range(3, 33):
            t, ty, tx, ta = _template(name, r)
            if ta < 0.5 * area:
                continue
            ts = cv2.warpAffine(t, np.float32([[1, 0, cx - tx], [0, 1, cy - ty]]), (64, 64))
            if abs(int(ts.sum()) - area) > 0.35 * area:
                continue
            inter = np.logical_and(comp > 0, ts > 0).sum(); union = np.logical_or(comp > 0, ts > 0).sum()
            iou = inter / union if union else 0.0
            if iou > best[0]:
                best = (iou, name)
    return best


def classify_shape(img, min_area=40):
    """Majority shape across components (IoU-weighted). Returns a name in NAMES, or None."""
    m = foreground_mask(np.asarray(img, dtype=np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    votes = collections.Counter()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        iou, name = classify_component((lab == i).astype(np.uint8))
        if name:
            votes[name] += iou
    return votes.most_common(1)[0][0] if votes else None
