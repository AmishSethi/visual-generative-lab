"""VGL skill queries expressed as text prompts for pretrained T2I models.

Reviewer HeVs' central concern, echoed by the AC, is that VGL's 64x64 toy
setting may not say anything about real generative models.  The direct test is
to pose the *same* skill queries to off-the-shelf text-to-image models and score
their outputs with the *same* rule-based metrics.  If the failure pattern
matches, the toy setting is a validated proxy rather than a curiosity.

Prompts deliberately request flat 2D shapes on plain white backgrounds so the
VGL extractors (Otsu threshold -> mask -> geometry) apply unchanged.
"""

# "red circle" pulls Stable Diffusion straight to bullseye/target imagery on
# textured backgrounds, which breaks the geometry extractor before the skill is
# ever tested.  "Solid filled dot", an explicit white-background anchor, and a
# negative prompt that names the failure modes keep the render inside the
# protocol the VGL extractors assume.
STYLE = ("simple flat vector clipart of {subject}, solid uniform fill, no outline, "
         "isolated on a pure white background, no shadow, no gradient, no border, no text")

# --- count -----------------------------------------------------------------
# VGL trains on 2..7 and calls 0, 1, 8, 9 extrapolation.  We query the whole
# range so the T2I accuracy-vs-count curve is directly comparable.
COUNT_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
COUNT_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def count_prompt(n):
    noun = "solid red dot" if n == 1 else "identical solid red dots"
    return STYLE.format(subject=f"exactly {COUNT_WORDS[n]} ({n}) separate {noun}")


# --- position --------------------------------------------------------------
# A 3x3 grid is the finest spatial control natural language reliably affords;
# VGL's continuous (x, y) has no prompt analogue, so we coarsen the metric
# rather than invent one.
POSITION_CELLS = [
    ("top left", (0, 0)), ("top center", (1, 0)), ("top right", (2, 0)),
    ("middle left", (0, 1)), ("center", (1, 1)), ("middle right", (2, 1)),
    ("bottom left", (0, 2)), ("bottom center", (1, 2)), ("bottom right", (2, 2)),
]


def position_prompt(cell_name):
    where = "the exact center" if cell_name == "center" else f"the {cell_name}"
    return STYLE.format(subject=f"one small solid red dot placed at {where} of the frame, "
                                f"and nothing else anywhere in the image")


# --- size ------------------------------------------------------------------
# Requested as a percentage of image width, which is the closest text analogue
# of VGL's radius conditioning.  Target fractions are the area the circle should
# occupy if the model honoured the request.
SIZE_LEVELS = [
    ("10%", 0.10), ("20%", 0.20), ("30%", 0.30),
    ("50%", 0.50), ("70%", 0.70), ("90%", 0.90),
]


def size_prompt(pct_label):
    return STYLE.format(
        subject=f"one solid red dot centered in the frame, its diameter exactly {pct_label} "
                f"of the image width"
    )


# --- rotation --------------------------------------------------------------
# Eight compass directions, the natural-language analogue of VGL's 1-degree
# angle grid.  Expected angles use VGL's convention: 0 deg = pointing right,
# increasing counter-clockwise.
ROTATION_DIRECTIONS = [
    ("right", 0.0), ("up and to the right", 45.0), ("straight up", 90.0),
    ("up and to the left", 135.0), ("left", 180.0), ("down and to the left", 225.0),
    ("straight down", 270.0), ("down and to the right", 315.0),
]


def rotation_prompt(direction):
    return STYLE.format(
        subject=f"one solid red arrow pointing {direction}, centered in the frame"
    )


NEGATIVE_PROMPT = (
    "target, bullseye, concentric rings, ring, donut, outline, stroke, "
    "dark background, black background, coloured background, textured background, pattern, "
    "photograph, 3d render, shadow, gradient, scenery, text, letters, numbers, watermark, "
    "signature, logo, multiple panels, collage, frame, border, ornament"
)


def build_all():
    """Returns {skill: [(condition_key, prompt), ...]}."""
    return {
        "count": [(n, count_prompt(n)) for n in COUNT_VALUES],
        "position": [(name, position_prompt(name)) for name, _ in POSITION_CELLS],
        "size": [(label, size_prompt(label)) for label, _ in SIZE_LEVELS],
        "rotation": [(name, rotation_prompt(name)) for name, _ in ROTATION_DIRECTIONS],
    }
