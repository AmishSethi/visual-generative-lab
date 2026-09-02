"""Fast smoke tests: catch a broken install before anyone burns GPU hours.

Run with:  pytest tests/ -v
These are CPU-only and take well under a minute.
"""
import numpy as np
import pytest
import torch


def test_package_imports():
    import vgl
    assert vgl.__version__


@pytest.mark.parametrize("module", [
    "vgl.models", "vgl.models_position", "vgl.models_rotation",
    "vgl.models_compositional", "vgl.unet_models", "vgl.unet_models_song",
    "vgl.diffusion", "vgl.flow_matching", "vgl.reproducibility_utils",
])
def test_module_imports(module):
    __import__(module)


def test_model_registries_populated():
    from vgl.models import DiT_models_continuous
    from vgl.models_rotation import DiT_models_rotation
    from vgl.models_position import DiT_models_position
    for reg, name in ((DiT_models_continuous, "continuous"),
                      (DiT_models_rotation, "rotation"),
                      (DiT_models_position, "position")):
        assert "DiT-S/2" in reg, f"{name} registry missing the paper baseline"


def test_baseline_forward_pass():
    """DiT-S/2 at the paper's configuration must accept a batch and return
    the learned-sigma output (2x input channels)."""
    from vgl.models import DiT_models_continuous
    model = DiT_models_continuous["DiT-S/2"](
        input_size=64, in_channels=3,
        radius_embedding_type="linear", conditioning_method="concat",
    ).eval()
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 250, (2,))
    r = torch.tensor([8.0, 15.0])
    with torch.no_grad():
        out = model(x, t, r)
    assert out.shape[0] == 2 and out.shape[-2:] == (64, 64)
    assert torch.isfinite(out).all(), "forward pass produced NaN/Inf"


def test_baseline_parameter_count():
    """The paper reports DiT-S/2 at ~22.2M parameters. A large deviation means
    the architecture drifted from what produced the published numbers."""
    from vgl.models import DiT_models_continuous
    m = DiT_models_continuous["DiT-S/2"](
        input_size=64, in_channels=3,
        radius_embedding_type="linear", conditioning_method="concat")
    millions = sum(p.numel() for p in m.parameters()) / 1e6
    assert 21.0 < millions < 24.0, f"expected ~22.2M parameters, got {millions:.1f}M"


def test_diffusion_sampler_constructs():
    from vgl.diffusion import create_diffusion
    d = create_diffusion(timestep_respacing="250")
    assert d.num_timesteps == 250


def test_seeding_is_deterministic():
    from vgl.reproducibility_utils import configure_reproducibility
    configure_reproducibility(0); a = torch.randn(8)
    configure_reproducibility(0); b = torch.randn(8)
    assert torch.equal(a, b), "configure_reproducibility did not make torch RNG reproducible"


def test_count_metric_recovers_ground_truth():
    """The watershed counter must read its own ground truth. If this fails,
    any count accuracy measured downstream is a property of the metric."""
    cv2 = pytest.importorskip("cv2")
    from paper_runs.table2.evaluate_table2 import (
        detect_count_with_watershed, COUNT_MIN_AREA)

    rng = np.random.default_rng(0)
    hits = 0
    trials = 12
    for _ in range(trials):
        n = int(rng.integers(2, 8))
        img = np.full((64, 64, 3), 245, dtype=np.uint8)
        # 16 px lattice with 8 px clearance: the released count geometry
        cells = rng.choice(16, size=n, replace=False)
        for c in cells:
            row, col = divmod(int(c), 4)
            cx, cy = col * 16 + 8, row * 16 + 8
            cv2.circle(img, (cx, cy), 4, (200, 40, 40), -1)
        hits += detect_count_with_watershed(img, min_area=COUNT_MIN_AREA) == n
    assert hits / trials >= 0.9, f"counter ceiling too low: {hits}/{trials}"


def test_shape_metric_recovers_ground_truth():
    """The shape classifier must read its own ground-truth renders. Before the
    release fix, a swallowed ImportError silently degraded this to a contour
    fallback that scored 25% (chance)."""
    pytest.importorskip("cv2")
    from scripts.generate_compositional_dataset_coverage import generate_shape_image
    from vgl.eval_compositional import evaluate_properties_comprehensive
    shape_to_id = {"circle": 0, "square": 1, "triangle": 2, "diamond": 3}
    hits = 0
    for name, sid in shape_to_id.items():
        for radius in (10, 14):
            img = np.array(generate_shape_image(radius=radius, position=(0, 0), shape=name,
                                                color_rgb=(255, 0, 0), image_size=64,
                                                rotation=0, count=1))
            # expected_properties is positional, aligned with include_properties
            m = evaluate_properties_comprehensive(img, [sid], ["shape"])
            hits += m.get("shape_accuracy", 0.0) == 1.0
    assert hits == 8, f"shape metric ceiling {hits}/8 on ground truth"
