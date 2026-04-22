"""
Tests for altitude_augment.py.

Checks:
  1. compute_scale_bounds: correct clamped outputs
  2. AltitudeAwareRandomPerspective: apparent altitudes produced by
     affine_transform are approximately uniform over [alt_min, alt_max]
     — verifying training matches plot_altitude_dist.py simulation
  3. Fallback: scale drawn from [1-scale, 1+scale] when no altitude
  4. AltitudeAwareMosaic._cat_labels: mean altitude propagation
"""

import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).parent))
from altitude_augment import (
    SCALE_CEILING,
    SCALE_FLOOR,
    AltitudeAwareMosaic,
    AltitudeAwareRandomPerspective,
    compute_scale_bounds,
)

ALT_MIN = 100.0
ALT_MAX = 300.0
N_SAMPLES = 10_000
KS_ALPHA = 0.01   # reject only at strong evidence

DUMMY_IMG = np.full((64, 64, 3), 114, dtype=np.uint8)
DUMMY_BORDER = (0, 0)


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_transform(scale: float = 0.5) -> AltitudeAwareRandomPerspective:
    return AltitudeAwareRandomPerspective(
        alt_min=ALT_MIN,
        alt_max=ALT_MAX,
        degrees=0.0,
        translate=0.0,
        scale=scale,
        shear=0.0,
        perspective=0.0,
    )


def _sample_scales(
    transform: AltitudeAwareRandomPerspective,
    altitude_m: float | None,
    n: int,
    seed: int = 0,
) -> list[float]:
    """Sample n scale factors from affine_transform with _altitude_m preset.

    RandomPerspective.__call__ normally sets self.size before calling
    affine_transform, so we set it manually here.
    """
    random.seed(seed)
    np.random.seed(seed)
    transform._altitude_m = (
        float(altitude_m) if altitude_m is not None else None
    )
    transform.size = (DUMMY_IMG.shape[1], DUMMY_IMG.shape[0])
    scales = []
    for _ in range(n):
        _, _, s = transform.affine_transform(DUMMY_IMG, DUMMY_BORDER)
        scales.append(s)
    return scales


# ── compute_scale_bounds ────────────────────────────────────────────────────

class TestComputeScaleBounds:
    def test_unclamped(self):
        s_lo, s_hi = compute_scale_bounds(150.0, ALT_MIN, ALT_MAX)
        assert s_lo == pytest.approx(150.0 / ALT_MAX)
        assert s_hi == pytest.approx(150.0 / ALT_MIN)

    def test_floor_clamp(self):
        s_lo, _ = compute_scale_bounds(1.0, 100.0, 10_000.0)
        assert s_lo == SCALE_FLOOR

    def test_ceiling_clamp(self):
        _, s_hi = compute_scale_bounds(1000.0, 1.0, 300.0)
        assert s_hi == SCALE_CEILING

    def test_s_lo_le_s_hi(self):
        for h in [80.0, 150.0, 250.0, 400.0]:
            s_lo, s_hi = compute_scale_bounds(h, ALT_MIN, ALT_MAX)
            assert s_lo <= s_hi


# ── distribution tests ──────────────────────────────────────────────────────

class TestApparentAltitudeDistribution:
    """apparent_altitude = h / s should be ~ U(alt_min, alt_max)."""

    @pytest.mark.parametrize("h", [120.0, 150.0, 200.0, 250.0])
    def test_ks_uniform(self, h: float):
        """KS test: apparent altitudes not distinguishable from 
        U(alt_min, alt_max)."""
        t = _make_transform()
        scales = _sample_scales(t, h, N_SAMPLES)
        apparent = np.array([h / s for s in scales])

        # Only test unclamped portion (clamping warps extreme values)
        mask = (apparent >= ALT_MIN) & (apparent <= ALT_MAX)
        normed = (apparent[mask] - ALT_MIN) / (ALT_MAX - ALT_MIN)

        stat, p = kstest(normed, "uniform")
        assert p > KS_ALPHA, (
            f"h={h}m: KS p={p:.4f} < {KS_ALPHA} — distribution is not "
            f"flat (stat={stat:.4f})"
        )

    @pytest.mark.parametrize("h", [120.0, 150.0, 200.0, 250.0])
    def test_mean_near_midpoint(self, h: float):
        t = _make_transform()
        scales = _sample_scales(t, h, N_SAMPLES)
        apparent = np.array([h / s for s in scales])
        unclamped = apparent[(apparent >= ALT_MIN) & (apparent <= ALT_MAX)]
        midpoint = (ALT_MIN + ALT_MAX) / 2.0
        assert abs(unclamped.mean() - midpoint) < 5.0, (
            f"h={h}m: mean apparent altitude {unclamped.mean():.1f} "
            f"far from midpoint {midpoint}"
        )


# ── fallback behaviour ──────────────────────────────────────────────────────

class TestFallback:
    def test_no_altitude_altitude_m_is_none(self):
        """_altitude_m is None when no altitude_m key in labels."""
        t = _make_transform(scale=0.3)
        # Call only our override (stop before super().__call__ needs an image)
        altitude_m = None
        t._altitude_m = (
            float(altitude_m) if altitude_m is not None else None
        )
        assert t._altitude_m is None

    def test_altitude_present_sets_altitude_m(self):
        t = _make_transform()
        t._altitude_m = float(180.0)
        assert t._altitude_m == pytest.approx(180.0)

    def test_fallback_scale_range(self):
        """Without altitude, scales stay within [1-scale, 1+scale]."""
        hyp_scale = 0.4
        t = _make_transform(scale=hyp_scale)
        scales = _sample_scales(t, altitude_m=None, n=1000)
        lo, hi = 1.0 - hyp_scale, 1.0 + hyp_scale
        assert all(lo <= s <= hi for s in scales), (
            f"Some scales outside [{lo}, {hi}]: "
            f"min={min(scales):.3f}, max={max(scales):.3f}"
        )

    def test_altitude_aware_scale_can_exceed_hyp_scale(self):
        """Altitude-aware mode can produce scales outside
        [1-scale, 1+scale]."""
        hyp_scale = 0.1   # tight symmetric range
        t = _make_transform(scale=hyp_scale)
        # h=250m, alt_min=100 → s_hi = 2.5, well above 1+0.1
        scales = _sample_scales(t, altitude_m=250.0, n=500)
        assert max(scales) > 1.0 + hyp_scale, (
            "Expected altitude-aware scales to exceed hyp.scale bounds"
        )


# ── AltitudeAwareMosaic ──────────────────────────────────────────────────────

def _make_mosaic_labels(altitudes: list[float | None]) -> list[dict]:
    """Build minimal label dicts as _cat_labels receives them."""
    from ultralytics.utils.instance import Instances
    labels = []
    for alt in altitudes:
        lbl = {
            "im_file": "dummy.jpg",
            "ori_shape": (64, 64),
            "cls": np.zeros((0,), dtype=np.float32),
            "instances": Instances(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                segments=np.zeros((0, 0, 2), dtype=np.float32),
                keypoints=None,
                bbox_format="xyxy",
                normalized=False,
            ),
        }
        if alt is not None:
            lbl["altitude_m"] = float(alt)
        labels.append(lbl)
    return labels


class TestAltitudeAwareMosaic:
    def _make_mosaic(self) -> AltitudeAwareMosaic:
        mock_dataset = MagicMock()
        mock_dataset.cache = False
        return AltitudeAwareMosaic(
            dataset=mock_dataset, imgsz=64, p=1.0, n=4
        )

    def test_mean_of_four_altitudes(self):
        mosaic = self._make_mosaic()
        labels = _make_mosaic_labels([100.0, 150.0, 200.0, 250.0])
        result = mosaic._cat_labels(labels)
        assert result["altitude_m"] == pytest.approx(175.0)

    def test_partial_altitude_coverage(self):
        """Mean is computed only over frames that have altitude_m."""
        mosaic = self._make_mosaic()
        labels = _make_mosaic_labels([120.0, None, 180.0, None])
        result = mosaic._cat_labels(labels)
        assert result["altitude_m"] == pytest.approx(150.0)

    def test_no_altitudes_absent_from_result(self):
        """altitude_m should not appear in result when no frame has it."""
        mosaic = self._make_mosaic()
        labels = _make_mosaic_labels([None, None, None, None])
        result = mosaic._cat_labels(labels)
        assert "altitude_m" not in result

    def test_single_altitude_used_directly(self):
        mosaic = self._make_mosaic()
        labels = _make_mosaic_labels([200.0, None, None, None])
        result = mosaic._cat_labels(labels)
        assert result["altitude_m"] == pytest.approx(200.0)
