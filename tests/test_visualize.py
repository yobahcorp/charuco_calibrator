"""Tests for visualize.py — DistortionHeatmap and load_calibration_yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from charuco_calibrator.calibration import CalibrationManager, CalibrationResult
from charuco_calibrator.visualize import DistortionHeatmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cal_yaml(tmp_path: Path, **overrides) -> Path:
    """Write a minimal calibration YAML and return its path."""
    data = {
        "image_width": 640,
        "image_height": 480,
        "camera_name": "test",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [0.1, -0.2, 0.0, 0.0, 0.05],
        },
        "distortion_model": "plumb_bob",
        "rectification_matrix": {
            "rows": 3, "cols": 3,
            "data": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        },
        "projection_matrix": {
            "rows": 3, "cols": 4,
            "data": [500, 0, 320, 0, 0, 500, 240, 0, 0, 0, 1, 0],
        },
        "rms_reprojection_error": 0.42,
    }
    data.update(overrides)
    p = tmp_path / "cal.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


# ---------------------------------------------------------------------------
# load_calibration_yaml tests
# ---------------------------------------------------------------------------

class TestLoadCalibrationYaml:

    def test_load_valid(self, tmp_path):
        p = _write_cal_yaml(tmp_path)
        result = CalibrationManager.load_calibration_yaml(p)
        assert result.valid
        assert result.camera_matrix.shape == (3, 3)
        assert result.dist_coeffs.shape == (1, 5)
        assert result.image_size == (640, 480)
        assert abs(result.rms - 0.42) < 1e-6

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            CalibrationManager.load_calibration_yaml("/nonexistent/cal.yaml")

    def test_load_missing_field(self, tmp_path):
        p = tmp_path / "bad.yaml"
        with open(p, "w") as f:
            yaml.dump({"image_width": 640, "image_height": 480}, f)
        with pytest.raises(KeyError):
            CalibrationManager.load_calibration_yaml(p)

    def test_roundtrip(self, tmp_path):
        mgr = CalibrationManager(image_size=(640, 480))
        mgr.result = CalibrationResult(
            rms=0.35,
            camera_matrix=np.array(
                [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64
            ),
            dist_coeffs=np.array([[0.1, -0.2, 0, 0, 0.05]], dtype=np.float64),
            image_size=(640, 480),
            valid=True,
        )
        yaml_path = mgr.save_yaml(tmp_path / "roundtrip.yaml", camera_name="test")
        loaded = CalibrationManager.load_calibration_yaml(yaml_path)
        assert loaded.valid
        np.testing.assert_allclose(loaded.camera_matrix, mgr.result.camera_matrix)
        np.testing.assert_allclose(loaded.dist_coeffs, mgr.result.dist_coeffs)
        assert loaded.image_size == (640, 480)


# ---------------------------------------------------------------------------
# DistortionHeatmap tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cal():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    D = np.array([[0.1, -0.25, 0, 0, 0.1]], dtype=np.float64)
    return K, D, (640, 480)


class TestDistortionHeatmap:

    def test_overlay_shape(self, sample_cal):
        K, D, size = sample_cal
        hm = DistortionHeatmap(K, D, size)
        assert hm.overlay.shape == (480, 640, 3)
        assert hm.overlay.dtype == np.uint8

    def test_overlay_nonzero(self, sample_cal):
        K, D, size = sample_cal
        hm = DistortionHeatmap(K, D, size)
        assert hm.overlay.max() > 0

    def test_blend_same_size(self, sample_cal):
        K, D, size = sample_cal
        hm = DistortionHeatmap(K, D, size)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        blended = hm.blend_onto(frame)
        assert blended.shape == frame.shape
        assert not np.array_equal(blended, frame)

    def test_blend_different_size(self, sample_cal):
        K, D, size = sample_cal
        hm = DistortionHeatmap(K, D, size)
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        blended = hm.blend_onto(frame)
        assert blended.shape == (240, 320, 3)

    def test_zero_distortion(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        D = np.zeros((1, 5), dtype=np.float64)
        hm = DistortionHeatmap(K, D, (640, 480))
        # With zero distortion, magnitude < 0.5px threshold → all zeros → uniform colormap
        assert np.all(hm.overlay == hm.overlay[0, 0])
