"""Tests for calibration.py — CalibrationManager, YAML export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from charuco_calibrator.calibration import CalibrationManager, CalibrationResult
from charuco_calibrator.config import BoardConfig
from charuco_calibrator.detector import CharucoDetectorWrapper


def _generate_observations(detector: CharucoDetectorWrapper, n: int = 6):
    """Generate synthetic observations by rendering the board at different transforms."""
    board_img = detector.generate_board_image(600, 800)
    if len(board_img.shape) == 2:
        board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    else:
        board_bgr = board_img

    observations = []
    h, w = 600, 800

    for i in range(n):
        # Create slightly different views by shifting / scaling
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        offset_x = 20 + i * 15
        offset_y = 10 + i * 10
        bh, bw = board_bgr.shape[:2]

        # Compute safe placement
        place_h = min(bh, h - offset_y)
        place_w = min(bw, w - offset_x)
        if place_h <= 0 or place_w <= 0:
            continue
        frame[offset_y : offset_y + place_h, offset_x : offset_x + place_w] = (
            board_bgr[:place_h, :place_w]
        )

        result = detector.detect(frame)
        if result.valid and result.num_corners >= 6:
            obj_pts, img_pts = detector.match_image_points(
                result.charuco_corners, result.charuco_ids
            )
            observations.append((obj_pts, img_pts, result.charuco_ids))

    return observations, (w, h)


class TestCalibrationManager:
    def test_add_observation(self, detector: CharucoDetectorWrapper):
        mgr = CalibrationManager()
        observations, image_size = _generate_observations(detector, n=1)
        if observations:
            obj_pts, img_pts, ids = observations[0]
            idx = mgr.add_observation(obj_pts, img_pts, ids)
            assert idx == 0
            assert mgr.num_observations == 1

    def test_calibrate_insufficient_frames(self):
        mgr = CalibrationManager()
        result = mgr.calibrate((640, 480))
        assert not result.valid

    def test_calibrate_with_observations(self, detector: CharucoDetectorWrapper):
        mgr = CalibrationManager()
        observations, image_size = _generate_observations(detector, n=8)
        assert len(observations) >= 4, f"Only got {len(observations)} observations"

        for obj_pts, img_pts, ids in observations:
            mgr.add_observation(obj_pts, img_pts, ids)

        result = mgr.calibrate(image_size)
        assert result.valid
        assert result.rms > 0
        assert result.camera_matrix is not None
        assert result.camera_matrix.shape == (3, 3)
        assert result.dist_coeffs is not None

    def test_save_yaml(self, detector: CharucoDetectorWrapper, tmp_path: Path):
        mgr = CalibrationManager()
        observations, image_size = _generate_observations(detector, n=8)
        for obj_pts, img_pts, ids in observations:
            mgr.add_observation(obj_pts, img_pts, ids)
        mgr.calibrate(image_size)

        yaml_path = tmp_path / "calib.yaml"
        mgr.save_yaml(yaml_path, camera_name="test_cam")
        assert yaml_path.exists()

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        assert data["camera_name"] == "test_cam"
        assert data["image_width"] == image_size[0]
        assert data["image_height"] == image_size[1]
        assert "camera_matrix" in data
        assert "distortion_coefficients" in data
        assert "projection_matrix" in data
        assert len(data["camera_matrix"]["data"]) == 9

    def test_save_yaml_no_result_raises(self, tmp_path: Path):
        mgr = CalibrationManager()
        with pytest.raises(RuntimeError):
            mgr.save_yaml(tmp_path / "bad.yaml")

    def test_save_observations_npz(self, detector: CharucoDetectorWrapper, tmp_path: Path):
        mgr = CalibrationManager()
        observations, _ = _generate_observations(detector, n=3)
        for obj_pts, img_pts, ids in observations:
            mgr.add_observation(obj_pts, img_pts, ids)

        npz_path = tmp_path / "obs.npz"
        mgr.save_observations_npz(npz_path)
        assert npz_path.exists()

        loaded = np.load(npz_path, allow_pickle=True)
        assert "object_points" in loaded
        assert "image_points" in loaded

    def test_reset(self, detector: CharucoDetectorWrapper):
        mgr = CalibrationManager()
        observations, image_size = _generate_observations(detector, n=6)
        for obj_pts, img_pts, ids in observations:
            mgr.add_observation(obj_pts, img_pts, ids)
        mgr.calibrate(image_size)
        assert mgr.result is not None

        mgr.reset()
        assert mgr.num_observations == 0
        assert mgr.result is None

    def test_per_view_errors(self, detector: CharucoDetectorWrapper):
        mgr = CalibrationManager()
        observations, image_size = _generate_observations(detector, n=8)
        for obj_pts, img_pts, ids in observations:
            mgr.add_observation(obj_pts, img_pts, ids)
        mgr.calibrate(image_size)

        errors = mgr.compute_per_view_errors_full()
        assert len(errors) == len(observations)
        assert all(e >= 0 for e in errors)
