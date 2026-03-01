"""Tests for visualize_3d.py — 3D calibration visualization (Plotly)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from charuco_calibrator.calibration import CalibrationManager, CalibrationResult
from charuco_calibrator.visualize_3d import (
    CalibrationData3D,
    CameraFrustumRenderer,
    CoverageSphereRenderer,
    ExtrinsicViewRenderer,
    ReprojectionQuiverRenderer,
    _build_frustum_scatter_traces,
    _error_to_rgb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_observations(detector, n=6):
    """Generate synthetic observations by rendering the board at different transforms."""
    board_img = detector.generate_board_image(600, 800)
    if len(board_img.shape) == 2:
        board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    else:
        board_bgr = board_img

    observations = []
    h, w = 600, 800

    for i in range(n):
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        offset_x = 20 + i * 15
        offset_y = 10 + i * 10
        bh, bw = board_bgr.shape[:2]

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_camera_matrix():
    return np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)


@pytest.fixture
def sample_dist_coeffs():
    return np.array([[0.1, -0.2, 0, 0, 0.05]], dtype=np.float64)


@pytest.fixture
def sample_data_3d(sample_camera_matrix, sample_dist_coeffs):
    """Create synthetic CalibrationData3D with known geometry."""
    n_views = 5
    n_corners = 12

    cal_result = CalibrationResult(
        rms=0.45,
        camera_matrix=sample_camera_matrix,
        dist_coeffs=sample_dist_coeffs,
        image_size=(640, 480),
        valid=True,
    )

    rng = np.random.RandomState(42)
    object_points = []
    image_points = []
    rvecs = []
    tvecs = []

    for _ in range(n_views):
        obj = np.zeros((n_corners, 1, 3), dtype=np.float32)
        for j in range(n_corners):
            obj[j, 0] = [j % 4 * 0.04, j // 4 * 0.04, 0]
        object_points.append(obj)

        rvec = rng.randn(3, 1).astype(np.float64) * 0.3
        tvec = rng.randn(3, 1).astype(np.float64) * 0.1
        tvec[2] = abs(tvec[2]) + 0.5
        rvecs.append(rvec)
        tvecs.append(tvec)

        projected, _ = cv2.projectPoints(
            obj, rvec, tvec, sample_camera_matrix, sample_dist_coeffs,
        )
        img = projected + rng.randn(*projected.shape).astype(np.float32) * 0.5
        image_points.append(img)

    per_view_errors = rng.rand(n_views) * 2.0

    return CalibrationData3D(
        cal_result=cal_result,
        object_points=object_points,
        image_points=image_points,
        rvecs=rvecs,
        tvecs=tvecs,
        per_view_errors=per_view_errors,
    )


# ---------------------------------------------------------------------------
# CalibrationData3D loading tests
# ---------------------------------------------------------------------------

class TestCalibrationData3DLoad:

    def test_load_with_rvecs_tvecs(self, detector, tmp_path):
        """NPZ with rvecs/tvecs should load directly."""
        mgr = CalibrationManager()
        observations, image_size = _generate_observations(detector, n=8)
        assert len(observations) >= 4
        for obj_pts, img_pts, ids in observations:
            mgr.add_observation(obj_pts, img_pts, ids)
        mgr.calibrate(image_size)

        yaml_path = mgr.save_yaml(tmp_path / "cal.yaml")
        npz_path = mgr.save_observations_npz(tmp_path / "obs.npz")

        data = CalibrationData3D.load(yaml_path, npz_path)
        assert len(data.rvecs) == len(observations)
        assert len(data.tvecs) == len(observations)
        assert data.per_view_errors is not None
        assert len(data.per_view_errors) == len(observations)

    def test_load_without_rvecs_tvecs(self, detector, tmp_path):
        """Old-format NPZ without rvecs/tvecs should recover via solvePnP."""
        mgr = CalibrationManager()
        observations, image_size = _generate_observations(detector, n=8)
        assert len(observations) >= 4
        for obj_pts, img_pts, ids in observations:
            mgr.add_observation(obj_pts, img_pts, ids)
        mgr.calibrate(image_size)

        yaml_path = mgr.save_yaml(tmp_path / "cal.yaml")

        # Save old-format NPZ (without rvecs/tvecs)
        npz_path = tmp_path / "obs_old.npz"
        obj_list = [obs.object_points for obs in mgr.observations]
        img_list = [obs.image_points for obs in mgr.observations]
        id_list = [obs.charuco_ids for obs in mgr.observations]
        np.savez(
            npz_path,
            object_points=np.array(obj_list, dtype=object),
            image_points=np.array(img_list, dtype=object),
            charuco_ids=np.array(id_list, dtype=object),
            image_size=np.array(mgr.image_size),
        )

        data = CalibrationData3D.load(yaml_path, npz_path)
        assert len(data.rvecs) == len(observations)
        assert len(data.tvecs) == len(observations)

    def test_load_missing_yaml(self, tmp_path):
        npz_path = tmp_path / "obs.npz"
        np.savez(
            npz_path,
            object_points=np.array([], dtype=object),
            image_points=np.array([], dtype=object),
            charuco_ids=np.array([], dtype=object),
            image_size=np.array([640, 480]),
        )
        with pytest.raises(FileNotFoundError):
            CalibrationData3D.load("/nonexistent/cal.yaml", npz_path)

    def test_load_missing_npz(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CalibrationData3D.load(tmp_path / "cal.yaml", "/nonexistent/obs.npz")


# ---------------------------------------------------------------------------
# CameraFrustumRenderer tests
# ---------------------------------------------------------------------------

class TestCameraFrustumRenderer:

    def test_frustum_points_shape(self, sample_camera_matrix):
        fr = CameraFrustumRenderer(sample_camera_matrix, (640, 480))
        pts = fr.get_frustum_points()
        assert pts.shape == (5, 3)

    def test_frustum_apex_at_origin(self, sample_camera_matrix):
        fr = CameraFrustumRenderer(sample_camera_matrix, (640, 480))
        pts = fr.get_frustum_points()
        np.testing.assert_array_equal(pts[0], [0, 0, 0])

    def test_frustum_corners_at_positive_z(self, sample_camera_matrix):
        fr = CameraFrustumRenderer(sample_camera_matrix, (640, 480), frustum_scale=0.5)
        pts = fr.get_frustum_points()
        np.testing.assert_allclose(pts[1:, 2], 0.5)

    def test_traces_returns_list(self, sample_camera_matrix):
        fr = CameraFrustumRenderer(sample_camera_matrix, (640, 480))
        result = fr.traces()
        assert isinstance(result, list)
        assert len(result) >= 2  # edges + mesh

    def test_traces_with_transform(self, sample_camera_matrix):
        fr = CameraFrustumRenderer(sample_camera_matrix, (640, 480))
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        result = fr.traces(R=R, t=t)
        assert isinstance(result, list)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Renderer trace tests
# ---------------------------------------------------------------------------

class TestExtrinsicViewRenderer:

    def test_traces_returns_list(self, sample_data_3d):
        renderer = ExtrinsicViewRenderer(sample_data_3d)
        result = renderer.traces()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_traces_contain_board_data(self, sample_data_3d):
        renderer = ExtrinsicViewRenderer(sample_data_3d)
        result = renderer.traces()
        # Should have: frustum edges, frustum mesh, board mesh, board centers, 3 axis traces
        assert len(result) >= 7


class TestReprojectionQuiverRenderer:

    def test_traces_returns_list(self, sample_data_3d):
        renderer = ReprojectionQuiverRenderer(sample_data_3d)
        result = renderer.traces()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_traces_contain_scatter(self, sample_data_3d):
        import plotly.graph_objects as go
        renderer = ReprojectionQuiverRenderer(sample_data_3d)
        result = renderer.traces()
        scatter_traces = [t for t in result if isinstance(t, go.Scatter3d)]
        assert len(scatter_traces) >= 1


class TestCoverageSphereRenderer:

    def test_traces_returns_list(self, sample_data_3d):
        renderer = CoverageSphereRenderer(sample_data_3d)
        result = renderer.traces()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_traces_contain_sphere_and_scatter(self, sample_data_3d):
        import plotly.graph_objects as go
        renderer = CoverageSphereRenderer(sample_data_3d)
        result = renderer.traces()
        scatter_traces = [t for t in result if isinstance(t, go.Scatter3d)]
        # At least: wireframe lines + radial lines + board directions
        assert len(scatter_traces) >= 3


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestBuildFrustumScatterTraces:

    def test_returns_list(self, sample_data_3d):
        result = _build_frustum_scatter_traces(sample_data_3d)
        assert isinstance(result, list)
        assert len(result) >= 3  # frustum edges, mesh, scatter


class TestErrorToRgb:

    def test_min_is_green(self):
        color = _error_to_rgb(0.0, 0.0, 1.0)
        assert "rgb(0," in color

    def test_max_is_red(self):
        color = _error_to_rgb(1.0, 0.0, 1.0)
        assert color.startswith("rgb(255,")

    def test_equal_range_returns_green(self):
        color = _error_to_rgb(0.5, 0.5, 0.5)
        assert color == "rgb(76,175,80)"
