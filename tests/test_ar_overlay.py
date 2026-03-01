"""Tests for ar_overlay.py — pose estimation, geometry, and rendering."""

from __future__ import annotations

import numpy as np
import pytest

import cv2

from charuco_calibrator.ar_overlay import (
    AROverlay,
    PoseResult,
    PoseSmoother,
    estimate_board_pose,
    load_obj_mesh,
    _make_axes_geometry,
    _make_wireframe_cube_geometry,
    _make_solid_cube_faces,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_intrinsics():
    """Realistic camera intrinsics."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    D = np.zeros((1, 5), dtype=np.float64)
    return K, D


@pytest.fixture
def sample_pose_data(sample_intrinsics):
    """Matched object/image points that produce a valid solvePnP result."""
    K, D = sample_intrinsics
    obj_pts = np.array(
        [
            [0, 0, 0],
            [0.04, 0, 0],
            [0.08, 0, 0],
            [0, 0.04, 0],
            [0.04, 0.04, 0],
            [0.08, 0.04, 0],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 3)

    rvec = np.array([[0.1], [0.2], [0.05]], dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float64)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    return obj_pts, img_pts.reshape(-1, 1, 2).astype(np.float32), K, D


# ---------------------------------------------------------------------------
# PoseSmoother
# ---------------------------------------------------------------------------


class TestPoseSmoother:
    def test_first_update_returns_input(self):
        smoother = PoseSmoother(alpha=0.5)
        rvec = np.array([[0.1], [0.2], [0.3]])
        tvec = np.array([[1.0], [2.0], [3.0]])
        sr, st = smoother.update(rvec, tvec)
        np.testing.assert_array_equal(sr, rvec)
        np.testing.assert_array_equal(st, tvec)

    def test_smoothing_dampens(self):
        smoother = PoseSmoother(alpha=0.5)
        r1 = np.array([[0.0], [0.0], [0.0]])
        t1 = np.array([[0.0], [0.0], [1.0]])
        smoother.update(r1, t1)

        r2 = np.array([[1.0], [1.0], [1.0]])
        t2 = np.array([[1.0], [1.0], [2.0]])
        sr, st = smoother.update(r2, t2)
        np.testing.assert_allclose(sr, [[0.5], [0.5], [0.5]])

    def test_reset_clears_state(self):
        smoother = PoseSmoother(alpha=0.5)
        smoother.update(np.zeros((3, 1)), np.zeros((3, 1)))
        smoother.reset()
        assert smoother._rvec is None
        assert smoother._tvec is None


# ---------------------------------------------------------------------------
# estimate_board_pose
# ---------------------------------------------------------------------------


class TestEstimateBoardPose:
    def test_valid_pose(self, sample_pose_data):
        obj_pts, img_pts, K, D = sample_pose_data
        result = estimate_board_pose(obj_pts, img_pts, K, D)
        assert result.success
        assert result.rvec.shape == (3, 1)
        assert result.tvec.shape == (3, 1)

    def test_insufficient_points(self, sample_intrinsics):
        K, D = sample_intrinsics
        obj = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32).reshape(
            -1, 1, 3
        )
        img = np.array([[100, 100], [200, 100]], dtype=np.float32).reshape(
            -1, 1, 2
        )
        result = estimate_board_pose(obj, img, K, D)
        assert not result.success

    def test_pose_recovers_tvec(self, sample_pose_data):
        obj_pts, img_pts, K, D = sample_pose_data
        result = estimate_board_pose(obj_pts, img_pts, K, D)
        # The original tvec had z=0.5
        assert abs(result.tvec[2, 0] - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_axes_geometry(self):
        verts, edges, colors = _make_axes_geometry(1.0, 0.04)
        assert verts.shape == (4, 3)
        assert len(edges) == 3
        assert len(colors) == 3

    def test_wireframe_cube(self):
        verts, edges = _make_wireframe_cube_geometry(1.0, 0.04)
        assert verts.shape == (8, 3)
        assert len(edges) == 12

    def test_solid_cube(self):
        verts, faces = _make_solid_cube_faces(1.0, 0.04)
        assert verts.shape == (8, 3)
        assert len(faces) == 6

    def test_scale_affects_geometry(self):
        v1, _ = _make_wireframe_cube_geometry(1.0, 0.04)
        v2, _ = _make_wireframe_cube_geometry(2.0, 0.04)
        # Scale 2 should produce vertices twice as far from origin
        np.testing.assert_allclose(v2, v1 * 2.0)


# ---------------------------------------------------------------------------
# load_obj_mesh
# ---------------------------------------------------------------------------


class TestLoadObjMesh:
    def test_load_simple_obj(self, tmp_path):
        obj_file = tmp_path / "cube.obj"
        obj_file.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "f 1 2 3 4\n"
        )
        verts, faces = load_obj_mesh(obj_file)
        assert verts.shape == (4, 3)
        assert len(faces) == 1
        assert faces[0] == (0, 1, 2, 3)  # 0-indexed

    def test_load_with_vt_format(self, tmp_path):
        obj_file = tmp_path / "tri.obj"
        obj_file.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "f 1/1 2/2 3/3\n"
        )
        verts, faces = load_obj_mesh(obj_file)
        assert verts.shape == (3, 3)
        assert faces[0] == (0, 1, 2)


# ---------------------------------------------------------------------------
# AROverlay
# ---------------------------------------------------------------------------


class TestAROverlay:
    def test_wireframe_init(self):
        ar = AROverlay(ar_object="wireframe", scale=1.0, square_length=0.04)
        assert ar._obj_verts is not None
        assert ar._obj_edges is not None

    def test_axes_init(self):
        ar = AROverlay(ar_object="axes", scale=1.0, square_length=0.04)
        assert ar._obj_verts is not None
        assert ar._edge_colors is not None

    def test_solid_init(self):
        ar = AROverlay(ar_object="solid", scale=1.0, square_length=0.04)
        assert ar._obj_verts is not None
        assert ar._obj_faces is not None
        assert ar._obj_edges is not None

    def test_render_no_pose_returns_unchanged(self):
        ar = AROverlay(ar_object="wireframe", scale=1.0, square_length=0.04)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        pose = PoseResult(
            rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), success=False
        )
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((1, 5), dtype=np.float64)
        result = ar.render(frame, pose, K, D)
        np.testing.assert_array_equal(result, frame)

    def test_render_valid_pose_modifies_frame(self, sample_pose_data):
        obj_pts, img_pts, K, D = sample_pose_data
        ar = AROverlay(ar_object="wireframe", scale=1.0, square_length=0.04)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        pose = estimate_board_pose(obj_pts, img_pts, K, D)
        result = ar.render(frame, pose, K, D)
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_render_axes_modifies_frame(self, sample_pose_data):
        obj_pts, img_pts, K, D = sample_pose_data
        ar = AROverlay(ar_object="axes", scale=1.0, square_length=0.04)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        pose = estimate_board_pose(obj_pts, img_pts, K, D)
        result = ar.render(frame, pose, K, D)
        assert not np.array_equal(result, frame)

    def test_render_solid_modifies_frame(self, sample_pose_data):
        obj_pts, img_pts, K, D = sample_pose_data
        ar = AROverlay(ar_object="solid", scale=1.0, square_length=0.04)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        pose = estimate_board_pose(obj_pts, img_pts, K, D)
        result = ar.render(frame, pose, K, D)
        assert not np.array_equal(result, frame)

    def test_obj_requires_path(self):
        with pytest.raises(ValueError, match="--ar-obj-path"):
            AROverlay(ar_object="obj", scale=1.0, square_length=0.04)

    def test_reset_clears_smoother(self):
        ar = AROverlay(ar_object="wireframe", scale=1.0, square_length=0.04)
        ar._smoother.update(np.zeros((3, 1)), np.zeros((3, 1)))
        ar.reset()
        assert ar._smoother._rvec is None
