"""Tests for scoring.py — blur, coverage, and frame scoring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from charuco_calibrator.detector import CharucoDetectorWrapper, DetectionResult
from charuco_calibrator.scoring import (
    CoverageState,
    FrameScore,
    compute_blur_variance,
    compute_convex_hull_area,
    compute_new_grid_coverage,
    score_frame,
)


class TestBlurVariance:
    def test_sharp_image_has_high_variance(self):
        """A checkerboard-like image should have high Laplacian variance."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[::2, :] = 255  # Horizontal stripes
        var = compute_blur_variance(img)
        assert var > 100

    def test_uniform_image_has_low_variance(self):
        """A uniform gray image should have near-zero variance."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        var = compute_blur_variance(img)
        assert var < 1.0


class TestConvexHullArea:
    def test_full_frame_corners(self):
        """Corners at image extremes should give ratio near 1.0."""
        corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)
        ratio = compute_convex_hull_area(corners, 640 * 480)
        assert ratio > 0.95

    def test_small_cluster(self):
        """A small cluster of corners should have small ratio."""
        corners = np.array([[100, 100], [110, 100], [105, 110]], dtype=np.float32)
        ratio = compute_convex_hull_area(corners, 640 * 480)
        assert ratio < 0.01

    def test_too_few_points(self):
        """Fewer than 3 points should return 0."""
        corners = np.array([[100, 100], [200, 200]], dtype=np.float32)
        ratio = compute_convex_hull_area(corners, 640 * 480)
        assert ratio == 0.0


class TestCoverageState:
    def test_initial_state(self):
        cov = CoverageState(grid_cols=4, grid_rows=4, scale_bins=5)
        assert cov.grid_coverage == 0.0
        assert cov.quadrant_coverage == 0.0
        assert cov.quality_meter == 0.0

    def test_update_increases_coverage(self):
        cov = CoverageState(grid_cols=4, grid_rows=4, scale_bins=5)
        # Corners spread across the frame
        corners = np.array(
            [[80, 60], [320, 60], [560, 60], [80, 420], [320, 240], [560, 420]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        cov.update(corners, 480, 640)
        assert cov.grid_coverage > 0.0
        assert cov.total_frames == 1

    def test_quadrant_coverage(self):
        cov = CoverageState(grid_cols=4, grid_rows=4, scale_bins=5)
        # One corner per quadrant
        corners = np.array(
            [[100, 100], [500, 100], [100, 400], [500, 400]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        cov.update(corners, 480, 640)
        assert cov.quadrant_coverage == 1.0

    def test_reset(self):
        cov = CoverageState(grid_cols=4, grid_rows=4, scale_bins=5)
        corners = np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2)
        cov.update(corners, 480, 640)
        assert cov.total_frames == 1
        cov.reset()
        assert cov.total_frames == 0
        assert cov.grid_coverage == 0.0

    def test_quality_meter_range(self):
        cov = CoverageState(grid_cols=4, grid_rows=4, scale_bins=5)
        # Add multiple diverse frames
        for _ in range(10):
            corners = np.random.uniform(
                low=[0, 0], high=[640, 480], size=(20, 2)
            ).astype(np.float32).reshape(-1, 1, 2)
            cov.update(corners, 480, 640)
        assert 0.0 <= cov.quality_meter <= 1.0


class TestNewGridCoverage:
    def test_first_frame_has_new_coverage(self):
        grid = np.zeros((4, 4), dtype=np.int32)
        corners = np.array([[80, 60], [320, 240]], dtype=np.float32).reshape(-1, 1, 2)
        new_cov = compute_new_grid_coverage(corners, 480, 640, grid, 4, 4)
        assert new_cov > 0.0

    def test_already_covered_cells(self):
        grid = np.ones((4, 4), dtype=np.int32)  # All already hit
        corners = np.array([[80, 60]], dtype=np.float32).reshape(-1, 1, 2)
        new_cov = compute_new_grid_coverage(corners, 480, 640, grid, 4, 4)
        assert new_cov == 0.0


class TestScoreFrame:
    def test_no_detection(self):
        result = DetectionResult()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cov = CoverageState()
        score = score_frame(result, frame, cov, total_corners=24)
        assert score.rejected
        assert "no detection" in score.reject_reason

    def test_valid_detection(
        self, detector: CharucoDetectorWrapper, synthetic_board_image: np.ndarray
    ):
        result = detector.detect(synthetic_board_image)
        cov = CoverageState()
        score = score_frame(
            result,
            synthetic_board_image,
            cov,
            total_corners=detector.total_corners,
            min_corners=4,
            min_blur_var=10.0,
            min_score=0.0,
        )
        assert not score.rejected
        assert score.total > 0
