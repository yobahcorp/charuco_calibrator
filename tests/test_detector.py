"""Tests for detector.py — CharucoDetector detection and point matching."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from charuco_calibrator.config import BoardConfig
from charuco_calibrator.detector import CharucoDetectorWrapper, DetectionResult


class TestCharucoDetector:
    def test_total_corners(self, detector: CharucoDetectorWrapper):
        """Total corners should be (squares_x-1) * (squares_y-1)."""
        assert detector.total_corners == 4 * 6  # 5-1 * 7-1

    def test_detect_on_board_image(
        self, detector: CharucoDetectorWrapper, synthetic_board_image: np.ndarray
    ):
        """Detection on a synthetic board image should find corners."""
        result = detector.detect(synthetic_board_image)
        assert result.valid
        assert result.num_corners > 0
        assert result.charuco_corners is not None
        assert result.charuco_ids is not None

    def test_detect_on_blank_frame(
        self, detector: CharucoDetectorWrapper, blank_frame: np.ndarray
    ):
        """Detection on a blank image should not find corners."""
        result = detector.detect(blank_frame)
        assert not result.valid
        assert result.num_corners == 0

    def test_match_image_points(
        self, detector: CharucoDetectorWrapper, synthetic_board_image: np.ndarray
    ):
        """match_image_points should return object and image points."""
        result = detector.detect(synthetic_board_image)
        assert result.valid

        obj_pts, img_pts = detector.match_image_points(
            result.charuco_corners, result.charuco_ids
        )
        assert obj_pts.shape[0] == img_pts.shape[0]
        assert obj_pts.shape[0] > 0

    def test_generate_board_image(self, detector: CharucoDetectorWrapper):
        """generate_board_image should return a valid image."""
        img = detector.generate_board_image(400, 500)
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_draw_detection(
        self, detector: CharucoDetectorWrapper, synthetic_board_image: np.ndarray
    ):
        """draw_detection should return an image of same size."""
        result = detector.detect(synthetic_board_image)
        vis = detector.draw_detection(synthetic_board_image, result)
        assert vis.shape == synthetic_board_image.shape

    def test_draw_detection_no_result(
        self, detector: CharucoDetectorWrapper, blank_frame: np.ndarray
    ):
        """draw_detection with empty result should not crash."""
        result = DetectionResult()
        vis = detector.draw_detection(blank_frame, result)
        assert vis.shape == blank_frame.shape
