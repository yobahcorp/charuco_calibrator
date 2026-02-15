"""Tests for detector.py — CharucoDetector detection and point matching."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from charuco_calibrator.config import BoardConfig
from charuco_calibrator.detector import CharucoDetectorWrapper, DetectionResult, probe_aruco_dictionaries


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


class TestProbeArucoDictionaries:
    def test_probe_finds_correct_dict(
        self, board_config: BoardConfig, synthetic_board_image: np.ndarray
    ):
        """Probing a DICT_4X4_50 board should return DICT_4X4_50 as top result."""
        results = probe_aruco_dictionaries(synthetic_board_image, board_config)
        assert len(results) > 0
        best_name, best_count = results[0]
        assert best_name == "DICT_4X4_50"
        assert best_count > 0

    def test_probe_small_geometry_limits_ids(
        self, synthetic_board_image: np.ndarray
    ):
        """Probing with smaller board geometry should count fewer valid markers."""
        # The synthetic image has a 5x7 board with markers 0..16.
        # A 3x3 config has max_marker_id = (3*3)//2 = 4, so only IDs 0-3 count.
        small_cfg = BoardConfig(
            squares_x=3, squares_y=3,
            square_length=0.04, marker_length=0.03,
            aruco_dict="DICT_4X4_50",
        )
        correct_cfg = BoardConfig(
            squares_x=5, squares_y=7,
            square_length=0.04, marker_length=0.03,
            aruco_dict="DICT_4X4_50",
        )
        small_results = probe_aruco_dictionaries(synthetic_board_image, small_cfg)
        full_results = probe_aruco_dictionaries(synthetic_board_image, correct_cfg)
        # Both should find something, but small geometry counts fewer
        assert len(full_results) > 0
        full_best = full_results[0][1]
        if small_results:
            small_best = small_results[0][1]
            assert small_best < full_best

    def test_probe_blank_image(self, board_config: BoardConfig, blank_frame: np.ndarray):
        """Probing a blank image should return an empty list."""
        results = probe_aruco_dictionaries(blank_frame, board_config)
        assert results == []


class TestReinitialize:
    def test_reinitialize_switches_detection(self):
        """After reinitialize with a different dict, detection should match the new dict."""
        cfg_4x4 = BoardConfig(
            squares_x=5, squares_y=7,
            square_length=0.04, marker_length=0.03,
            aruco_dict="DICT_4X4_50",
        )
        detector = CharucoDetectorWrapper(cfg_4x4)

        # Generate a 5X5 board image to detect against
        cfg_5x5 = BoardConfig(
            squares_x=5, squares_y=7,
            square_length=0.04, marker_length=0.03,
            aruco_dict="DICT_5X5_100",
        )
        other_detector = CharucoDetectorWrapper(cfg_5x5)
        board_img = other_detector.generate_board_image(600, 800)
        if len(board_img.shape) == 2:
            board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
        padded = np.full((1000, 800, 3), 200, dtype=np.uint8)
        padded[100:100 + board_img.shape[0], 100:100 + board_img.shape[1]] = board_img

        # With wrong dict, detection should fail
        result = detector.detect(padded)
        assert not result.valid

        # After reinitialize, detection should succeed
        detector.reinitialize(cfg_5x5)
        assert detector.board_cfg.aruco_dict == "DICT_5X5_100"
        result = detector.detect(padded)
        assert result.valid
        assert result.num_corners > 0
