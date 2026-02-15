"""Shared fixtures for ChArUco calibrator tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from charuco_calibrator.config import BoardConfig
from charuco_calibrator.detector import CharucoDetectorWrapper


@pytest.fixture
def board_config() -> BoardConfig:
    """Default board configuration for tests."""
    return BoardConfig(
        squares_x=5,
        squares_y=7,
        square_length=0.04,
        marker_length=0.03,
        aruco_dict="DICT_4X4_50",
    )


@pytest.fixture
def detector(board_config: BoardConfig) -> CharucoDetectorWrapper:
    """CharucoDetector wrapper instance."""
    return CharucoDetectorWrapper(board_config)


@pytest.fixture
def synthetic_board_image(detector: CharucoDetectorWrapper) -> np.ndarray:
    """A synthetic ChArUco board image (grayscale rendered then converted to BGR).

    This simulates a 'perfect' view of the board — useful for testing detection.
    """
    board_img = detector.generate_board_image(600, 800)
    # Add white border to simulate the board inside a larger frame
    padded = np.full((1000, 800, 3), 200, dtype=np.uint8)
    # Convert board to BGR if needed
    if len(board_img.shape) == 2:
        board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    else:
        board_bgr = board_img
    # Place board in center with some margin
    y_off = 100
    x_off = 100
    bh, bw = board_bgr.shape[:2]
    padded[y_off : y_off + bh, x_off : x_off + bw] = board_bgr
    return padded


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A plain gray 640x480 BGR frame with no board."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)
