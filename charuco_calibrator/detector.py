"""ChArUco board detection wrapper with OpenCV 4.8+ OO API and legacy fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import BoardConfig, resolve_aruco_dict, _ARUCO_DICT_MAP


@dataclass
class DetectionResult:
    """Result of ChArUco corner detection on a single frame."""

    charuco_corners: Optional[np.ndarray] = None  # (N, 1, 2) float32
    charuco_ids: Optional[np.ndarray] = None  # (N, 1) int32
    marker_corners: Optional[list] = None  # list of (4,1,2) arrays
    marker_ids: Optional[np.ndarray] = None  # (M, 1) int32
    num_corners: int = 0
    valid: bool = False


def probe_aruco_dictionaries(
    gray: np.ndarray, board_cfg: BoardConfig
) -> list[tuple[str, int]]:
    """Probe all ArUco dictionaries via marker detection filtered by board geometry.

    Uses fast detectMarkers (works on real camera feeds), then keeps only markers
    whose IDs fall within the valid range for the configured board size.  This
    avoids both false positives from unrelated patterns and the brittleness of
    full detectBoard on noisy real-world frames.

    Returns a list of (dict_name, num_valid_markers) sorted descending by count,
    filtered to count > 0.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Number of markers expected on the board (alternating squares)
    max_marker_id = (board_cfg.squares_x * board_cfg.squares_y) // 2

    results: list[tuple[str, int]] = []
    use_oo = hasattr(cv2.aruco, "ArucoDetector")

    for name, dict_id in _ARUCO_DICT_MAP.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()

        if use_oo:
            ad = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = ad.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=params
            )

        if ids is None or len(ids) == 0:
            continue

        # Only count markers whose IDs are valid for this board geometry
        valid = sum(1 for mid in ids.flatten() if mid < max_marker_id)
        if valid > 0:
            results.append((name, valid))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


class CharucoDetectorWrapper:
    """Wraps OpenCV's CharucoDetector (OO API) with a legacy fallback."""

    def __init__(self, board_cfg: BoardConfig) -> None:
        self.board_cfg = board_cfg
        dict_id = resolve_aruco_dict(board_cfg.aruco_dict)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        self.board = cv2.aruco.CharucoBoard(
            (board_cfg.squares_x, board_cfg.squares_y),
            board_cfg.square_length,
            board_cfg.marker_length,
            self.aruco_dict,
        )

        # Total possible interior corners
        self.total_corners = (board_cfg.squares_x - 1) * (board_cfg.squares_y - 1)

        # Try OO API (OpenCV 4.8+)
        self._use_oo_api = hasattr(cv2.aruco, "CharucoDetector")
        if self._use_oo_api:
            charuco_params = cv2.aruco.CharucoParameters()
            detector_params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.CharucoDetector(
                self.board, charuco_params, detector_params
            )

    def reinitialize(self, board_cfg: BoardConfig) -> None:
        """Re-run init setup with a new BoardConfig (e.g. after dictionary swap)."""
        self.__init__(board_cfg)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detect ChArUco corners in a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if self._use_oo_api:
            return self._detect_oo(gray)
        return self._detect_legacy(gray)

    def _detect_oo(self, gray: np.ndarray) -> DetectionResult:
        """Detection using OpenCV 4.8+ CharucoDetector.detectBoard()."""
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self._detector.detectBoard(gray)
        )

        if charuco_corners is not None and len(charuco_corners) > 0:
            return DetectionResult(
                charuco_corners=charuco_corners,
                charuco_ids=charuco_ids,
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                num_corners=len(charuco_corners),
                valid=True,
            )
        return DetectionResult(
            marker_corners=marker_corners, marker_ids=marker_ids
        )

    def _detect_legacy(self, gray: np.ndarray) -> DetectionResult:
        """Fallback detection using deprecated API."""
        detector_params = cv2.aruco.DetectorParameters()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=detector_params
        )

        if marker_ids is None or len(marker_ids) == 0:
            return DetectionResult()

        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board
        )

        if ret > 0 and charuco_corners is not None:
            return DetectionResult(
                charuco_corners=charuco_corners,
                charuco_ids=charuco_ids,
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                num_corners=ret,
                valid=True,
            )
        return DetectionResult(
            marker_corners=marker_corners, marker_ids=marker_ids
        )

    def match_image_points(
        self, charuco_corners: np.ndarray, charuco_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get matched object points and image points for calibration.

        Uses board.matchImagePoints() (OpenCV 4.8+) or manual lookup.
        """
        if hasattr(self.board, "matchImagePoints"):
            obj_pts, img_pts = self.board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            return obj_pts, img_pts

        # Manual fallback: build object points from board geometry
        all_obj = self.board.getChessboardCorners()
        ids_flat = charuco_ids.flatten()
        obj_pts = np.array([all_obj[i] for i in ids_flat], dtype=np.float32)
        img_pts = charuco_corners.reshape(-1, 1, 2).astype(np.float32)
        return obj_pts.reshape(-1, 1, 3), img_pts

    def generate_board_image(self, width: int = 800, height: int = 1000) -> np.ndarray:
        """Generate a reference image of the ChArUco board."""
        return self.board.generateImage((width, height))

    def draw_detection(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detected corners and markers on a frame (returns a copy)."""
        vis = frame.copy()
        if result.marker_corners and result.marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, result.marker_corners, result.marker_ids)
        if result.valid and result.charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                vis, result.charuco_corners, result.charuco_ids
            )
        return vis
