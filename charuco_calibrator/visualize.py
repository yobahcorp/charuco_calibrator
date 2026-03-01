"""Distortion visualization mode: overlay a distortion heatmap on a live stream."""

from __future__ import annotations

import collections
import sys
import time

import cv2
import numpy as np

from .calibration import CalibrationManager
from .config import AppConfig
from .image_source import create_source
from .ui import Action, UIRenderer

WINDOW_NAME = "ChArUco Calibrator \u2014 Distortion View"


class DistortionHeatmap:
    """Precomputes a distortion magnitude heatmap from calibration parameters.

    The heatmap is computed once and then blended onto every frame.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: tuple[int, int],
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> None:
        self.alpha = alpha
        w, h = image_size
        self._overlay = self._compute_overlay(camera_matrix, dist_coeffs, w, h, colormap)

    @staticmethod
    def _compute_overlay(
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        w: int,
        h: int,
        colormap: int,
    ) -> np.ndarray:
        """Compute a BGR heatmap showing pixel displacement magnitude."""
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            (w, h),
            cv2.CV_32FC1,
        )

        x_grid, y_grid = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        dx = map_x - x_grid
        dy = map_y - y_grid
        magnitude = np.sqrt(dx * dx + dy * dy)

        max_mag = magnitude.max()
        if max_mag > 0.5:  # Skip normalization for sub-pixel noise
            normed = (magnitude / max_mag * 255).astype(np.uint8)
        else:
            normed = np.zeros((h, w), dtype=np.uint8)

        return cv2.applyColorMap(normed, colormap)

    def blend_onto(self, frame: np.ndarray) -> np.ndarray:
        """Alpha-blend the precomputed distortion heatmap onto a BGR frame."""
        h, w = frame.shape[:2]
        overlay = self._overlay
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(frame, 1.0 - self.alpha, overlay, self.alpha, 0)

    @property
    def overlay(self) -> np.ndarray:
        """Raw precomputed overlay."""
        return self._overlay


def run_visualize(cfg: AppConfig, calibration_path: str) -> int:
    """Run the distortion visualization loop.

    Args:
        cfg: Application config (only cfg.source is used).
        calibration_path: Path to the calibration YAML file.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        cal_result = CalibrationManager.load_calibration_yaml(calibration_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    source = create_source(cfg.source)
    if not source.is_open:
        print("ERROR: Could not open image source.", file=sys.stderr)
        return 1

    ok, first_frame = source.read()
    if not ok or first_frame is None:
        print("ERROR: Could not read first frame from source.", file=sys.stderr)
        source.release()
        return 1

    h, w = first_frame.shape[:2]

    distortion_heatmap = DistortionHeatmap(
        camera_matrix=cal_result.camera_matrix,
        dist_coeffs=cal_result.dist_coeffs,
        image_size=(w, h),
    )

    ui = UIRenderer()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    _frame_times: collections.deque[float] = collections.deque(maxlen=30)
    _last_frame = first_frame
    _stream_ended = False
    show_undistort = False
    frame = first_frame

    try:
        while True:
            # Read next frame (skip on first iteration — already have first_frame)
            if frame is not first_frame:
                if not _stream_ended:
                    ok, frame = source.read()
                    if not ok or frame is None:
                        if cfg.source.video_path or cfg.source.image_folder:
                            _stream_ended = True
                            frame = _last_frame
                        else:
                            time.sleep(0.01)
                            continue
                    _last_frame = frame
                else:
                    frame = _last_frame

            now = time.time()
            _frame_times.append(now)
            fps = (
                (len(_frame_times) - 1) / (_frame_times[-1] - _frame_times[0])
                if len(_frame_times) >= 2
                else 0.0
            )

            if show_undistort:
                vis = cv2.undistort(frame, cal_result.camera_matrix, cal_result.dist_coeffs)
            else:
                vis = distortion_heatmap.blend_onto(frame)

            vis = ui.draw_visualize_status_bar(
                vis,
                rms=cal_result.rms,
                fps=fps,
                show_undistort=show_undistort,
                image_size=cal_result.image_size,
            )
            vis = ui.draw_visualize_help_hint(vis)

            cv2.imshow(WINDOW_NAME, vis)

            action = ui.poll_key(wait_ms=1)
            if action == Action.QUIT:
                break
            elif action == Action.UNDISTORT:
                show_undistort = not show_undistort

            # After first frame, allow loop to read next
            if frame is first_frame:
                frame = None

    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0
