"""Camera calibration manager: cv2.calibrateCamera wrapper, YAML/NPZ export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclass
class Observation:
    """A single accepted frame's calibration data."""

    object_points: np.ndarray  # (N, 1, 3) float32
    image_points: np.ndarray  # (N, 1, 2) float32
    charuco_ids: np.ndarray  # (N, 1) int32
    frame_index: int = 0


@dataclass
class CalibrationResult:
    """Result of cv2.calibrateCamera."""

    rms: float = 0.0
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    rvecs: Optional[list] = None
    tvecs: Optional[list] = None
    per_view_errors: Optional[np.ndarray] = None
    image_size: tuple[int, int] = (0, 0)  # (width, height)
    valid: bool = False


class CalibrationManager:
    """Accumulates observations and runs cv2.calibrateCamera."""

    def __init__(self, image_size: tuple[int, int] = (0, 0)) -> None:
        """
        Args:
            image_size: (width, height) of the calibration images.
        """
        self.image_size = image_size
        self.observations: list[Observation] = []
        self.result: Optional[CalibrationResult] = None
        self._frame_counter = 0

    def add_observation(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        charuco_ids: np.ndarray,
    ) -> int:
        """Add a calibration observation. Returns the observation index."""
        obs = Observation(
            object_points=object_points,
            image_points=image_points,
            charuco_ids=charuco_ids,
            frame_index=self._frame_counter,
        )
        self.observations.append(obs)
        self._frame_counter += 1
        return len(self.observations) - 1

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    def calibrate(self, image_size: Optional[tuple[int, int]] = None) -> CalibrationResult:
        """Run cv2.calibrateCamera on all accumulated observations.

        Args:
            image_size: (width, height). If None, uses the stored image_size.

        Returns:
            CalibrationResult with calibration parameters.
        """
        if image_size is not None:
            self.image_size = image_size

        if len(self.observations) < 4:
            return CalibrationResult()

        obj_points_list = [obs.object_points for obs in self.observations]
        img_points_list = [obs.image_points for obs in self.observations]

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list,
            img_points_list,
            self.image_size,
            None,
            None,
        )

        result = CalibrationResult(
            rms=rms,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
            image_size=self.image_size,
            valid=True,
        )

        result.per_view_errors = self.compute_per_view_errors(result)
        self.result = result
        return result

    @staticmethod
    def compute_per_view_errors(result: CalibrationResult) -> np.ndarray:
        """Compute per-view RMS reprojection error."""
        if not result.valid or result.rvecs is None:
            return np.array([])

        errors = []
        # This requires access to observations — handled via the stored result
        return np.array(errors)

    def compute_per_view_errors_full(self) -> np.ndarray:
        """Compute per-view RMS using stored observations and result."""
        if self.result is None or not self.result.valid:
            return np.array([])

        errors = []
        for i, obs in enumerate(self.observations):
            projected, _ = cv2.projectPoints(
                obs.object_points,
                self.result.rvecs[i],
                self.result.tvecs[i],
                self.result.camera_matrix,
                self.result.dist_coeffs,
            )
            err = np.sqrt(
                np.mean((obs.image_points.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2)
            )
            errors.append(err)
        return np.array(errors)

    def save_yaml(
        self,
        path: str | Path,
        camera_name: str = "camera",
    ) -> Path:
        """Save calibration result in ROS camera_info_manager compatible YAML.

        Args:
            path: Output file path.
            camera_name: Camera name for the YAML header.

        Returns:
            The Path that was written.
        """
        if self.result is None or not self.result.valid:
            raise RuntimeError("No valid calibration result to save.")

        r = self.result
        w, h = r.image_size

        # Build distortion model string
        n_dist = len(r.dist_coeffs.flatten())
        if n_dist <= 5:
            distortion_model = "plumb_bob"
        else:
            distortion_model = "rational_polynomial"

        # Rectification matrix (identity for monocular)
        rect_matrix = np.eye(3, dtype=np.float64)

        # Projection matrix: [K | 0] for monocular
        proj_matrix = np.zeros((3, 4), dtype=np.float64)
        proj_matrix[:3, :3] = r.camera_matrix

        data = {
            "image_width": w,
            "image_height": h,
            "camera_name": camera_name,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": r.camera_matrix.flatten().tolist(),
            },
            "distortion_model": distortion_model,
            "distortion_coefficients": {
                "rows": 1,
                "cols": n_dist,
                "data": r.dist_coeffs.flatten().tolist(),
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": rect_matrix.flatten().tolist(),
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": proj_matrix.flatten().tolist(),
            },
            "rms_reprojection_error": float(r.rms),
        }

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return out

    def save_observations_npz(self, path: str | Path) -> Path:
        """Save all observations as an NPZ archive for reproducibility."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        obj_list = [obs.object_points for obs in self.observations]
        img_list = [obs.image_points for obs in self.observations]
        id_list = [obs.charuco_ids for obs in self.observations]

        np.savez(
            out,
            object_points=np.array(obj_list, dtype=object),
            image_points=np.array(img_list, dtype=object),
            charuco_ids=np.array(id_list, dtype=object),
            image_size=np.array(self.image_size),
        )
        return out

    def reset(self) -> None:
        """Clear all observations and results."""
        self.observations.clear()
        self.result = None
        self._frame_counter = 0
