"""Camera calibration manager: cv2.calibrateCamera wrapper, YAML/NPZ export."""

from __future__ import annotations

import threading
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
        self._lock = threading.Lock()
        self._calibrating = False
        self._cal_thread: Optional[threading.Thread] = None

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
        with self._lock:
            self.observations.append(obs)
        self._frame_counter += 1
        return len(self.observations) - 1

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    def calibrate_async(self, image_size: Optional[tuple[int, int]] = None) -> None:
        """Run calibration in a background thread (non-blocking).

        The result is stored in ``self.result`` once complete.
        Check ``self.is_calibrating`` to know when it finishes.
        """
        if self._calibrating:
            return

        if image_size is not None:
            self.image_size = image_size

        # Snapshot observations under lock so the main thread can keep appending
        with self._lock:
            obs_copy = list(self.observations)

        if len(obs_copy) < 4:
            return

        self._calibrating = True

        def _run() -> None:
            try:
                obj_points_list = [obs.object_points for obs in obs_copy]
                img_points_list = [obs.image_points for obs in obs_copy]

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
                self.result = result
            finally:
                self._calibrating = False

        self._cal_thread = threading.Thread(target=_run, daemon=True)
        self._cal_thread.start()

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
        """Compute per-view RMS using stored observations and result.

        Only computes errors for observations that have corresponding
        rvecs/tvecs in the result (observations added after calibration
        are excluded).
        """
        if self.result is None or not self.result.valid:
            return np.array([])
        if self.result.rvecs is None or self.result.tvecs is None:
            return np.array([])

        n = min(len(self.observations), len(self.result.rvecs))
        errors = []
        for i in range(n):
            obs = self.observations[i]
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

    def pop_observation(self) -> Optional[Observation]:
        """Remove and return the last observation, or None if empty."""
        with self._lock:
            if not self.observations:
                return None
            return self.observations.pop()

    def prune_outliers(
        self,
        threshold: float = 2.0,
        max_iterations: int = 3,
    ) -> tuple[int, float, float]:
        """Remove observations with per-view error > threshold * mean RMS.

        Recalibrates after each round of pruning until no outliers remain
        or max_iterations is reached.

        Returns:
            (num_pruned, old_rms, new_rms)
        """
        if self.result is None or not self.result.valid:
            return (0, 0.0, 0.0)

        old_rms = self.result.rms
        total_pruned = 0

        for _ in range(max_iterations):
            errors = self.compute_per_view_errors_full()
            if len(errors) == 0:
                break

            mean_err = float(np.mean(errors))
            cutoff = threshold * mean_err
            outlier_mask = errors > cutoff

            if not np.any(outlier_mask):
                break

            # Remove outliers (iterate in reverse to preserve indices)
            indices_to_remove = sorted(np.where(outlier_mask)[0], reverse=True)
            with self._lock:
                for idx in indices_to_remove:
                    self.observations.pop(idx)
            total_pruned += len(indices_to_remove)

            # Re-calibrate with remaining observations
            if len(self.observations) < 4:
                break
            self.calibrate(self.image_size)

        new_rms = self.result.rms if self.result and self.result.valid else old_rms
        return (total_pruned, old_rms, new_rms)

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
        """Save all observations (and rvecs/tvecs if available) as an NPZ archive."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        obj_list = [obs.object_points for obs in self.observations]
        img_list = [obs.image_points for obs in self.observations]
        id_list = [obs.charuco_ids for obs in self.observations]

        save_kwargs: dict = dict(
            object_points=np.array(obj_list, dtype=object),
            image_points=np.array(img_list, dtype=object),
            charuco_ids=np.array(id_list, dtype=object),
            image_size=np.array(self.image_size),
        )

        # Persist extrinsics when available (needed for 3D visualization)
        if (
            self.result is not None
            and self.result.valid
            and self.result.rvecs is not None
            and self.result.tvecs is not None
        ):
            save_kwargs["rvecs"] = np.array(
                [r.flatten() for r in self.result.rvecs], dtype=np.float64
            )
            save_kwargs["tvecs"] = np.array(
                [t.flatten() for t in self.result.tvecs], dtype=np.float64
            )

        np.savez(out, **save_kwargs)
        return out

    @staticmethod
    def load_observations_npz(path: str | Path) -> dict:
        """Load observations (and optionally rvecs/tvecs) from an NPZ archive.

        Returns:
            Dict with keys: ``object_points``, ``image_points``,
            ``charuco_ids``, ``image_size``, and optionally ``rvecs``,
            ``tvecs``.

        Raises:
            FileNotFoundError: If the NPZ file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Observations file not found: {p}")

        data = np.load(p, allow_pickle=True)
        result: dict = {
            "object_points": list(data["object_points"]),
            "image_points": list(data["image_points"]),
            "charuco_ids": list(data["charuco_ids"]),
            "image_size": tuple(data["image_size"]),
        }
        if "rvecs" in data:
            result["rvecs"] = [r.reshape(3, 1) for r in data["rvecs"]]
        if "tvecs" in data:
            result["tvecs"] = [t.reshape(3, 1) for t in data["tvecs"]]
        return result

    def reset(self) -> None:
        """Clear all observations and results."""
        self.observations.clear()
        self.result = None
        self._frame_counter = 0

    @staticmethod
    def load_calibration_yaml(path: str | Path) -> CalibrationResult:
        """Load a calibration result from a ROS camera_info_manager YAML file.

        Args:
            path: Path to the calibration YAML file.

        Returns:
            CalibrationResult with camera_matrix, dist_coeffs, and image_size.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If required fields are missing.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Calibration file not found: {p}")

        with open(p) as f:
            data = yaml.safe_load(f)

        cm = data["camera_matrix"]
        camera_matrix = np.array(cm["data"], dtype=np.float64).reshape(cm["rows"], cm["cols"])

        dc = data["distortion_coefficients"]
        dist_coeffs = np.array(dc["data"], dtype=np.float64).reshape(dc["rows"], dc["cols"])

        image_size = (int(data["image_width"]), int(data["image_height"]))
        rms = float(data.get("rms_reprojection_error", 0.0))

        return CalibrationResult(
            rms=rms,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=image_size,
            valid=True,
        )
