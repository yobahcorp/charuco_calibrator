"""3D calibration visualization using Plotly (WebGL)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import CalibrationManager, CalibrationResult


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class CalibrationData3D:
    """All data needed for 3D visualization, loaded from YAML + NPZ."""

    cal_result: CalibrationResult
    object_points: list[np.ndarray]
    image_points: list[np.ndarray]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    per_view_errors: Optional[np.ndarray] = None

    @classmethod
    def load(cls, yaml_path: str | Path, npz_path: str | Path) -> "CalibrationData3D":
        """Load calibration data from a YAML intrinsics file and an NPZ observations file.

        If the NPZ does not contain rvecs/tvecs (old format), recovers them
        via per-view ``cv2.solvePnP``.
        """
        cal_result = CalibrationManager.load_calibration_yaml(yaml_path)
        obs_data = CalibrationManager.load_observations_npz(npz_path)

        object_points = obs_data["object_points"]
        image_points = obs_data["image_points"]

        if "rvecs" in obs_data and "tvecs" in obs_data:
            rvecs = obs_data["rvecs"]
            tvecs = obs_data["tvecs"]
        else:
            rvecs, tvecs = cls._recover_extrinsics(
                object_points,
                image_points,
                cal_result.camera_matrix,
                cal_result.dist_coeffs,
            )

        per_view_errors = cls._compute_per_view_errors(
            object_points, image_points, rvecs, tvecs,
            cal_result.camera_matrix, cal_result.dist_coeffs,
        )

        return cls(
            cal_result=cal_result,
            object_points=object_points,
            image_points=image_points,
            rvecs=rvecs,
            tvecs=tvecs,
            per_view_errors=per_view_errors,
        )

    @staticmethod
    def _recover_extrinsics(
        object_points: list[np.ndarray],
        image_points: list[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Recover per-view rvec/tvec via solvePnP when NPZ lacks them."""
        rvecs: list[np.ndarray] = []
        tvecs: list[np.ndarray] = []
        for obj, img in zip(object_points, image_points):
            ok, rvec, tvec = cv2.solvePnP(
                obj.reshape(-1, 1, 3),
                img.reshape(-1, 1, 2),
                camera_matrix,
                dist_coeffs,
            )
            if ok:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                rvecs.append(np.zeros((3, 1), dtype=np.float64))
                tvecs.append(np.zeros((3, 1), dtype=np.float64))
        return rvecs, tvecs

    @staticmethod
    def _compute_per_view_errors(
        object_points: list[np.ndarray],
        image_points: list[np.ndarray],
        rvecs: list[np.ndarray],
        tvecs: list[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Compute per-view RMS reprojection error."""
        errors: list[float] = []
        for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
            err = float(np.sqrt(np.mean(
                (img.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2
            )))
            errors.append(err)
        return np.array(errors)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _error_to_rgb(value: float, vmin: float, vmax: float) -> str:
    """Map a scalar to a green-yellow-red RGB string."""
    if vmax <= vmin:
        return "rgb(76,175,80)"
    t = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
    r = int(min(255, 510 * t))
    g = int(min(255, 510 * (1 - t)))
    return f"rgb({r},{g},0)"


# ---------------------------------------------------------------------------
# Camera frustum renderer
# ---------------------------------------------------------------------------


class CameraFrustumRenderer:
    """Renders a 3D camera frustum pyramid based on intrinsics."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        image_size: tuple[int, int],
        frustum_scale: float = 0.3,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.image_size = image_size
        self.frustum_scale = frustum_scale

    def get_frustum_points(self) -> np.ndarray:
        """Compute the 5 frustum vertices (apex + 4 image corners) in camera frame.

        Returns:
            (5, 3) array: [apex, top-left, top-right, bottom-right, bottom-left]
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        w, h = self.image_size
        d = self.frustum_scale

        corners_px = np.array([
            [0, 0], [w, 0], [w, h], [0, h],
        ], dtype=np.float64)

        corners_3d = np.zeros((4, 3))
        for i, (px, py) in enumerate(corners_px):
            corners_3d[i] = [(px - cx) / fx * d, (py - cy) / fy * d, d]

        apex = np.array([[0.0, 0.0, 0.0]])
        return np.vstack([apex, corners_3d])

    def traces(self, R=None, t=None, color="royalblue") -> list:
        """Return Plotly traces for the frustum."""
        import plotly.graph_objects as go

        pts = self.get_frustum_points()
        if R is not None and t is not None:
            pts = (R @ pts.T).T + np.asarray(t).reshape(1, 3)

        apex = pts[0]
        corners = pts[1:]
        result = []

        # Edges: apex to corners + image-plane rectangle
        ex, ey, ez = [], [], []
        for c in corners:
            ex.extend([apex[0], c[0], None])
            ey.extend([apex[1], c[1], None])
            ez.extend([apex[2], c[2], None])
        for j in range(4):
            c1, c2 = corners[j], corners[(j + 1) % 4]
            ex.extend([c1[0], c2[0], None])
            ey.extend([c1[1], c2[1], None])
            ez.extend([c1[2], c2[2], None])

        result.append(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="lines", line=dict(color=color, width=4),
            showlegend=False, hoverinfo="skip",
        ))

        # Frustum faces as single mesh: 0=apex, 1-4=corners
        all_pts = np.vstack([apex.reshape(1, 3), corners])
        result.append(go.Mesh3d(
            x=all_pts[:, 0], y=all_pts[:, 1], z=all_pts[:, 2],
            i=[1, 1, 0, 0, 0, 0],
            j=[2, 3, 1, 2, 3, 4],
            k=[3, 4, 2, 3, 4, 1],
            color=color, opacity=0.12, flatshading=True,
            showlegend=False, hoverinfo="skip",
        ))

        return result


# ---------------------------------------------------------------------------
# Extrinsic view renderer
# ---------------------------------------------------------------------------


class ExtrinsicViewRenderer:
    """Renders camera + all board poses in camera coordinates."""

    def __init__(self, data: CalibrationData3D) -> None:
        self.data = data
        self.frustum = CameraFrustumRenderer(
            data.cal_result.camera_matrix,
            data.cal_result.image_size,
        )

    def traces(self) -> list:
        """Return Plotly traces for the extrinsic view."""
        import plotly.graph_objects as go

        result = list(self.frustum.traces())

        errors = self.data.per_view_errors
        has_errors = errors is not None and len(errors) > 0
        if has_errors:
            emin, emax = float(errors.min()), float(errors.max())

        # Batch board faces into one mesh
        all_vx, all_vy, all_vz = [], [], []
        face_i, face_j, face_k = [], [], []
        face_colors: list[str] = []
        board_centers = []
        hover_texts = []

        for idx, (rvec, tvec) in enumerate(zip(self.data.rvecs, self.data.tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            obj_pts = self.data.object_points[idx].reshape(-1, 3)

            corners_obj = np.array([
                [obj_pts[:, 0].min(), obj_pts[:, 1].min(), 0],
                [obj_pts[:, 0].max(), obj_pts[:, 1].min(), 0],
                [obj_pts[:, 0].max(), obj_pts[:, 1].max(), 0],
                [obj_pts[:, 0].min(), obj_pts[:, 1].max(), 0],
            ])
            corners_cam = (R @ corners_obj.T).T + t

            base = len(all_vx)
            for c in corners_cam:
                all_vx.append(float(c[0]))
                all_vy.append(float(c[1]))
                all_vz.append(float(c[2]))
            face_i.extend([base, base])
            face_j.extend([base + 1, base + 2])
            face_k.extend([base + 2, base + 3])

            if has_errors:
                color = _error_to_rgb(errors[idx], emin, emax)
            else:
                color = "rgb(76,175,80)"
            face_colors.extend([color, color])

            center = corners_cam.mean(axis=0)
            board_centers.append(center)
            err_str = f"{errors[idx]:.3f}" if has_errors else "N/A"
            hover_texts.append(f"Board {idx}<br>Error: {err_str} px")

        # Board faces
        result.append(go.Mesh3d(
            x=all_vx, y=all_vy, z=all_vz,
            i=face_i, j=face_j, k=face_k,
            facecolor=face_colors, opacity=0.5, flatshading=True,
            showlegend=False, hoverinfo="skip",
        ))

        # Board center hover markers
        bc = np.array(board_centers)
        result.append(go.Scatter3d(
            x=bc[:, 0], y=bc[:, 1], z=bc[:, 2],
            mode="markers",
            marker=dict(size=4, color="black", opacity=0.7),
            text=hover_texts, hoverinfo="text", showlegend=False,
        ))

        # Board coordinate axes (batched by color)
        for axis_idx, axis_color in enumerate(["red", "green", "blue"]):
            ax_x, ax_y, ax_z = [], [], []
            for idx, rvec in enumerate(self.data.rvecs):
                R, _ = cv2.Rodrigues(rvec)
                center = board_centers[idx]
                end = center + R[:, axis_idx] * 0.03
                ax_x.extend([center[0], end[0], None])
                ax_y.extend([center[1], end[1], None])
                ax_z.extend([center[2], end[2], None])
            result.append(go.Scatter3d(
                x=ax_x, y=ax_y, z=ax_z,
                mode="lines", line=dict(color=axis_color, width=3),
                showlegend=False, hoverinfo="skip",
            ))

        return result


# ---------------------------------------------------------------------------
# Reprojection error quiver plot
# ---------------------------------------------------------------------------


class ReprojectionQuiverRenderer:
    """Renders per-corner reprojection errors as a 3D quiver plot."""

    def __init__(self, data: CalibrationData3D) -> None:
        self.data = data

    def traces(self, scale: float = 50.0) -> list:
        """Return Plotly traces for the reprojection error quiver."""
        import plotly.graph_objects as go

        d = self.data
        all_positions = []
        all_errors_u = []
        all_errors_v = []
        all_magnitudes = []

        for i, (rvec, tvec) in enumerate(zip(d.rvecs, d.tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            obj_pts = d.object_points[i].reshape(-1, 3)
            img_pts = d.image_points[i].reshape(-1, 2)

            projected, _ = cv2.projectPoints(
                d.object_points[i], rvec, tvec,
                d.cal_result.camera_matrix, d.cal_result.dist_coeffs,
            )
            projected = projected.reshape(-1, 2)

            error_px = img_pts - projected
            pts_cam = (R @ obj_pts.T).T + t

            all_positions.append(pts_cam)
            all_errors_u.append(error_px[:, 0])
            all_errors_v.append(error_px[:, 1])
            all_magnitudes.append(np.linalg.norm(error_px, axis=1))

        positions = np.vstack(all_positions)
        errors_u = np.concatenate(all_errors_u)
        errors_v = np.concatenate(all_errors_v)
        magnitudes = np.concatenate(all_magnitudes)

        fx = d.cal_result.camera_matrix[0, 0]
        arrow_scale = scale / fx

        result = []

        # Scatter colored by error magnitude
        result.append(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="markers",
            marker=dict(
                size=2, color=magnitudes, colorscale="Hot",
                showscale=True, colorbar=dict(title="Error (px)", x=1.02),
            ),
            text=[f"Error: {m:.3f} px" for m in magnitudes],
            hoverinfo="text", showlegend=False,
        ))

        # Error arrows as cones
        du = errors_u * arrow_scale
        dv = errors_v * arrow_scale
        dw = np.zeros_like(du)
        mask = magnitudes > 1e-6
        if np.any(mask):
            result.append(go.Cone(
                x=positions[mask, 0], y=positions[mask, 1], z=positions[mask, 2],
                u=du[mask], v=dv[mask], w=dw[mask],
                sizemode="scaled", sizeref=1.5, anchor="tail",
                colorscale="Hot", showscale=False, opacity=0.5,
                hoverinfo="skip", showlegend=False,
            ))

        return result


# ---------------------------------------------------------------------------
# Coverage sphere renderer
# ---------------------------------------------------------------------------


class CoverageSphereRenderer:
    """Visualizes angular and distance coverage of board poses on a sphere."""

    def __init__(self, data: CalibrationData3D) -> None:
        self.data = data

    def traces(self) -> list:
        """Return Plotly traces for the coverage sphere."""
        import plotly.graph_objects as go

        centers = []
        distances = []
        for tvec in self.data.tvecs:
            t = tvec.flatten()
            dist = float(np.linalg.norm(t))
            direction = t / max(dist, 1e-8)
            centers.append(direction)
            distances.append(dist)

        centers = np.array(centers)
        distances = np.array(distances)

        result = []

        # Wireframe sphere: longitude lines
        for lon in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            lat = np.linspace(0, np.pi, 40)
            result.append(go.Scatter3d(
                x=(np.sin(lat) * np.cos(lon)).tolist(),
                y=(np.sin(lat) * np.sin(lon)).tolist(),
                z=np.cos(lat).tolist(),
                mode="lines", line=dict(color="lightgray", width=1),
                opacity=0.15, showlegend=False, hoverinfo="skip",
            ))

        # Latitude lines
        for lat_angle in np.linspace(np.pi * 0.1, np.pi * 0.9, 6):
            lon = np.linspace(0, 2 * np.pi, 40)
            result.append(go.Scatter3d(
                x=(np.sin(lat_angle) * np.cos(lon)).tolist(),
                y=(np.sin(lat_angle) * np.sin(lon)).tolist(),
                z=np.full_like(lon, np.cos(lat_angle)).tolist(),
                mode="lines", line=dict(color="lightgray", width=1),
                opacity=0.15, showlegend=False, hoverinfo="skip",
            ))

        # Radial lines (batched)
        rx, ry, rz = [], [], []
        for c in centers:
            rx.extend([0, float(c[0]), None])
            ry.extend([0, float(c[1]), None])
            rz.extend([0, float(c[2]), None])
        result.append(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode="lines", line=dict(color="gray", width=1),
            opacity=0.3, showlegend=False, hoverinfo="skip",
        ))

        # Board direction scatter
        result.append(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers",
            marker=dict(
                size=8, color=distances, colorscale="Viridis",
                showscale=True, colorbar=dict(title="Distance (m)", x=1.02),
                line=dict(color="black", width=1),
            ),
            text=[f"Pose {i}<br>Distance: {d:.3f} m" for i, d in enumerate(distances)],
            hoverinfo="text", showlegend=False,
        ))

        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_VIEW_NAMES = [
    "1: Extrinsic View",
    "2: Reprojection Quiver",
    "3: Camera Frustum",
    "4: Coverage Sphere",
]


def _build_frustum_scatter_traces(data: CalibrationData3D) -> list:
    """Build traces for view 3: camera frustum + board center scatter."""
    import plotly.graph_objects as go

    frustum = CameraFrustumRenderer(
        data.cal_result.camera_matrix, data.cal_result.image_size,
    )
    result = list(frustum.traces())

    board_centers = np.array([t.flatten() for t in data.tvecs])
    distances = np.linalg.norm(board_centers, axis=1)

    result.append(go.Scatter3d(
        x=board_centers[:, 0], y=board_centers[:, 1], z=board_centers[:, 2],
        mode="markers",
        marker=dict(
            size=6, color=distances, colorscale="Reds",
            showscale=True, colorbar=dict(title="Distance (m)", x=1.02),
            line=dict(color="black", width=1),
        ),
        text=[f"Board {i}<br>Distance: {d:.3f} m" for i, d in enumerate(distances)],
        hoverinfo="text", showlegend=False,
    ))

    return result


def run_visualize_3d(calibration_path: str, observations_path: str) -> int:
    """Entry point for 3D visualization mode.

    Opens an interactive Plotly WebGL figure in the browser with buttons
    to switch between four views.

    Args:
        calibration_path: Path to calibration YAML file.
        observations_path: Path to observations NPZ file.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    import plotly.graph_objects as go

    try:
        data = CalibrationData3D.load(calibration_path, observations_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Build traces for each view
    view_traces = [
        ExtrinsicViewRenderer(data).traces(),
        ReprojectionQuiverRenderer(data).traces(),
        _build_frustum_scatter_traces(data),
        CoverageSphereRenderer(data).traces(),
    ]

    fig = go.Figure()
    trace_counts = []
    for view_idx, vt in enumerate(view_traces):
        for tr in vt:
            tr.visible = (view_idx == 0)
            fig.add_trace(tr)
        trace_counts.append(len(vt))

    # Scene layout per view
    scene_meters = dict(
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
        aspectmode="data",
    )
    scene_unit = dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
        aspectmode="cube",
    )
    scene_configs = [scene_meters, scene_meters, scene_meters, scene_unit]

    # Build visibility masks and buttons
    buttons = []
    for view_idx, name in enumerate(_VIEW_NAMES):
        visible = []
        for vi, count in enumerate(trace_counts):
            visible.extend([vi == view_idx] * count)
        buttons.append(dict(
            label=name,
            method="update",
            args=[{"visible": visible}, {"scene": scene_configs[view_idx]}],
        ))

    rms = data.cal_result.rms
    n_views = len(data.rvecs)
    w, h = data.cal_result.image_size

    fig.update_layout(
        title=dict(
            text=(
                f"3D Calibration Analysis  |  RMS: {rms:.4f} px  |  "
                f"{n_views} views  |  {w}x{h}"
            ),
            x=0.5, xanchor="center",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="down",
            x=0.0, xanchor="left",
            y=0.95, yanchor="top",
            buttons=buttons,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            pad=dict(l=10, t=10),
        )],
        scene=scene_configs[0],
        margin=dict(l=180, r=0, t=60, b=0),
    )

    fig.show()
    return 0
