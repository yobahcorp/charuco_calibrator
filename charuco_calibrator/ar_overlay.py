"""AR overlay: project 3D objects onto a detected ChArUco board."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class PoseResult:
    """Board pose estimation result."""

    rvec: np.ndarray  # (3,1) rotation vector
    tvec: np.ndarray  # (3,1) translation vector
    success: bool = False


class PoseSmoother:
    """Exponential moving average filter for rvec/tvec to reduce jitter."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None

    def update(
        self, rvec: np.ndarray, tvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply EMA smoothing. Returns smoothed (rvec, tvec)."""
        if self._rvec is None:
            self._rvec = rvec.copy()
            self._tvec = tvec.copy()
        else:
            self._rvec = self.alpha * rvec + (1.0 - self.alpha) * self._rvec
            self._tvec = self.alpha * tvec + (1.0 - self.alpha) * self._tvec
        return self._rvec.copy(), self._tvec.copy()

    def reset(self) -> None:
        self._rvec = None
        self._tvec = None


def estimate_board_pose(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> PoseResult:
    """Estimate the board pose using cv2.solvePnP.

    Args:
        obj_pts: (N,1,3) or (N,3) matched object points.
        img_pts: (N,1,2) or (N,2) matched image points.
        camera_matrix: 3x3 camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.

    Returns:
        PoseResult with rvec, tvec, and success flag.
    """
    obj = obj_pts.reshape(-1, 1, 3).astype(np.float64)
    img = img_pts.reshape(-1, 1, 2).astype(np.float64)

    if len(obj) < 4:
        return PoseResult(
            rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), success=False
        )

    success, rvec, tvec = cv2.solvePnP(
        obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return PoseResult(rvec=rvec, tvec=tvec, success=bool(success))


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Load a Wavefront OBJ file (vertices + face indices).

    Returns:
        (vertices, faces) where vertices is (V,3) float32 and
        faces is a list of tuples of 0-based vertex indices.
    """
    vertices: list[list[float]] = []
    faces: list[tuple[int, ...]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif parts[0] == "f":
                face_indices = []
                for token in parts[1:]:
                    idx = int(token.split("/")[0]) - 1  # OBJ is 1-indexed
                    face_indices.append(idx)
                faces.append(tuple(face_indices))
    return np.array(vertices, dtype=np.float32), faces


def _make_axes_geometry(
    scale: float, square_length: float
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Generate 3D coordinate axes geometry.

    Returns:
        (vertices, edges, edge_colors_bgr)
    """
    s = scale * square_length
    verts = np.array(
        [
            [0, 0, 0],  # origin
            [s, 0, 0],  # X tip
            [0, s, 0],  # Y tip
            [0, 0, -s],  # Z tip (negative Z = up from board)
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (0, 2), (0, 3)]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # R, G, B in BGR
    return verts, edges, colors


def _make_wireframe_cube_geometry(
    scale: float, square_length: float
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate a wireframe cube sitting on the board surface.

    The cube base is on the Z=0 plane, top at Z=-s (above board).

    Returns:
        (vertices, edges)
    """
    s = scale * square_length
    verts = np.array(
        [
            [0, 0, 0],
            [s, 0, 0],
            [s, s, 0],
            [0, s, 0],  # bottom face
            [0, 0, -s],
            [s, 0, -s],
            [s, s, -s],
            [0, s, -s],  # top face
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    return verts, edges


def _make_solid_cube_faces(
    scale: float, square_length: float
) -> tuple[np.ndarray, list[list[int]]]:
    """Generate cube vertices and face index lists for solid rendering.

    Returns:
        (vertices, faces) where faces is list of [i0, i1, i2, i3] quads.
    """
    s = scale * square_length
    verts = np.array(
        [
            [0, 0, 0],
            [s, 0, 0],
            [s, s, 0],
            [0, s, 0],
            [0, 0, -s],
            [s, 0, -s],
            [s, s, -s],
            [0, s, -s],
        ],
        dtype=np.float32,
    )
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]
    return verts, faces


class AROverlay:
    """Renders 3D AR objects on a frame using calibrated pose."""

    # Face colors for solid cube (BGR, semi-transparent)
    _FACE_COLORS = [
        (200, 100, 100),
        (100, 200, 100),
        (100, 100, 200),
        (200, 200, 100),
        (200, 100, 200),
        (100, 200, 200),
    ]

    def __init__(
        self,
        ar_object: str = "wireframe",
        obj_path: Optional[str] = None,
        scale: float = 1.0,
        square_length: float = 0.04,
        smooth_alpha: float = 0.6,
    ) -> None:
        self.ar_object = ar_object
        self.scale = scale
        self.square_length = square_length
        self._smoother = PoseSmoother(alpha=smooth_alpha)

        self._obj_verts: Optional[np.ndarray] = None
        self._obj_faces: Optional[list] = None
        self._obj_edges: Optional[list] = None
        self._edge_colors: Optional[list] = None

        if ar_object == "axes":
            self._obj_verts, self._obj_edges, self._edge_colors = (
                _make_axes_geometry(scale, square_length)
            )
        elif ar_object == "wireframe":
            self._obj_verts, self._obj_edges = _make_wireframe_cube_geometry(
                scale, square_length
            )
        elif ar_object == "solid":
            self._obj_verts, self._obj_faces = _make_solid_cube_faces(
                scale, square_length
            )
            _, self._obj_edges = _make_wireframe_cube_geometry(
                scale, square_length
            )
        elif ar_object == "obj":
            if obj_path is None:
                raise ValueError("--ar-obj-path required when --ar-object=obj")
            raw_verts, raw_faces = load_obj_mesh(obj_path)
            extent = raw_verts.max(axis=0) - raw_verts.min(axis=0)
            max_extent = max(float(extent.max()), 1e-6)
            normalized = (raw_verts - raw_verts.min(axis=0)) / max_extent
            self._obj_verts = normalized * scale * square_length
            # Flip Z so object sits above the board
            self._obj_verts[:, 2] = -self._obj_verts[:, 2]
            self._obj_faces = [list(f) for f in raw_faces]
            edge_set: set[tuple[int, int]] = set()
            for face in raw_faces:
                for i in range(len(face)):
                    a, b = face[i], face[(i + 1) % len(face)]
                    edge_set.add((min(a, b), max(a, b)))
            self._obj_edges = list(edge_set)

    def render(
        self,
        frame: np.ndarray,
        pose: PoseResult,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Project and draw the AR object onto the frame.

        Args:
            frame: BGR image to draw on (will be copied).
            pose: PoseResult from estimate_board_pose.
            camera_matrix: 3x3 intrinsic matrix.
            dist_coeffs: Distortion coefficients.

        Returns:
            Frame with AR overlay drawn.
        """
        if not pose.success or self._obj_verts is None:
            return frame

        rvec, tvec = self._smoother.update(pose.rvec, pose.tvec)

        pts_2d, _ = cv2.projectPoints(
            self._obj_verts.reshape(-1, 1, 3).astype(np.float64),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)

        vis = frame.copy()

        if self.ar_object == "axes":
            self._draw_axes(vis, pts_2d)
        elif self.ar_object == "wireframe":
            self._draw_wireframe(vis, pts_2d)
        elif self.ar_object == "solid":
            self._draw_solid(vis, pts_2d)
        elif self.ar_object == "obj":
            self._draw_mesh(vis, pts_2d)

        return vis

    def _draw_axes(self, vis: np.ndarray, pts: np.ndarray) -> None:
        """Draw coordinate axes with colored lines."""
        if self._obj_edges is None or self._edge_colors is None:
            return
        for (i, j), color in zip(self._obj_edges, self._edge_colors):
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, 3, cv2.LINE_AA)
        labels = ["X", "Y", "Z"]
        for k, label in enumerate(labels):
            cv2.putText(
                vis,
                label,
                tuple(pts[k + 1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self._edge_colors[k],
                2,
            )

    def _draw_wireframe(self, vis: np.ndarray, pts: np.ndarray) -> None:
        """Draw wireframe edges."""
        if self._obj_edges is None:
            return
        color = (0, 255, 0)
        for i, j in self._obj_edges:
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)

    def _draw_solid(self, vis: np.ndarray, pts: np.ndarray) -> None:
        """Draw filled faces with painter's algorithm and alpha blending."""
        if self._obj_faces is None:
            return

        face_depths = []
        for face_idx, face in enumerate(self._obj_faces):
            avg_z = np.mean([self._obj_verts[v][2] for v in face])
            face_depths.append((avg_z, face_idx))
        face_depths.sort(key=lambda x: x[0])  # far faces first

        alpha = 0.6
        overlay = vis.copy()
        for _, face_idx in face_depths:
            face = self._obj_faces[face_idx]
            poly = np.array([pts[v] for v in face], dtype=np.int32)
            color = self._FACE_COLORS[face_idx % len(self._FACE_COLORS)]
            cv2.fillPoly(overlay, [poly], color)

        cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)
        self._draw_wireframe(vis, pts)

    def _draw_mesh(self, vis: np.ndarray, pts: np.ndarray) -> None:
        """Draw OBJ mesh as wireframe."""
        if self._obj_edges is None:
            return
        color = (255, 200, 0)
        for i, j in self._obj_edges:
            if i < len(pts) and j < len(pts):
                cv2.line(
                    vis, tuple(pts[i]), tuple(pts[j]), color, 1, cv2.LINE_AA
                )

    def reset(self) -> None:
        """Reset pose smoother state."""
        self._smoother.reset()
