"""Frame scoring, blur estimation, coverage tracking, and quality meter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import CoverageConfig
from .detector import DetectionResult


# ---------------------------------------------------------------------------
# Blur estimation
# ---------------------------------------------------------------------------


def compute_blur_variance(gray: np.ndarray) -> float:
    """Compute the variance of the Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Convex hull spread
# ---------------------------------------------------------------------------


def compute_convex_hull_area(corners: np.ndarray, image_area: float) -> float:
    """Fraction of image area covered by the convex hull of detected corners.

    Args:
        corners: (N, 1, 2) or (N, 2) array of corner pixel coordinates.
        image_area: total pixel area of the image.

    Returns:
        Ratio in [0, 1].
    """
    pts = corners.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)
    return min(area / image_area, 1.0)


# ---------------------------------------------------------------------------
# Coverage state
# ---------------------------------------------------------------------------


@dataclass
class CoverageState:
    """Tracks spatial, scale, and view diversity coverage."""

    grid_cols: int = 4
    grid_rows: int = 4
    scale_bins: int = 5

    # Internal state — set by reset()
    grid: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    quadrants: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    scale_histogram: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    centroid_history: list = field(default_factory=list, repr=False)
    total_frames: int = 0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        self.quadrants = np.zeros(4, dtype=np.int32)
        self.scale_histogram = np.zeros(self.scale_bins, dtype=np.int32)
        self.centroid_history = []
        self.total_frames = 0

    # -- Update methods --

    def update(
        self,
        corners: np.ndarray,
        image_h: int,
        image_w: int,
    ) -> None:
        """Update all coverage metrics from a set of accepted corners."""
        pts = corners.reshape(-1, 2)
        self.total_frames += 1

        self._update_grid(pts, image_h, image_w)
        self._update_quadrants(pts, image_h, image_w)
        self._update_scale(pts, image_h, image_w)
        self._update_centroid(pts, image_h, image_w)

    def _update_grid(self, pts: np.ndarray, h: int, w: int) -> None:
        cell_w = w / self.grid_cols
        cell_h = h / self.grid_rows
        for x, y in pts:
            col = min(int(x / cell_w), self.grid_cols - 1)
            row = min(int(y / cell_h), self.grid_rows - 1)
            self.grid[row, col] += 1

    def _update_quadrants(self, pts: np.ndarray, h: int, w: int) -> None:
        cx, cy = w / 2.0, h / 2.0
        for x, y in pts:
            qi = (1 if x >= cx else 0) + (2 if y >= cy else 0)
            self.quadrants[qi] += 1

    def _update_scale(self, pts: np.ndarray, h: int, w: int) -> None:
        if len(pts) < 3:
            return
        hull = cv2.convexHull(pts.astype(np.float32))
        area_ratio = cv2.contourArea(hull) / (h * w)
        bin_idx = min(int(area_ratio * self.scale_bins / 0.5), self.scale_bins - 1)
        bin_idx = max(bin_idx, 0)
        self.scale_histogram[bin_idx] += 1

    def _update_centroid(self, pts: np.ndarray, h: int, w: int) -> None:
        cx = float(pts[:, 0].mean()) / w
        cy = float(pts[:, 1].mean()) / h
        self.centroid_history.append((cx, cy))

    # -- Metric computations --

    @property
    def grid_coverage(self) -> float:
        """Fraction of grid cells that have been hit."""
        if self.grid.size == 0:
            return 0.0
        return float(np.count_nonzero(self.grid)) / self.grid.size

    @property
    def quadrant_coverage(self) -> float:
        """Fraction of quadrants that have been hit."""
        return float(np.count_nonzero(self.quadrants)) / 4.0

    @property
    def scale_coverage(self) -> float:
        """Fraction of scale bins that have been hit."""
        if self.scale_bins == 0:
            return 0.0
        return float(np.count_nonzero(self.scale_histogram)) / self.scale_bins

    @property
    def view_diversity(self) -> float:
        """Diversity of centroid positions (0..1 based on spread)."""
        if len(self.centroid_history) < 2:
            return 0.0
        centroids = np.array(self.centroid_history)
        spread = centroids.std(axis=0).mean()
        # Normalize: a spread of ~0.25 in normalized coords is "excellent"
        return min(spread / 0.25, 1.0)

    @property
    def quality_meter(self) -> float:
        """Weighted combination of all coverage metrics (0..1)."""
        return (
            self.grid_coverage * 0.35
            + self.quadrant_coverage * 0.20
            + self.scale_coverage * 0.20
            + self.view_diversity * 0.25
        )


# ---------------------------------------------------------------------------
# Per-frame scoring
# ---------------------------------------------------------------------------


def compute_new_grid_coverage(
    corners: np.ndarray,
    image_h: int,
    image_w: int,
    current_grid: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> float:
    """Fraction of grid cells that would be newly covered by these corners."""
    pts = corners.reshape(-1, 2)
    cell_w = image_w / grid_cols
    cell_h = image_h / grid_rows
    new_cells = 0
    total_empty = max(int((current_grid == 0).sum()), 1)

    visited = set()
    for x, y in pts:
        col = min(int(x / cell_w), grid_cols - 1)
        row = min(int(y / cell_h), grid_rows - 1)
        key = (row, col)
        if key not in visited and current_grid[row, col] == 0:
            new_cells += 1
            visited.add(key)

    return min(new_cells / total_empty, 1.0)


@dataclass
class FrameScore:
    """Breakdown of a single frame's quality score."""

    corner_ratio: float = 0.0
    hull_spread: float = 0.0
    new_coverage: float = 0.0
    blur_norm: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


def score_frame(
    result: DetectionResult,
    frame: np.ndarray,
    coverage: CoverageState,
    total_corners: int,
    *,
    min_corners: int = 6,
    min_blur_var: float = 50.0,
    min_score: float = 0.3,
) -> FrameScore:
    """Score a detection result for acceptance.

    Returns a FrameScore with the total and rejection info.
    """
    score = FrameScore()

    if not result.valid or result.charuco_corners is None:
        score.rejected = True
        score.reject_reason = "no detection"
        return score

    n = result.num_corners
    if n < min_corners:
        score.rejected = True
        score.reject_reason = f"too few corners ({n} < {min_corners})"
        return score

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blur_var = compute_blur_variance(gray)
    if blur_var < min_blur_var:
        score.rejected = True
        score.reject_reason = f"too blurry (var={blur_var:.1f} < {min_blur_var})"
        return score

    h, w = frame.shape[:2]
    image_area = float(h * w)

    score.corner_ratio = min(n / total_corners, 1.0)
    score.hull_spread = compute_convex_hull_area(result.charuco_corners, image_area)
    score.new_coverage = compute_new_grid_coverage(
        result.charuco_corners,
        h,
        w,
        coverage.grid,
        coverage.grid_rows,
        coverage.grid_cols,
    )
    # Normalize blur: map [min_blur_var, 1000] → [0, 1]
    blur_range = 1000.0
    score.blur_norm = min((blur_var - min_blur_var) / blur_range, 1.0)

    score.total = (
        score.corner_ratio * 0.3
        + score.hull_spread * 0.2
        + score.new_coverage * 0.3
        + score.blur_norm * 0.2
    )

    if score.total < min_score:
        score.rejected = True
        score.reject_reason = f"score too low ({score.total:.2f} < {min_score})"

    return score
