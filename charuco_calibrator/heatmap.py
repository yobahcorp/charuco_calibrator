"""Corner heatmap accumulator with Gaussian splat rendering."""

from __future__ import annotations

import numpy as np
import cv2


class CornerHeatmap:
    """Accumulates detected corner positions and renders a heatmap overlay."""

    def __init__(self, height: int, width: int, sigma: float = 15.0) -> None:
        self.height = height
        self.width = width
        self.sigma = sigma
        self._accumulator = np.zeros((height, width), dtype=np.float64)
        self._kernel = self._make_kernel(sigma)

    @staticmethod
    def _make_kernel(sigma: float) -> np.ndarray:
        """Create a precomputed 2D Gaussian kernel."""
        radius = int(3 * sigma)
        size = 2 * radius + 1
        x = np.arange(size) - radius
        g = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = np.outer(g, g)
        kernel /= kernel.max()
        return kernel

    def add_corners(self, corners: np.ndarray) -> None:
        """Add Gaussian splats at each corner location.

        Args:
            corners: (N, 1, 2) or (N, 2) array of corner pixel coordinates.
        """
        pts = corners.reshape(-1, 2)
        radius = self._kernel.shape[0] // 2

        for x, y in pts:
            cx, cy = int(round(x)), int(round(y))

            # Compute ROI bounds, clipped to image
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius + 1, self.height)
            x0 = max(cx - radius, 0)
            x1 = min(cx + radius + 1, self.width)

            # Corresponding kernel slice
            ky0 = y0 - (cy - radius)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (cx - radius)
            kx1 = kx0 + (x1 - x0)

            if y1 > y0 and x1 > x0:
                self._accumulator[y0:y1, x0:x1] += self._kernel[ky0:ky1, kx0:kx1]

    def render(self) -> np.ndarray:
        """Render the heatmap as an 8-bit BGR colormap image."""
        if self._accumulator.max() > 0:
            normed = (self._accumulator / self._accumulator.max() * 255).astype(np.uint8)
        else:
            normed = np.zeros((self.height, self.width), dtype=np.uint8)
        return cv2.applyColorMap(normed, cv2.COLORMAP_JET)

    def blend_onto(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Alpha-blend the heatmap onto a BGR frame.

        Returns a new image (does not modify the input).
        """
        if self._accumulator.max() == 0:
            return frame.copy()

        heatmap_bgr = self.render()
        # Only blend where there is actual heat
        mask = self._accumulator > 0
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(
            frame[mask].reshape(-1, 1, 3),
            1.0 - alpha,
            heatmap_bgr[mask].reshape(-1, 1, 3),
            alpha,
            0,
        ).reshape(-1, 3)
        return blended

    def reset(self) -> None:
        """Clear the accumulated heatmap."""
        self._accumulator[:] = 0

    @property
    def accumulator(self) -> np.ndarray:
        """Raw accumulator array (read-only view)."""
        return self._accumulator
