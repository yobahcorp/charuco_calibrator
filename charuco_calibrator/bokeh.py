"""Synthetic bokeh / depth-of-field effect using board pose."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _make_disk_kernel(size: int) -> np.ndarray:
    """Create a circular disk kernel for lens-style bokeh blur.

    Args:
        size: Kernel size (will be forced to odd).

    Returns:
        Normalized float32 kernel.
    """
    size = size | 1  # ensure odd
    radius = size // 2
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y) <= radius * radius
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return kernel


def create_board_mask(
    frame_shape: tuple[int, ...],
    charuco_corners: np.ndarray,
    feather: int = 31,
) -> np.ndarray:
    """Create a soft binary mask where the board region is 1.0 (sharp).

    Args:
        frame_shape: (H, W) or (H, W, C) shape.
        charuco_corners: (N,1,2) or (N,2) detected corner positions.
        feather: Gaussian kernel size for softening mask edges (must be odd).

    Returns:
        float32 mask of shape (H, W) in range [0.0, 1.0].
    """
    h, w = frame_shape[:2]
    pts = charuco_corners.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        return np.zeros((h, w), dtype=np.float32)

    hull = cv2.convexHull(pts).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    feather = feather | 1  # ensure odd
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (feather, feather), 0)
    np.clip(mask_f, 0.0, 1.0, out=mask_f)
    return mask_f


def create_tilt_shift_mask(
    frame_shape: tuple[int, ...],
    band_center_y: float,
    band_height_frac: float = 0.3,
    feather: int = 31,
) -> np.ndarray:
    """Create a horizontal band mask for tilt-shift effect.

    Args:
        frame_shape: (H, W) or (H, W, C).
        band_center_y: Normalized Y center of the sharp band [0..1].
        band_height_frac: Fraction of frame height for the sharp band.
        feather: Gaussian kernel size for softening edges.

    Returns:
        float32 mask of shape (H, W) in range [0.0, 1.0].
    """
    h, w = frame_shape[:2]
    center_px = int(band_center_y * h)
    half_band = int(band_height_frac * h / 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    y_top = max(center_px - half_band, 0)
    y_bot = min(center_px + half_band, h)
    mask[y_top:y_bot, :] = 255

    feather = feather | 1
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (feather, feather), 0)
    return mask_f


class BokehEffect:
    """Applies synthetic depth-of-field to frames using board detection."""

    def __init__(
        self,
        strength: int = 25,
        feather: int = 31,
        blur_mode: str = "gaussian",
        focus_mode: str = "board_focus",
    ) -> None:
        self.strength = strength | 1  # ensure odd
        self.feather = feather | 1
        self.blur_mode = blur_mode
        self.focus_mode = focus_mode
        self._disk_kernel: Optional[np.ndarray] = None

        if blur_mode == "lens":
            self._disk_kernel = _make_disk_kernel(self.strength)

    def apply(
        self,
        frame: np.ndarray,
        charuco_corners: Optional[np.ndarray],
        board_center_y_norm: Optional[float] = None,
    ) -> np.ndarray:
        """Apply the bokeh effect to a frame.

        Args:
            frame: BGR input image.
            charuco_corners: (N,1,2) detected corners (None if no detection).
            board_center_y_norm: Normalized Y center of board (for tilt-shift).

        Returns:
            Frame with bokeh effect applied.
        """
        if charuco_corners is None or len(charuco_corners.reshape(-1, 2)) < 3:
            return frame

        # Create mask: board region is sharp (1.0), rest is blurred (0.0)
        if self.focus_mode == "tilt_shift" and board_center_y_norm is not None:
            mask = create_tilt_shift_mask(
                frame.shape, board_center_y_norm, feather=self.feather
            )
        else:
            mask = create_board_mask(
                frame.shape, charuco_corners, feather=self.feather
            )

        # Generate blurred version
        if self.blur_mode == "lens" and self._disk_kernel is not None:
            blurred = cv2.filter2D(frame, -1, self._disk_kernel)
        else:
            blurred = cv2.GaussianBlur(
                frame, (self.strength, self.strength), 0
            )

        # Blend: output = sharp * mask + blurred * (1 - mask)
        mask_3ch = mask[:, :, np.newaxis]
        output = frame.astype(np.float32) * mask_3ch + blurred.astype(
            np.float32
        ) * (1.0 - mask_3ch)
        return output.astype(np.uint8)

    def toggle_focus_mode(self) -> str:
        """Toggle between board_focus and tilt_shift. Returns new mode."""
        if self.focus_mode == "board_focus":
            self.focus_mode = "tilt_shift"
        else:
            self.focus_mode = "board_focus"
        return self.focus_mode
