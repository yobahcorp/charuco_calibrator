"""Tests for bokeh.py — mask generation and bokeh effect application."""

from __future__ import annotations

import numpy as np
import pytest

from charuco_calibrator.bokeh import (
    BokehEffect,
    create_board_mask,
    create_tilt_shift_mask,
    _make_disk_kernel,
)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------


class TestDiskKernel:
    def test_shape_odd(self):
        kernel = _make_disk_kernel(15)
        assert kernel.shape == (15, 15)

    def test_shape_even_becomes_odd(self):
        kernel = _make_disk_kernel(14)
        assert kernel.shape[0] % 2 == 1

    def test_normalized(self):
        kernel = _make_disk_kernel(11)
        np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-6)

    def test_circular_shape(self):
        kernel = _make_disk_kernel(21)
        center = kernel.shape[0] // 2
        assert kernel[center, center] > 0
        assert kernel[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


class TestCreateBoardMask:
    def test_mask_shape(self):
        corners = np.array(
            [[100, 100], [200, 100], [200, 200], [100, 200]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mask = create_board_mask((480, 640), corners, feather=31)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32

    def test_mask_range(self):
        corners = np.array(
            [[100, 100], [200, 100], [200, 200], [100, 200]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mask = create_board_mask((480, 640), corners, feather=31)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_board_region_is_sharp(self):
        corners = np.array(
            [[200, 200], [400, 200], [400, 300], [200, 300]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mask = create_board_mask((480, 640), corners, feather=5)
        assert mask[250, 300] > 0.8

    def test_outside_region_is_blurred(self):
        corners = np.array(
            [[200, 200], [400, 200], [400, 300], [200, 300]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mask = create_board_mask((480, 640), corners, feather=5)
        assert mask[10, 10] < 0.2

    def test_too_few_corners(self):
        corners = np.array(
            [[100, 100], [200, 200]], dtype=np.float32
        ).reshape(-1, 1, 2)
        mask = create_board_mask((480, 640), corners, feather=31)
        assert mask.max() == 0.0


class TestCreateTiltShiftMask:
    def test_mask_shape(self):
        mask = create_tilt_shift_mask((480, 640), 0.5, feather=31)
        assert mask.shape == (480, 640)

    def test_center_is_sharp(self):
        mask = create_tilt_shift_mask(
            (480, 640), 0.5, band_height_frac=0.3, feather=5
        )
        assert mask[240, 320] > 0.8

    def test_edges_are_blurred(self):
        mask = create_tilt_shift_mask(
            (480, 640), 0.5, band_height_frac=0.2, feather=5
        )
        assert mask[10, 320] < 0.2
        assert mask[470, 320] < 0.2


# ---------------------------------------------------------------------------
# BokehEffect
# ---------------------------------------------------------------------------


class TestBokehEffect:
    def test_gaussian_mode(self):
        bokeh = BokehEffect(strength=15, feather=11, blur_mode="gaussian")
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        corners = np.array(
            [[200, 200], [400, 200], [400, 300], [200, 300]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        result = bokeh.apply(frame, corners)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_lens_mode(self):
        bokeh = BokehEffect(strength=15, feather=11, blur_mode="lens")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        corners = np.array(
            [[200, 200], [400, 200], [400, 300], [200, 300]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        result = bokeh.apply(frame, corners)
        assert result.shape == frame.shape

    def test_no_corners_returns_original(self):
        bokeh = BokehEffect()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = bokeh.apply(frame, None)
        np.testing.assert_array_equal(result, frame)

    def test_too_few_corners_returns_original(self):
        bokeh = BokehEffect()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        corners = np.array(
            [[100, 100], [200, 200]], dtype=np.float32
        ).reshape(-1, 1, 2)
        result = bokeh.apply(frame, corners)
        np.testing.assert_array_equal(result, frame)

    def test_toggle_focus_mode(self):
        bokeh = BokehEffect(focus_mode="board_focus")
        assert bokeh.toggle_focus_mode() == "tilt_shift"
        assert bokeh.toggle_focus_mode() == "board_focus"

    def test_tilt_shift_mode(self):
        bokeh = BokehEffect(
            strength=15, feather=11, focus_mode="tilt_shift"
        )
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        corners = np.array(
            [[200, 200], [400, 200], [400, 300], [200, 300]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        result = bokeh.apply(frame, corners, board_center_y_norm=0.5)
        assert result.shape == frame.shape

    def test_bokeh_modifies_frame(self):
        bokeh = BokehEffect(strength=25, feather=11, blur_mode="gaussian")
        # Use non-uniform frame so blur is visible
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:200] = 255
        corners = np.array(
            [[250, 250], [400, 250], [400, 350], [250, 350]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        result = bokeh.apply(frame, corners)
        assert not np.array_equal(result, frame)

    def test_strength_forced_odd(self):
        bokeh = BokehEffect(strength=24)
        assert bokeh.strength % 2 == 1
