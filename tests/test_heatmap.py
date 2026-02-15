"""Tests for heatmap.py — CornerHeatmap accumulation and rendering."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from charuco_calibrator.heatmap import CornerHeatmap


class TestCornerHeatmap:
    def test_initial_state(self):
        hm = CornerHeatmap(480, 640)
        assert hm.accumulator.shape == (480, 640)
        assert hm.accumulator.max() == 0

    def test_add_corners(self):
        hm = CornerHeatmap(480, 640, sigma=10.0)
        corners = np.array([[320, 240]], dtype=np.float32).reshape(-1, 1, 2)
        hm.add_corners(corners)
        assert hm.accumulator.max() > 0
        # Peak should be near the added corner
        assert hm.accumulator[240, 320] > 0

    def test_add_multiple_corners(self):
        hm = CornerHeatmap(480, 640, sigma=10.0)
        corners = np.array([[100, 100], [500, 400]], dtype=np.float32).reshape(-1, 1, 2)
        hm.add_corners(corners)
        assert hm.accumulator[100, 100] > 0
        assert hm.accumulator[400, 500] > 0

    def test_render_shape(self):
        hm = CornerHeatmap(480, 640)
        img = hm.render()
        assert img.shape == (480, 640, 3)  # BGR
        assert img.dtype == np.uint8

    def test_blend_onto(self):
        hm = CornerHeatmap(480, 640, sigma=10.0)
        corners = np.array([[320, 240]], dtype=np.float32).reshape(-1, 1, 2)
        hm.add_corners(corners)

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        blended = hm.blend_onto(frame, alpha=0.5)
        assert blended.shape == frame.shape
        # Blended should differ from original at the corner location
        assert not np.array_equal(blended[240, 320], frame[240, 320])

    def test_blend_empty_heatmap(self):
        hm = CornerHeatmap(480, 640)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        blended = hm.blend_onto(frame)
        assert np.array_equal(blended, frame)

    def test_reset(self):
        hm = CornerHeatmap(480, 640, sigma=10.0)
        corners = np.array([[320, 240]], dtype=np.float32).reshape(-1, 1, 2)
        hm.add_corners(corners)
        assert hm.accumulator.max() > 0
        hm.reset()
        assert hm.accumulator.max() == 0

    def test_edge_corners(self):
        """Corners near image edges should not cause errors."""
        hm = CornerHeatmap(480, 640, sigma=15.0)
        corners = np.array(
            [[0, 0], [639, 0], [0, 479], [639, 479]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        hm.add_corners(corners)  # Should not raise
        assert hm.accumulator.max() > 0
