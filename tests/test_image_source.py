"""Tests for image_source.py — source factory and CameraSource basics."""

from __future__ import annotations

import pytest

from charuco_calibrator.config import SourceConfig
from charuco_calibrator.image_source import CameraSource, ImageSource, create_source


class TestCreateSource:
    def test_creates_camera_source_by_default(self):
        cfg = SourceConfig(camera_id=0)
        source = create_source(cfg)
        assert isinstance(source, CameraSource)
        source.release()

    def test_creates_camera_source_for_video(self, tmp_path):
        # Non-existent video — won't open but shouldn't crash
        cfg = SourceConfig(video_path=str(tmp_path / "nonexistent.mp4"))
        source = create_source(cfg)
        assert isinstance(source, CameraSource)
        assert not source.is_open
        source.release()

    def test_ros_source_raises_without_rclpy(self):
        cfg = SourceConfig(ros_topic="/camera/image_raw")
        # This should raise ImportError if rclpy isn't installed
        try:
            source = create_source(cfg)
            source.release()
        except ImportError as e:
            assert "ROS 2" in str(e)


class TestCameraSource:
    def test_read_on_nonexistent(self, tmp_path):
        source = CameraSource(video_path=str(tmp_path / "nope.avi"))
        ok, frame = source.read()
        assert not ok
        assert frame is None
        source.release()
