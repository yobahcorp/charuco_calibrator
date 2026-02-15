"""Tests for config.py — dataclasses, YAML loading, CLI overrides."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pytest
import yaml

from charuco_calibrator.config import (
    AppConfig,
    BoardConfig,
    ThresholdConfig,
    apply_cli_overrides,
    config_to_dict,
    load_config,
    resolve_aruco_dict,
)


class TestResolveArucoDict:
    def test_valid_name(self):
        import cv2

        assert resolve_aruco_dict("DICT_4X4_50") == cv2.aruco.DICT_4X4_50

    def test_without_prefix(self):
        import cv2

        assert resolve_aruco_dict("4X4_50") == cv2.aruco.DICT_4X4_50

    def test_lowercase(self):
        import cv2

        assert resolve_aruco_dict("dict_5x5_100") == cv2.aruco.DICT_5X5_100

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
            resolve_aruco_dict("DICT_99X99_999")


class TestAppConfigDefaults:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.board.squares_x == 5
        assert cfg.board.squares_y == 7
        assert cfg.thresholds.min_corners == 6
        assert cfg.coverage.grid_cols == 4
        assert cfg.source.camera_id == 0
        assert cfg.output.output_dir == "calibration_output"


class TestLoadConfig:
    def test_load_nonexistent_returns_defaults(self, tmp_path: Path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.board.squares_x == 5

    def test_load_partial_yaml(self, tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            yaml.dump({"board": {"squares_x": 8, "aruco_dict": "DICT_6X6_50"}})
        )
        cfg = load_config(yaml_path)
        assert cfg.board.squares_x == 8
        assert cfg.board.aruco_dict == "DICT_6X6_50"
        # Unchanged defaults
        assert cfg.board.squares_y == 7

    def test_load_full_yaml(self, tmp_path: Path):
        data = {
            "board": {"squares_x": 10, "squares_y": 10},
            "thresholds": {"min_corners": 10},
            "auto_capture": True,
        }
        yaml_path = tmp_path / "full.yaml"
        yaml_path.write_text(yaml.dump(data))
        cfg = load_config(yaml_path)
        assert cfg.board.squares_x == 10
        assert cfg.thresholds.min_corners == 10
        assert cfg.auto_capture is True


class TestApplyCliOverrides:
    def test_camera_override(self):
        cfg = AppConfig()
        args = argparse.Namespace(camera=2, video=None, ros_topic=None, output_dir=None, camera_name=None)
        cfg = apply_cli_overrides(cfg, args)
        assert cfg.source.camera_id == 2

    def test_video_override(self):
        cfg = AppConfig()
        args = argparse.Namespace(camera=None, video="/tmp/test.mp4", ros_topic=None, output_dir=None, camera_name=None)
        cfg = apply_cli_overrides(cfg, args)
        assert cfg.source.video_path == "/tmp/test.mp4"

    def test_ros_topic_override(self):
        cfg = AppConfig()
        args = argparse.Namespace(camera=None, video=None, ros_topic="/camera/image_raw", output_dir=None, camera_name=None)
        cfg = apply_cli_overrides(cfg, args)
        assert cfg.source.ros_topic == "/camera/image_raw"


class TestConfigToDict:
    def test_round_trip(self):
        cfg = AppConfig()
        d = config_to_dict(cfg)
        assert d["board"]["squares_x"] == 5
        assert isinstance(d, dict)
