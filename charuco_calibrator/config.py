"""Configuration dataclasses and YAML loading for ChArUco calibrator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import yaml


# ---------------------------------------------------------------------------
# ArUco dictionary resolution
# ---------------------------------------------------------------------------

_ARUCO_DICT_MAP: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def resolve_aruco_dict(name: str) -> int:
    """Resolve an ArUco dictionary name string to its OpenCV integer constant."""
    key = name.upper().replace("-", "_")
    if not key.startswith("DICT_"):
        key = f"DICT_{key}"
    if key not in _ARUCO_DICT_MAP:
        valid = ", ".join(sorted(_ARUCO_DICT_MAP))
        raise ValueError(f"Unknown ArUco dictionary '{name}'. Valid: {valid}")
    return _ARUCO_DICT_MAP[key]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BoardConfig:
    """ChArUco board geometry."""

    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.04  # metres
    marker_length: float = 0.03  # metres
    aruco_dict: str = "DICT_4X4_50"


@dataclass
class ThresholdConfig:
    """Frame acceptance thresholds."""

    min_corners: int = 6
    min_blur_var: float = 50.0
    min_score: float = 0.3
    capture_cooldown_ms: int = 1500


@dataclass
class CoverageConfig:
    """Coverage tracking parameters."""

    grid_cols: int = 4
    grid_rows: int = 4
    scale_bins: int = 5


@dataclass
class SourceConfig:
    """Image source configuration."""

    camera_id: int = 0
    video_path: Optional[str] = None
    image_folder: Optional[str] = None
    ros_topic: Optional[str] = None
    width: int = 0  # 0 = use default
    height: int = 0


@dataclass
class OutputConfig:
    """Calibration output configuration."""

    output_dir: str = "calibration_output"
    camera_name: str = "camera"
    save_observations: bool = True


@dataclass
class ARConfig:
    """AR overlay configuration."""

    enabled: bool = False
    ar_object: str = "wireframe"  # axes, wireframe, solid, obj
    obj_path: Optional[str] = None
    scale: float = 1.0
    smooth_alpha: float = 0.6  # EMA pose smoothing (0=max smooth, 1=full follow)


@dataclass
class BokehConfig:
    """Synthetic bokeh configuration."""

    enabled: bool = False
    strength: int = 25  # max blur kernel size (must be odd)
    feather: int = 31  # mask feathering kernel size (must be odd)
    blur_mode: str = "gaussian"  # gaussian or lens
    focus_mode: str = "board_focus"  # board_focus or tilt_shift


@dataclass
class AppConfig:
    """Top-level application configuration."""

    board: BoardConfig = field(default_factory=BoardConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ar: ARConfig = field(default_factory=ARConfig)
    bokeh: BokehConfig = field(default_factory=BokehConfig)
    auto_capture: bool = False
    show_heatmap: bool = False
    recalibrate_every: int = 5
    auto_prune: bool = True
    prune_threshold: float = 2.0
    gpu: bool = False


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _merge_dict_into_dataclass(dc: Any, data: dict) -> None:
    """Recursively merge a dict into a dataclass instance."""
    for key, value in data.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(yaml_path: str | Path) -> AppConfig:
    """Load configuration from a YAML file, merging on top of defaults."""
    cfg = AppConfig()
    path = Path(yaml_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _merge_dict_into_dataclass(cfg, data)
    return cfg


def apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply CLI argument overrides on top of a loaded config."""
    if getattr(args, "camera", None) is not None:
        cfg.source.camera_id = args.camera
        cfg.source.video_path = None
        cfg.source.ros_topic = None
    if getattr(args, "video", None) is not None:
        cfg.source.video_path = args.video
        cfg.source.image_folder = None
        cfg.source.ros_topic = None
    if getattr(args, "image_folder", None) is not None:
        cfg.source.image_folder = args.image_folder
        cfg.source.video_path = None
        cfg.source.ros_topic = None
    if getattr(args, "ros_topic", None) is not None:
        cfg.source.ros_topic = args.ros_topic
    if getattr(args, "output_dir", None) is not None:
        cfg.output.output_dir = args.output_dir
    if getattr(args, "camera_name", None) is not None:
        cfg.output.camera_name = args.camera_name

    # AR overrides
    if getattr(args, "ar", False):
        cfg.ar.enabled = True
    if getattr(args, "ar_object", None) is not None:
        cfg.ar.ar_object = args.ar_object
    if getattr(args, "ar_obj_path", None) is not None:
        cfg.ar.obj_path = args.ar_obj_path
    if getattr(args, "ar_scale", None) is not None:
        cfg.ar.scale = args.ar_scale

    # Bokeh overrides
    if getattr(args, "bokeh", False):
        cfg.bokeh.enabled = True
    if getattr(args, "bokeh_strength", None) is not None:
        cfg.bokeh.strength = args.bokeh_strength
    if getattr(args, "bokeh_feather", None) is not None:
        cfg.bokeh.feather = args.bokeh_feather
    if getattr(args, "bokeh_mode", None) is not None:
        cfg.bokeh.blur_mode = args.bokeh_mode

    return cfg


def config_to_dict(cfg: AppConfig) -> dict:
    """Serialize an AppConfig to a plain dict."""
    return asdict(cfg)
