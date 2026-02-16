# ChArUco Calibrator

Interactive camera calibration tool using ChArUco boards. Provides real-time detection feedback, frame scoring, coverage tracking, corner heatmaps, and auto-capture — so you can get a high-quality calibration with minimal effort.

Works with any USB webcam or video file via OpenCV. Optionally subscribes to a ROS 2 image topic for robotic workflows.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Live ChArUco detection** with corner and marker overlay
- **Frame scoring** based on corner count, hull spread, new coverage, and sharpness
- **Coverage tracking** — grid, quadrant, scale, and view diversity
- **Quality meter** — single 0–100% bar combining all coverage dimensions
- **Corner heatmap** — Gaussian-splat overlay to visualize observation density
- **Auto-capture** — hands-free data collection with configurable cooldown
- **Auto-recalibration** — RMS updates as you collect more frames
- **YAML export** — ROS `camera_info_manager` compatible format
- **ROS 2 integration** — standalone `--ros-topic` flag or dedicated ROS 2 package

## Quick Start

### Install

```bash
git clone https://github.com/yobahcorp/charuco_calibrator.git
cd charuco_calibrator
pip install -e .
```

For Ubuntu system dependencies, see [INSTALL_UBUNTU.md](INSTALL_UBUNTU.md).

### Run

```bash
# Webcam
charuco-calibrate --camera 0

# Video file
charuco-calibrate --video recording.mp4

# ROS 2 topic (requires rclpy + cv_bridge)
charuco-calibrate --ros-topic /camera/image_raw
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture current frame |
| `A` | Toggle auto-capture |
| `C` | Run calibration (needs >= 4 frames) |
| `S` | Save calibration YAML + observations |
| `R` | Reset all data |
| `H` | Toggle heatmap overlay |
| `Q` / `ESC` | Quit |

## Configuration

All parameters can be set via a YAML config file and/or CLI flags. CLI flags override the config file.

```bash
charuco-calibrate --config my_config.yaml --camera 0 --output-dir results/
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML config file |
| `--camera <id>` | Camera device ID (integer) |
| `--video <path>` | Path to a video file |
| `--ros-topic <topic>` | ROS 2 image topic name |
| `--output-dir <dir>` | Output directory for calibration files |
| `--camera-name <name>` | Camera name written into the YAML |

### Config File

See [`config/default_config.yaml`](config/default_config.yaml) for the full reference. Key sections:

| Section | Parameters |
|---------|------------|
| **board** | `squares_x`, `squares_y`, `square_length`, `marker_length`, `aruco_dict` |
| **thresholds** | `min_corners`, `min_blur_var`, `min_score`, `capture_cooldown_ms` |
| **coverage** | `grid_cols`, `grid_rows`, `scale_bins` |
| **source** | `camera_id`, `video_path`, `ros_topic`, `width`, `height` |
| **output** | `output_dir`, `camera_name`, `save_observations` |

## Board Setup

1. Print a ChArUco board matching your config (`squares_x`, `squares_y`, `aruco_dict`). A reference image is included at [`charuco_board.png`](charuco_board.png).
2. Mount it on something rigid and flat (foam board, clipboard).
3. Measure the printed square and marker sizes with a ruler and set `square_length` and `marker_length` to the actual values in metres.

**Important:** `squares_x` is the number of squares **horizontally** (columns) and `squares_y` is **vertically** (rows) as seen by the camera. `marker_length` must be less than `square_length`.

### Supported ArUco Dictionaries

`DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`,
`DICT_5X5_50`, `DICT_5X5_100`, `DICT_5X5_250`, `DICT_5X5_1000`,
`DICT_6X6_50`, `DICT_6X6_100`, `DICT_6X6_250`, `DICT_6X6_1000`,
`DICT_7X7_50`, `DICT_7X7_100`, `DICT_7X7_250`, `DICT_7X7_1000`

Use the smallest dictionary that covers your marker count. For a 9x7 board (31 markers), `DICT_4X4_50` works well.

## Tips for a Good Calibration

1. **Fill the coverage grid** — move the board to cover all cells
2. **Vary distance** — show the board close-up and far away
3. **Vary angle** — tilt the board in different directions
4. **Stay sharp** — hold still; blurry frames are rejected automatically
5. **Watch the quality meter** — aim for 80%+ before calibrating
6. **Collect 20–40 frames** — more diverse frames reduce RMS error
7. **Check RMS** — a good calibration typically achieves RMS < 1.0 px

## ROS 2 Integration

There are two ways to use the calibrator with ROS 2:

### Option A: Standalone CLI with `--ros-topic`

```bash
charuco-calibrate --ros-topic /camera/image_raw
```

Lazily imports `rclpy` and `cv_bridge` — no ROS dependency at install time.

### Option B: Dedicated ROS 2 package (recommended)

```bash
colcon build --packages-select charuco_calibrator_ros
source install/setup.bash
ros2 launch charuco_calibrator_ros calibrator.launch.py
```

All board, threshold, and output settings are exposed as ROS parameters:

```bash
ros2 run charuco_calibrator_ros charuco_calibrator_node --ros-args \
    -p image_topic:=/camera/image_raw \
    -p squares_x:=9 \
    -p squares_y:=7 \
    -p width:=1280 \
    -p height:=720 \
    -p auto_capture:=true
```

See [`charuco_calibrator_ros/README.md`](charuco_calibrator_ros/README.md) for full setup, parameter list, and troubleshooting.

## Output

After calibrating (`C`) and saving (`S`), two files are written to `output_dir` (default `calibration_output/`):

| File | Contents |
|------|----------|
| `calibration.yaml` | Camera intrinsics in ROS `camera_info` compatible format |
| `observations.npz` | Raw corner observations for offline re-calibration |

## Project Structure

```
charuco_calibrator/
  config.py          # Dataclasses + YAML loading
  image_source.py    # Camera, video, and ROS 2 image sources
  detector.py        # ChArUco detection (OpenCV 4.8+ OO API with legacy fallback)
  scoring.py         # Frame scoring and coverage tracking
  heatmap.py         # Gaussian-splat corner heatmap
  calibration.py     # cv2.calibrateCamera wrapper + YAML export
  ui.py              # Overlay rendering + keyboard dispatch
  main.py            # run(cfg) loop + CLI entry point
config/
  default_config.yaml
tests/
charuco_calibrator_ros/   # Optional ROS 2 package
```

## License

MIT
