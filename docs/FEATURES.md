# ChArUco Calibrator — Feature Documentation

## Overview

ChArUco Calibrator is a standalone Python camera calibration tool that uses ChArUco boards to compute intrinsic camera parameters. It provides real-time visual feedback, coverage tracking, corner heatmaps, and auto-capture to guide you toward a high-quality calibration with minimal effort.

Works out of the box with any USB webcam or video file via OpenCV. Optionally subscribes to a ROS 2 image topic for integration with robotic systems.

---

## Quick Start

```bash
# Webcam
charuco-calibrate --camera 0

# Video file
charuco-calibrate --video recording.mp4

# ROS 2 topic (requires rclpy + cv_bridge)
charuco-calibrate --ros-topic /camera/image_raw
```

---

## Features

### 1. Live ChArUco Detection

Detected ArUco markers and interpolated ChArUco corners are drawn on every frame in real time. The detector uses OpenCV 4.8+ object-oriented API (`CharucoDetector.detectBoard`) and falls back automatically to the legacy `detectMarkers` + `interpolateCornersCharuco` pipeline on older OpenCV builds.

- Green diamonds mark each detected ChArUco corner
- Colored rectangles outline each detected ArUco marker
- Detection runs every frame with no manual trigger required

### 2. Frame Scoring & Accept/Reject

Every frame is scored on four weighted criteria:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Corner ratio | 0.30 | Fraction of possible ChArUco corners detected |
| Hull spread | 0.20 | Convex hull area of detected corners relative to image area |
| New coverage | 0.30 | How many previously-empty grid cells this frame would fill |
| Blur norm | 0.20 | Image sharpness (Laplacian variance, higher is better) |

A frame is **rejected** if any of these conditions hold:
- Fewer corners than `min_corners` (default: 6)
- Laplacian variance below `min_blur_var` (default: 50)
- Total score below `min_score` (default: 0.3)

The score breakdown is shown live in the status panel. Rejected frames display the reason in red.

### 3. Coverage Tracking

Coverage is tracked across four dimensions to ensure calibration data is diverse:

**Grid coverage** — The image is divided into a configurable grid (default 4x4). Each cell turns green when corners have been observed in that region. Coverage percentage is shown next to the grid.

**Quadrant coverage** — Tracks whether corners have been seen in all four image quadrants (top-left, top-right, bottom-left, bottom-right).

**Scale coverage** — Bins the apparent board size (convex hull area ratio) into 5 buckets, encouraging the user to show the board at different distances.

**View diversity** — Measures the spread of board centroid positions across accepted frames, encouraging movement of the board around the field of view.

### 4. Quality Meter

A single 0–100% bar that combines all coverage dimensions:

| Metric | Weight |
|--------|--------|
| Grid coverage | 0.35 |
| Quadrant coverage | 0.20 |
| Scale coverage | 0.20 |
| View diversity | 0.25 |

The bar color shifts from orange (low) to yellow (medium) to green (high). Aim for 80%+ before calibrating.

### 5. Corner Heatmap

A Gaussian-splat heatmap accumulates the pixel positions of every accepted corner. Toggle it with **H** to see which areas of the image have strong corner coverage and which need more observations.

- Each corner adds a Gaussian kernel (σ = 15 px) to the accumulator
- Rendered as a JET colormap, alpha-blended onto the live image
- Helps identify dead zones where you should position the board

### 6. Auto-Capture

Press **A** to toggle auto-capture mode. When enabled, frames are accepted automatically whenever:
- The frame score meets the `min_score` threshold
- The cooldown timer has elapsed (default: 1500 ms between captures)

This allows hands-free data collection — just move the board slowly and the system captures good frames for you.

### 7. Calibration

Press **C** to run `cv2.calibrateCamera` on all accumulated observations. Results include:
- **Camera matrix** (3x3 intrinsic parameters: fx, fy, cx, cy)
- **Distortion coefficients** (radial + tangential)
- **RMS reprojection error** (displayed live in the status panel after calibration)
- **Per-view errors** (per-frame reprojection error breakdown)

The system also auto-recalibrates every N accepted frames (default: 5) so the RMS updates as you collect more data.

Minimum 4 frames are required to calibrate. 20–40 diverse frames are recommended for production use.

### 8. Save & Export

Press **S** to save calibration results:

**`calibration.yaml`** — ROS `camera_info_manager` compatible format:
```yaml
image_width: 640
image_height: 480
camera_name: camera
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
projection_matrix:
  rows: 3
  cols: 4
  data: [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
rms_reprojection_error: 0.45
```

**`observations.npz`** — Raw object points, image points, and corner IDs for every accepted frame. Useful for offline re-calibration or analysis.

Output directory defaults to `calibration_output/` and is configurable via `--output-dir`.

### 9. ROS 2 Integration (Optional)

There are two ways to use the calibrator with ROS 2:

#### A) Standalone CLI with `--ros-topic`

Pass `--ros-topic /camera/image_raw` to subscribe to a ROS 2 image topic instead of using a local camera. The adapter:
- Lazily imports `rclpy` and `cv_bridge` — no ROS dependency at install time
- Runs a background spin thread for non-blocking frame delivery
- Gives a clear `ImportError` message if ROS 2 packages are not available

```bash
charuco-calibrate --ros-topic /camera/image_raw
```

#### B) Dedicated ROS 2 package (recommended for ROS workflows)

The `charuco_calibrator_ros` package provides a proper ROS 2 node where **all** configuration comes from ROS parameters and the **only** image input is a ROS topic. Build it with colcon inside your workspace:

```bash
colcon build --packages-select charuco_calibrator_ros
source install/setup.bash
ros2 launch charuco_calibrator_ros calibrator.launch.py
```

All board geometry, thresholds, and output settings are exposed as ROS parameters and can be set via a YAML parameter file or on the command line:

```bash
ros2 run charuco_calibrator_ros charuco_calibrator_node --ros-args \
    -p image_topic:=/camera/image_raw \
    -p squares_x:=8 \
    -p squares_y:=11 \
    -p auto_capture:=true
```

See `charuco_calibrator_ros/README.md` for full setup instructions, the complete parameter list, and troubleshooting.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture current frame (manual) |
| `A` | Toggle auto-capture mode |
| `C` | Run calibration |
| `R` | Reset all data (observations, coverage, heatmap) |
| `S` | Save calibration YAML + observations NPZ |
| `H` | Toggle heatmap overlay |
| `Q` / `ESC` | Quit |

---

## Configuration

All parameters can be set via a YAML config file and/or CLI flags. CLI flags override the config file, which overrides built-in defaults.

```bash
charuco-calibrate --config my_config.yaml --camera 0 --output-dir results/
```

See `config/default_config.yaml` for the full reference with all parameters documented.

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML config file |
| `--camera <id>` | Camera device ID (integer) |
| `--video <path>` | Path to a video file |
| `--ros-topic <topic>` | ROS 2 image topic name |
| `--output-dir <dir>` | Output directory for calibration files |
| `--camera-name <name>` | Camera name written into the YAML |

### Config Sections

**board** — Board geometry: `squares_x`, `squares_y`, `square_length`, `marker_length`, `aruco_dict`

**thresholds** — Acceptance criteria: `min_corners`, `min_blur_var`, `min_score`, `capture_cooldown_ms`

**coverage** — Grid dimensions: `grid_cols`, `grid_rows`, `scale_bins`

**source** — Image source: `camera_id`, `video_path`, `ros_topic`, `width`, `height`

**output** — Export settings: `output_dir`, `camera_name`, `save_observations`

**Top-level** — `auto_capture` (bool), `recalibrate_every` (int)

---

## Supported ArUco Dictionaries

Any of the standard OpenCV ArUco dictionaries can be used:

`DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`,
`DICT_5X5_50`, `DICT_5X5_100`, `DICT_5X5_250`, `DICT_5X5_1000`,
`DICT_6X6_50`, `DICT_6X6_100`, `DICT_6X6_250`, `DICT_6X6_1000`,
`DICT_7X7_50`, `DICT_7X7_100`, `DICT_7X7_250`, `DICT_7X7_1000`

---

## Tips for a Good Calibration

1. **Print a flat board** — mount it on rigid, flat material (foam board, clipboard)
2. **Fill the grid** — move the board to cover all 16 cells of the coverage grid
3. **Vary distance** — show the board close-up and far away to fill the scale bins
4. **Vary angle** — tilt the board in different directions for view diversity
5. **Stay sharp** — hold still at each position; blurry frames are rejected automatically
6. **Watch the quality meter** — aim for 80%+ before running calibration
7. **Collect 20–40 frames** — more diverse frames generally reduce RMS error
8. **Check RMS** — a good monocular calibration typically achieves RMS < 1.0 px
