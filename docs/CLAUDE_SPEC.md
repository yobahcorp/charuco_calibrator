# ChArUco Calibrator — Implementation Spec

## Goal
Build a standalone Python ChArUco camera calibration app with live feedback, coverage tracking,
heatmaps, and auto-capture. Works independently via OpenCV VideoCapture, with an optional
ROS 2 image topic adapter.

## Architecture
```
CLI (main.py)  ─or─  ROS node (calibrator_node.py)
       │                        │
       ▼                        ▼
  parse_args()         declare ROS params
  load_config()        build AppConfig
       │                        │
       └──────► run(cfg) ◄──────┘
                   │
     Config → ImageSource (camera / video / ROS topic)
                   ↓ frame
     Detector → Scoring → Accept? → Heatmap + CalibrationManager
                   └─→ UI (overlay + keyboard) → cv2.imshow
```
- The core calibration loop lives in `run(cfg: AppConfig)`, callable from the CLI or from the ROS node
- All modules are pure Python + OpenCV — zero ROS 2 dependency except the optional `RosImageSource`
- Uses OpenCV 4.8+ object-oriented ArUco API (`CharucoDetector.detectBoard`, `board.matchImagePoints`,
  `cv2.calibrateCamera`) with automatic legacy fallback
- A separate `charuco_calibrator_ros` ament_python package provides a ROS 2 node that reads all
  configuration from ROS parameters and feeds images exclusively via a ROS topic

## Core Features
1. **Image acquisition**: OpenCV VideoCapture (webcam/video file) or ROS 2 image topic (lazy import)
2. **ChArUco detection**: OpenCV 4.8+ `CharucoDetector.detectBoard()` with legacy API fallback
3. **Observation accumulation**: store matched object/image point pairs per accepted frame
4. **Calibration**: `cv2.calibrateCamera` with `board.matchImagePoints()` for correspondences
5. **YAML export**: ROS `camera_info_manager` compatible format

## Additional Features
### A) Live coverage feedback
- 4×4 grid bins with per-bin hit tracking
- Quadrant coverage (4 quadrants)
- Scale coverage (5 bins of apparent board size)
- View diversity (centroid position spread)
- **Quality meter**: weighted combo — grid×0.35 + quadrant×0.20 + scale×0.20 + diversity×0.25

### B) Corner heatmap
- Gaussian splat (σ=15px) at each accepted corner location
- Alpha-blended overlay on live image, toggled with H key

### C) Running reprojection error
- After calibration, RMS displayed live in status panel
- Auto-recalibrates every N accepted frames (configurable)
- Per-view error computation available

### D) Frame scoring + accept/reject
- **Frame score**: corner_ratio×0.3 + hull_spread×0.2 + new_coverage×0.3 + blur_norm×0.2
- Rejection criteria: too few corners, too blurry (Laplacian variance), score below threshold
- Manual capture (SPACE) and auto-capture with configurable cooldown

### E) UX
- Live overlay: detected markers/corners, score breakdown, coverage %, frame count, RMS
- "ACCEPTED" flash on capture
- Status panel, coverage grid, quality meter bar, help hints
- **Keyboard**: SPACE=capture, A=auto, C=calibrate, R=reset, S=save, H=heatmap, Q=quit

## File Layout
```
charuco_calibrator/
  __init__.py          # Version string
  config.py            # Dataclasses + YAML loading + CLI override
  image_source.py      # Abstract ImageSource, CameraSource, RosImageSource (lazy rclpy)
  detector.py          # CharucoBoard + CharucoDetector wrapper (OO + legacy fallback)
  scoring.py           # Frame scoring, blur, CoverageState, quality meter
  heatmap.py           # Gaussian splat accumulator + alpha-blend render
  calibration.py       # cv2.calibrateCamera wrapper, YAML export (ROS-compatible)
  ui.py                # Overlay drawing, coverage grid, quality bar, keyboard dispatch
  main.py              # run(cfg) calibration loop + CLI entry point
config/
  default_config.yaml  # Full reference config with all parameters documented
tests/
  conftest.py          # Shared fixtures (synthetic board images)
  test_config.py
  test_detector.py
  test_scoring.py
  test_heatmap.py
  test_calibration.py
  test_image_source.py
pyproject.toml         # Package metadata + charuco-calibrate entry point

charuco_calibrator_ros/          # ROS 2 ament_python wrapper package
  package.xml                    # ROS 2 package manifest
  setup.py                       # ament_python setup
  setup.cfg                      # ament install script dirs
  resource/
    charuco_calibrator_ros       # ament index marker (empty)
  charuco_calibrator_ros/
    __init__.py
    calibrator_node.py           # ROS node: declares params, builds AppConfig, calls run()
  launch/
    calibrator.launch.py         # Launch file with params_file + image_topic args
  config/
    default_params.yaml          # Default ROS parameter values
  README.md                      # Full ROS 2 usage documentation
```

## Configuration
All parameters live in a YAML config file (`config/default_config.yaml`) with CLI overrides:
- `--config <path>` — custom config file
- `--camera <id>` — camera device ID
- `--video <path>` — video file
- `--ros-topic <topic>` — ROS 2 image topic
- `--output-dir <dir>` — calibration output directory
- `--camera-name <name>` — camera name for YAML

## Usage

### Standalone (webcam)
```bash
pip install -e .
charuco-calibrate --camera 0
```

### Standalone (video file)
```bash
charuco-calibrate --video recording.mp4
```

### With ROS 2 — standalone CLI (requires rclpy + cv_bridge)
```bash
charuco-calibrate --ros-topic /camera/image_raw
```

### With ROS 2 — dedicated ROS package (recommended)
```bash
# Build inside a colcon workspace
colcon build --packages-select charuco_calibrator_ros
source install/setup.bash

# Launch with default parameters
ros2 launch charuco_calibrator_ros calibrator.launch.py

# Or override the image topic
ros2 launch charuco_calibrator_ros calibrator.launch.py image_topic:=/my_camera/image_raw

# Or run the node directly with parameter overrides
ros2 run charuco_calibrator_ros charuco_calibrator_node --ros-args \
    -p image_topic:=/camera/image_raw \
    -p squares_x:=8 \
    -p squares_y:=11 \
    -p auto_capture:=true
```

See `charuco_calibrator_ros/README.md` for full ROS 2 setup and usage instructions.

## Key Design Decisions
- **OpenCV 4.8+ OO API**: `CharucoDetector.detectBoard()` with automatic legacy fallback
- **Calibration**: `cv2.calibrateCamera` (not deprecated `calibrateCameraCharuco`)
- **Standalone-first**: works without ROS 2; ROS support via lazy-imported `RosImageSource`
- **`run(cfg)` extraction**: the calibration loop is a standalone function that accepts an `AppConfig`,
  enabling both CLI and ROS node entry points without code duplication
- **ROS package separation**: `charuco_calibrator_ros` is a separate ament_python package that depends
  on the core `charuco-calibrator` pip package — keeping ROS concerns out of the core library
- **YAML output**: ROS `camera_info_manager` compatible (camera_matrix, distortion_coefficients, etc.)

## Acceptance Criteria
- `charuco-calibrate --camera 0` opens webcam with live detection overlay
- ChArUco corners detected and drawn when board is visible
- SPACE captures frames; coverage grid fills up
- After ~20 frames, C calibrates — RMS appears in status panel
- S saves `calibration_output/calibration.yaml` with correct structure
- `pytest tests/` — all tests pass
- `--ros-topic` works with ROS 2 or gives graceful ImportError without it
- `colcon build --packages-select charuco_calibrator_ros` builds cleanly
- `ros2 launch charuco_calibrator_ros calibrator.launch.py` runs the calibrator with ROS topic input
- `ros2 run charuco_calibrator_ros charuco_calibrator_node --ros-args -p image_topic:=/camera/image_raw` works with parameter overrides
