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
     Config → ImageSource (camera / video / image folder / ROS topic)
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
1. **Image acquisition**: OpenCV VideoCapture (webcam/video file), image folder, or ROS 2 image topic (lazy import)
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
- Per-view error bar chart overlay (color-coded green/yellow/red relative to mean)

### D) Frame scoring + accept/reject
- **Frame score**: corner_ratio×0.3 + hull_spread×0.2 + new_coverage×0.3 + blur_norm×0.2
- Rejection criteria: too few corners, too blurry (Laplacian variance), score below threshold
- Manual capture (SPACE) and auto-capture with configurable cooldown

### E) Background calibration
- Calibration runs in a background thread (`calibrate_async()`) — UI never freezes
- Lock-protected observation snapshot to avoid contention with main thread
- Pulsing "Calibrating..." indicator with animated dots during computation
- RMS flash message on completion

### F) Auto-prune & undo
- Auto-prune outlier frames after calibration: removes frames with per-view error > threshold * mean, re-calibrates iteratively
- Config: `auto_prune: true`, `prune_threshold: 2.0`
- Frame undo (Z key): removes last captured observation
- Undistortion preview (U key): live `cv2.undistort()` toggle after calibration
- Auto-save prompt on quit: "Save before quitting? Y/N" when unsaved calibration exists

### G) GPU acceleration
- `gpu: false` config option — enables OpenCL transparent acceleration
- Auto-detects OpenCL at startup via `cv2.ocl.haveOpenCL()`, logs device name
- Graceful CPU fallback when GPU not available

### H) Board generator
- `--print-board <output.png>` generates a printable ChArUco board image matching config and exits
- No camera or display required

### J) Distortion visualization
- `--visualize --calibration <path>` loads a saved calibration YAML and overlays a distortion magnitude heatmap
- Precomputed via `cv2.initUndistortRectifyMap()` — pixel displacement magnitude rendered through JET colormap
- U key toggles between heatmap overlay and live undistorted preview
- Standalone event loop — no scoring, capture, or coverage logic
- Works with any source (camera, video, image folder)

### K) AR Object Overlay
- `--ar` enables AR overlay mode, combinable with normal calibration or standalone with `--calibration`
- Per-frame board pose estimation via `cv2.solvePnP` using matched charuco points + calibrated intrinsics
- Built-in objects: coordinate axes (RGB), wireframe cube, solid cube (alpha-blended faces), custom OBJ mesh
- `--ar-object {axes,wireframe,solid,obj}`, `--ar-obj-path`, `--ar-scale`
- EMA pose smoothing to reduce solvePnP jitter (`ar.smooth_alpha`)
- Geometry precomputed at init; per-frame cost is solvePnP + projectPoints + draw

### L) Synthetic Bokeh / Depth of Field
- `--bokeh` enables synthetic depth-of-field effect
- Board region (convex hull of detected corners) stays sharp, background blurs
- Feathered mask blending: `output = sharp * mask + blurred * (1 - mask)`
- Two focus modes (toggle with B key): board_focus (default) and tilt_shift (horizontal band)
- Two blur types: gaussian (GaussianBlur) and lens (circular disk kernel via filter2D)
- `--bokeh-strength`, `--bokeh-feather`, `--bokeh-mode`

### I) UX
- Live overlay: detected markers/corners, score breakdown, coverage %, frame count, RMS
- "ACCEPTED" flash on capture with green border pulse
- FPS counter (rolling 30-frame average) in status panel top-right
- Coverage guidance arrow pointing toward least-covered grid region
- Per-view error bar chart after calibration
- Status panel, coverage grid, quality meter bar, dark help hint bar
- **Keyboard**: SPACE=capture, A=auto, C=calibrate, R=reset, S=save, H=heatmap, U=undistort, Z=undo, B=bokeh, Q=quit

## File Layout
```
charuco_calibrator/
  __init__.py          # Version string
  config.py            # Dataclasses + YAML loading + CLI override
  image_source.py      # Abstract ImageSource, CameraSource, ImageFolderSource, RosImageSource (lazy rclpy)
  detector.py          # CharucoBoard + CharucoDetector wrapper (OO + legacy fallback)
  scoring.py           # Frame scoring, blur, CoverageState, quality meter
  heatmap.py           # Gaussian splat accumulator + alpha-blend render
  calibration.py       # cv2.calibrateCamera wrapper, async calibration, prune, YAML load/export
  ar_overlay.py        # AR object overlay — pose estimation (solvePnP), geometry, projection, rendering
  bokeh.py             # Synthetic bokeh — board mask, tilt-shift mask, gaussian/lens blur, blending
  visualize.py         # Distortion heatmap + visualization event loop (--visualize mode)
  ui.py                # Overlay drawing, coverage grid, quality bar, per-view errors, guidance arrow, keyboard dispatch
  main.py              # run(cfg) calibration loop + CLI entry point + board generator + visualize dispatch
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
  test_ar_overlay.py
  test_bokeh.py
  test_visualize.py
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
- `--image-folder <path>` — folder of images (sorted alphabetically)
- `--ros-topic <topic>` — ROS 2 image topic
- `--output-dir <dir>` — calibration output directory
- `--camera-name <name>` — camera name for YAML
- `--print-board <output>` — generate printable board image and exit
- `--visualize` — enter distortion visualization mode
- `--calibration <path>` — calibration YAML file (for `--visualize`, `--ar`, or `--bokeh`)
- `--ar` — enable AR object overlay
- `--ar-object <type>` — AR object type: `axes`, `wireframe`, `solid`, `obj`
- `--ar-obj-path <path>` — path to .obj file (for `--ar-object=obj`)
- `--ar-scale <float>` — AR object scale relative to board square length
- `--bokeh` — enable synthetic bokeh / depth-of-field effect
- `--bokeh-strength <int>` — max blur kernel size
- `--bokeh-feather <int>` — mask feathering kernel size
- `--bokeh-mode <type>` — blur type: `gaussian` or `lens`

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

### Standalone (image folder)
```bash
charuco-calibrate --image-folder /path/to/frames/
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

## Session & Backlog Rules

### Session Log (`SESSION.md`)
- Every working session must be logged in `SESSION.md` at the project root
- Each entry is dated (`## YYYY-MM-DD`) and contains:
  - **Changes** — what was added, modified, or removed, with commit hashes
  - **Decisions** — key design choices made and their rationale
- Entries are appended chronologically (newest at the bottom)
- Keep entries concise: bullet points, not paragraphs
- Log the session before the final commit/push

### Backlog (`BACKLOG.md`)
- All planned features and improvements live in `BACKLOG.md` at the project root
- Each feature gets its own `## Feature N: Title` section with:
  - **Files** — which source files will be modified
  - A brief description of what to implement
- A **Priority Order** section at the bottom ranks all items
- When a backlog item is completed:
  - Remove it from `BACKLOG.md`
  - Log it in `SESSION.md` under the current date
  - Update any affected documentation (README, FEATURES, CLAUDE_SPEC)
- New ideas or requested features are added to the backlog before implementation
- Do not implement backlog items without explicit user request
- **When to update the backlog during execution:**
  - **Before implementation** — add the feature to the backlog first
  - **During implementation** — if scope changes, new sub-tasks emerge, or blockers are discovered, update the feature's description and plan
  - **After implementation** — remove the completed item, log it in `SESSION.md`
  - **When priorities shift** — if a completed feature unblocks or changes the priority of other items, reorder the Priority Order section

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
