# charuco_calibrator_ros

ROS 2 wrapper for the [ChArUco camera calibrator](../). All image input comes
from a ROS topic (`sensor_msgs/msg/Image`). Board geometry, quality thresholds,
and output settings are configured entirely through ROS parameters.

Tested on **ROS 2 Humble** (Ubuntu 22.04).

## Prerequisites

| Requirement | Notes |
|---|---|
| ROS 2 Humble desktop | `ros-humble-desktop` (includes `rclpy`, `sensor_msgs`, `cv_bridge`) |
| Python >= 3.8 | Shipped with Humble |
| charuco-calibrator (core package) | The sibling directory in this repo |
| A camera driver publishing `sensor_msgs/msg/Image` | e.g. `usb_cam`, `v4l2_camera`, `realsense2_camera` |

## Installation

### 1. Install ROS 2 Humble

Follow the official guide if you don't have it yet:
<https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>

Then source the setup file:

```bash
source /opt/ros/humble/setup.bash
```

### 2. Create (or reuse) a colcon workspace

```bash
mkdir -p ~/calibration_ws/src
cd ~/calibration_ws/src
```

### 3. Clone or symlink this repository

```bash
# Option A: clone the whole repo
git clone <your-repo-url>

# Option B: symlink if you already have the repo on disk
ln -s /path/to/charuco-calibrator .
```

### 4. Install the core calibrator package

The ROS node imports `charuco_calibrator` at runtime, so install it into the
Python environment that ROS uses:

```bash
cd ~/calibration_ws/src/charuco-calibrator   # or wherever the repo root is
pip install -e .
```

> If you are using the system Python managed by ROS, you may need
> `pip install --user -e .` or install inside the workspace venv that colcon
> uses.

### 5. Build the ROS package

```bash
cd ~/calibration_ws
colcon build --packages-select charuco_calibrator_ros
source install/setup.bash
```

## Quick start

### Start a camera driver

You need a node that publishes `sensor_msgs/msg/Image`. For example, with
`usb_cam`:

```bash
# Terminal 1
ros2 run usb_cam usb_cam_node_exe --ros-args -p pixel_format:=yuyv
```

Or with `v4l2_camera`:

```bash
ros2 run v4l2_camera v4l2_camera_node
```

Verify the topic is live:

```bash
ros2 topic list          # look for /image_raw or /camera/image_raw
ros2 topic hz /image_raw # confirm frames are arriving
```

### Launch the calibrator

```bash
# Terminal 2
source ~/calibration_ws/install/setup.bash
ros2 launch charuco_calibrator_ros calibrator.launch.py
```

A fullscreen OpenCV window opens showing the live camera feed with ChArUco
detection overlays. Point your camera at a printed ChArUco board and use the
keyboard controls below to capture frames and calibrate.

### Override the image topic

If your camera publishes on a different topic:

```bash
ros2 launch charuco_calibrator_ros calibrator.launch.py \
    image_topic:=/my_camera/image_raw
```

### Run the node directly (without the launch file)

```bash
ros2 run charuco_calibrator_ros charuco_calibrator_node \
    --ros-args -p image_topic:=/camera/image_raw
```

You can override any parameter on the command line:

```bash
ros2 run charuco_calibrator_ros charuco_calibrator_node --ros-args \
    -p image_topic:=/camera/image_raw \
    -p squares_x:=8 \
    -p squares_y:=11 \
    -p square_length:=0.025 \
    -p marker_length:=0.019 \
    -p auto_capture:=true
```

## Keyboard controls

Once the calibrator window is open:

| Key | Action |
|---|---|
| **Space** | Capture current frame |
| **A** | Toggle auto-capture on/off |
| **C** | Run calibration (needs >= 4 captured frames) |
| **S** | Save calibration YAML and observations to disk |
| **R** | Reset all captured frames and heatmap |
| **H** | Toggle corner heatmap overlay |
| **Y / N** | Accept / dismiss a suggested ArUco dictionary switch |
| **Q** or **Esc** | Quit |

## ROS parameters

All parameters are declared with sensible defaults. Override them via a YAML
file or on the command line.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_topic` | string | `/camera/image_raw` | ROS image topic to subscribe to |
| `aruco_dict` | string | `DICT_4X4_50` | ArUco dictionary name (e.g. `DICT_5X5_100`, `DICT_6X6_250`) |
| `squares_x` | int | `5` | Number of chessboard squares along X |
| `squares_y` | int | `7` | Number of chessboard squares along Y |
| `square_length` | double | `0.04` | Chessboard square side length in metres |
| `marker_length` | double | `0.03` | ArUco marker side length in metres |
| `min_corners` | int | `6` | Minimum detected corners to accept a frame |
| `min_blur_var` | double | `50.0` | Minimum Laplacian variance (blur rejection) |
| `min_score` | double | `0.3` | Minimum overall quality score to accept |
| `capture_cooldown_ms` | int | `1500` | Milliseconds between auto-captures |
| `output_dir` | string | `calibration_output` | Directory for output files |
| `camera_name` | string | `camera` | Camera name written into the calibration YAML |
| `auto_capture` | bool | `false` | Start with auto-capture enabled |

### Using a custom parameter file

Copy and edit the default file:

```bash
cp ~/calibration_ws/install/charuco_calibrator_ros/share/charuco_calibrator_ros/config/default_params.yaml \
   ~/my_params.yaml
# edit ~/my_params.yaml
```

Then pass it to the launch file:

```bash
ros2 launch charuco_calibrator_ros calibrator.launch.py \
    params_file:=$HOME/my_params.yaml
```

The YAML format follows the standard ROS 2 parameter file convention:

```yaml
charuco_calibrator:
  ros__parameters:
    image_topic: /camera/image_raw
    aruco_dict: DICT_4X4_50
    squares_x: 5
    squares_y: 7
    square_length: 0.04
    marker_length: 0.03
    min_corners: 6
    min_blur_var: 50.0
    min_score: 0.3
    capture_cooldown_ms: 1500
    output_dir: calibration_output
    camera_name: camera
    auto_capture: false
```

## Output files

After pressing **C** (calibrate) and then **S** (save), two files are written
to the `output_dir`:

| File | Contents |
|---|---|
| `calibration.yaml` | Camera intrinsics in ROS `camera_info` compatible format |
| `observations.npz` | Raw corner observations (NumPy archive) for reproducibility |

## Typical calibration workflow

1. Print a ChArUco board matching your parameter settings (squares_x, squares_y,
   square_length, marker_length, aruco_dict). A reference board image is
   included at `charuco_board.png` in the repo root.
2. Measure the printed square and marker sizes with a ruler/caliper and set
   `square_length` and `marker_length` to the actual measured values in metres.
3. Start your camera driver node.
4. Launch the calibrator.
5. Move the board through the camera's field of view, covering corners, edges,
   and different distances. The coverage grid in the UI shows which regions
   still need more observations.
6. Capture at least 15-20 frames for a good calibration. Use **Space** for
   manual capture, or press **A** to enable auto-capture.
7. Press **C** to run calibration. Check the RMS reprojection error shown in
   the status panel (below 0.5 px is good, below 0.3 px is excellent).
8. Press **S** to save the result.

## Troubleshooting

**No image received / window stays black:**
- Check that the camera driver is publishing: `ros2 topic hz /camera/image_raw`
- Verify the `image_topic` parameter matches the actual topic name
- Ensure `cv_bridge` can decode the image encoding (BGR8, RGB8, mono8 are
  supported; Bayer or compressed topics need conversion first)

**"Could not open image source" error:**
- The node could not subscribe to the topic. Make sure `rclpy` is properly
  initialized (the ROS environment is sourced).

**OpenCV window doesn't appear:**
- `cv2.imshow` requires a display. On a headless machine, use X11 forwarding
  (`ssh -X`) or a physical monitor. Wayland users may need `QT_QPA_PLATFORM=xcb`.

**colcon build fails with "charuco_calibrator not found":**
- Install the core package first: `pip install -e .` from the repo root.

**Low detection rate:**
- The calibrator will auto-probe alternative ArUco dictionaries after 30 frames
  with no detection. Press **Y** to accept a suggested switch.
- Ensure your printed board matches the configured `aruco_dict`, `squares_x`,
  and `squares_y` exactly.
