# Installing on Ubuntu

## 1. System dependencies

OpenCV requires system libraries for GUI display and camera access:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 v4l-utils
```

> If `libgl1-mesa-glx` doesn't exist on newer Ubuntu, try `libgl1` instead.

## 2. Clone the project

```bash
git clone <your-repo-url>
cd charuco-calibrator
```

## 3. Create a virtual environment and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 4. (Optional) Install dev tools for tests/linting

```bash
pip install -e ".[dev]"
```

## 5. Run

```bash
charuco-calibrate --camera 0
```

## 6. Run tests

```bash
pytest tests/
```

## 7. (Optional) ROS 2 Humble setup

If you want to run the calibrator as a ROS 2 node with all configuration via ROS parameters, follow these additional steps. This assumes ROS 2 Humble is already installed (`ros-humble-desktop`).

### Source ROS and create a workspace

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/calibration_ws/src
```

### Symlink or copy the repo into the workspace

```bash
ln -s /path/to/charuco-calibrator ~/calibration_ws/src/
```

### Install the core package into the ROS Python environment

```bash
cd ~/calibration_ws/src/charuco-calibrator
pip install -e .
```

### Build the ROS package

```bash
cd ~/calibration_ws
colcon build --packages-select charuco_calibrator_ros
source install/setup.bash
```

### Run

Start a camera driver (e.g. `usb_cam` or `v4l2_camera`) in one terminal, then launch the calibrator in another:

```bash
ros2 launch charuco_calibrator_ros calibrator.launch.py
```

Or run directly with parameter overrides:

```bash
ros2 run charuco_calibrator_ros charuco_calibrator_node --ros-args \
    -p image_topic:=/camera/image_raw \
    -p auto_capture:=true
```

See `charuco_calibrator_ros/README.md` for the full parameter list and troubleshooting.

## Notes

- **Headless servers:** OpenCV's `imshow` requires a display. On a headless machine you need X11 forwarding (`ssh -X`) or a physical monitor.
- **Verify camera:** Use `v4l2-ctl --list-devices` (from `v4l-utils`) to check that your camera is detected.
