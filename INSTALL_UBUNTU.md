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

## Notes

- **Headless servers:** OpenCV's `imshow` requires a display. On a headless machine you need X11 forwarding (`ssh -X`) or a physical monitor.
- **Verify camera:** Use `v4l2-ctl --list-devices` (from `v4l-utils`) to check that your camera is detected.
