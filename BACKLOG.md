# Backlog

---

## Feature 2B: CUDA Acceleration (Optional)

**Files:** `charuco_calibrator/image_source.py`, `charuco_calibrator/scoring.py`, `charuco_calibrator/heatmap.py`

- Gate behind `cv2.cuda.getCudaEnabledDeviceCount() > 0` check
- Use `cv2.cuda.GpuMat` for:
  - `cv2.cuda.resize()` — frame resize in image sources
  - `cv2.cuda.cvtColor()` — BGR to GRAY conversion
  - Heatmap blending via `cv2.cuda.addWeighted()`
- Fallback to CPU path when CUDA is not available
- **Note:** Not applicable for AMD GPUs (A2141 MacBook Pro)
- **Note:** ArUco detection has no CUDA implementation in OpenCV
- **Note:** `cv2.calibrateCamera()` has no CUDA variant — threading is the solution

---

## Feature 6: Stereo Calibration

**Files:** `charuco_calibrator/config.py`, `charuco_calibrator/image_source.py`, `charuco_calibrator/calibration.py`, `charuco_calibrator/main.py`

- Add `--camera2` / `--image-folder2` CLI flags for a second source
- Add `StereoCalibrationManager` using `cv2.stereoCalibrate()`
- Output stereo calibration YAML with rotation/translation between cameras
- Dual-view UI layout showing both camera feeds side by side

---

## Feature 14: Web UI

**Branch:** `feature/web-ui`

**New files:** `charuco_calibrator/web_server.py`, `charuco_calibrator/static/index.html`, `charuco_calibrator/static/app.js`, `charuco_calibrator/static/style.css`

**Modified files:** `charuco_calibrator/main.py`, `charuco_calibrator/config.py`, `pyproject.toml`

- FastAPI backend serving the calibration loop, streaming frames via WebSocket (MJPEG-encoded)
- Single-page HTML/JS/CSS frontend with proper layout:
  - Video stream panel (detection overlay + heatmap only, no text overlays)
  - Side panel: coverage grid, quality meter, per-view error bars, score breakdown
  - Top bar: frames, auto, RMS, FPS, dictionary
  - Bottom controls: buttons for capture, auto, calibrate, reset, save, heatmap, undistort, undo
- Keyboard shortcuts still work in the browser
- `--web` flag to launch web UI (default), `--no-web` to fall back to OpenCV window
- Dependencies: `fastapi`, `uvicorn`, `websockets`
- Existing OpenCV UI remains as fallback

---

## Priority Order

1. **14** — Web UI (branch: `feature/web-ui`)
2. **6** — Stereo calibration (large scope, separate milestone)
3. **2B** — CUDA acceleration (only if UMat insufficient)
