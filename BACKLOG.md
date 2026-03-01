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

## Priority Order

1. **6** — Stereo calibration (large scope, separate milestone)
2. **2B** — CUDA acceleration (only if UMat insufficient)
