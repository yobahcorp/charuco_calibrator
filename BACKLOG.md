# Backlog

## Context
All OpenCV operations run on CPU only — no GPU acceleration is used anywhere.

---

## Feature 2: GPU Acceleration

### Target Hardware

- MacBook Pro A2141 (2019 16") — AMD Radeon Pro 5300M/5500M
- CUDA is NOT available (NVIDIA only) — use OpenCL path

### Problem

- All OpenCV operations run on CPU
- Main per-frame bottlenecks: ArUco detection, blur variance (Laplacian), image resize
- Periodic bottleneck: `cv2.calibrateCamera()` (Levenberg-Marquardt solver)

### Plan

#### 2A. UMat (OpenCL) acceleration — Primary path (AMD/Intel/NVIDIA)

**Files:** `charuco_calibrator/detector.py`, `charuco_calibrator/scoring.py`, `charuco_calibrator/heatmap.py`, `charuco_calibrator/image_source.py`

- Convert input frames to `cv2.UMat` at the source level (`image_source.py`) — this enables transparent GPU offload for all downstream OpenCV calls that support it
- `cv2.UMat` uses OpenCL under the hood — works with AMD GPUs on macOS natively
- Verify OpenCL availability at startup: `cv2.ocl.haveOpenCL()` and `cv2.ocl.setUseOpenCL(True)`
- Log detected GPU device: `cv2.ocl.Device.getDefault().name()`
- Key operations that benefit: `cv2.cvtColor`, `cv2.Laplacian`, `cv2.resize`, `cv2.addWeighted`, `cv2.applyColorMap`
- **Limitation:** `cv2.aruco.detectMarkers()` and `cv2.calibrateCamera()` do NOT support UMat — must convert back to numpy via `umat.get()` for these calls
- Automatic CPU fallback when OpenCL is not available (UMat degrades gracefully)

#### 2B. CUDA acceleration — Optional, for NVIDIA hardware only

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

#### 2C. Add GPU config option

**Files:** `charuco_calibrator/config.py`, `config/default_config.yaml`

- Add `gpu: bool = False` to `AppConfig`
- Add `gpu: false` to default config YAML
- When enabled, detect available backend: OpenCL (AMD/Intel) or CUDA (NVIDIA)
- Log GPU device info at startup
- Auto-detect GPU availability and warn if `gpu: true` but no GPU found
- Graceful fallback to CPU when GPU is unavailable

---

## Feature 3: Undistortion Preview

**Files:** `charuco_calibrator/main.py`, `charuco_calibrator/ui.py`

- Add `U` key to toggle a live undistorted view using `cv2.undistort()` with the computed camera matrix
- Only available after calibration has been run
- Shows the corrected image so users can visually verify calibration quality before saving
- Add "Undistort: ON/OFF" to the status panel

---

## Feature 4: Frame Deletion / Undo

**Files:** `charuco_calibrator/calibration.py`, `charuco_calibrator/main.py`, `charuco_calibrator/ui.py`

- Add `Z` key to remove the last accepted observation
- `CalibrationManager.pop_observation()` — removes last entry from the observations list
- Update coverage state and heatmap accordingly (requires tracking per-frame contributions or recomputing)
- Flash message: "Removed frame N (X remaining)"

---

## Feature 5: Auto-Save on Quit

**Files:** `charuco_calibrator/main.py`, `charuco_calibrator/ui.py`

- When pressing `Q`/`ESC` with an unsaved calibration result, show a prompt: "Save before quitting? Y/N"
- Track a `_saved` flag on `CalibrationManager` that resets after each new calibration
- If user presses `Y`, save and quit; `N`, quit without saving

---

## Feature 6: Stereo Calibration

**Files:** `charuco_calibrator/config.py`, `charuco_calibrator/image_source.py`, `charuco_calibrator/calibration.py`, `charuco_calibrator/main.py`

- Add `--camera2` / `--image-folder2` CLI flags for a second source
- Add `StereoCalibrationManager` using `cv2.stereoCalibrate()`
- Output stereo calibration YAML with rotation/translation between cameras
- Dual-view UI layout showing both camera feeds side by side

---

## Feature 7: Board Generator

**Files:** `charuco_calibrator/main.py`, `charuco_calibrator/detector.py`

- Add `--print-board <output.png>` CLI flag
- Generate a printable ChArUco board image matching the current config using `board.generateImage()`
- Print to file and exit (no GUI needed)
- Saves users from needing a separate tool to create matching boards

---

## Feature 9: Per-View Error Display

**Files:** `charuco_calibrator/ui.py`, `charuco_calibrator/main.py`

- After calibrating, show per-view reprojection errors (data already computed in `compute_per_view_errors_full()`)
- Highlight the worst frames so users know which observations are dragging up the RMS
- Could display as a small bar chart overlay or list the top 3 worst frames

---

## Feature 11: Coverage Guidance Arrows

**Files:** `charuco_calibrator/ui.py`, `charuco_calibrator/scoring.py`

- Analyze empty grid cells and draw an arrow or highlight pointing toward the region needing the most coverage
- Guide the user where to move the board next
- Could use the centroid of uncovered cells to determine arrow direction

---

## Feature 13: Auto-Prune Low-Quality Frames

**Files:** `charuco_calibrator/calibration.py`, `charuco_calibrator/main.py`, `charuco_calibrator/scoring.py`

- After each `calibrateCamera()` call, compute per-view reprojection errors via `compute_per_view_errors_full()`
- Identify frames whose per-view error exceeds a threshold (e.g., 2x the mean RMS)
- Automatically remove those observations and re-calibrate
- Repeat until no outliers remain or a max iteration count is reached
- Flash message showing how many frames were pruned: "Pruned 3 frames, RMS: 0.42 -> 0.31"
- Add `auto_prune: bool = True` and `prune_threshold: float = 2.0` to config
- Also useful: reject frames at capture time if they would increase RMS above a threshold (requires trial calibration)

---

## Priority Order
1. **13** — Auto-prune low-quality frames (direct calibration improvement)
2. **4** — Frame deletion / undo (medium, common need)
3. **3** — Undistortion preview (medium, verification tool)
4. **5** — Auto-save on quit (medium, prevents data loss)
5. **2A** — UMat transparent acceleration (easiest GPU win)
6. **9** — Per-view error display (medium, data already available)
7. **11** — Coverage guidance arrows (medium, guides workflow)
8. **2C** — GPU config option
9. **7** — Board generator (nice-to-have utility)
10. **6** — Stereo calibration (large scope, separate milestone)
11. **2B** — CUDA acceleration (only if UMat insufficient)
