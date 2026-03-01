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

## Feature 7: AR Object Overlay

**Files:** `charuco_calibrator/ar_overlay.py` (new), `charuco_calibrator/main.py`, `charuco_calibrator/detector.py`

Render virtual 3D objects on the ChArUco board in the live camera feed using the per-frame board pose (rvec/tvec) and calibrated intrinsics.

### Requirements

- New `--ar` CLI flag to enable AR overlay mode (combinable with normal calibration or standalone with `--calibration`)
- Detect the board each frame and recover `rvec`/`tvec` via `cv2.solvePnP` (or reuse the existing detection pipeline)
- Project 3D geometry onto the image with `cv2.projectPoints`

### Built-in Objects

| Object | Description |
|--------|-------------|
| **Coordinate axes** | RGB XYZ axes at the board origin (baseline, always available) |
| **Wireframe cube** | Unit cube sitting on the board surface, edges drawn with `cv2.line` |
| **Solid cube** | Filled faces with transparency via `cv2.fillPoly` + alpha blending |
| **Custom OBJ mesh** | Load a simple `.obj` file, project vertices, draw edges / filled faces |

- `--ar-object {axes,wireframe,solid,obj}` flag to select the object (default: `wireframe`)
- `--ar-obj-path <file>` for custom OBJ mesh
- `--ar-scale <float>` to control object size relative to board square length

### Rendering Pipeline

1. Detect board corners each frame (existing `detector.py`)
2. Estimate board pose with `cv2.solvePnP` using calibrated intrinsics
3. Define 3D vertices of the chosen object in board-local coordinates
4. Project to 2D with `cv2.projectPoints`
5. Draw edges/faces on the frame with OpenCV drawing primitives
6. Optional: basic lighting via face-normal dot product for shading

### Stretch Goals

- Smooth pose with exponential moving average to reduce jitter
- Object animation (rotation, bounce) driven by frame count
- Shadow projection onto the board plane
- Multiple objects placed at different board positions

---

## Feature 8: Synthetic Bokeh / Depth of Field

**Files:** `charuco_calibrator/bokeh.py` (new), `charuco_calibrator/main.py`

Apply a synthetic depth-of-field blur to the live feed using the known board distance from calibration. The board stays sharp while the background blurs, simulating a shallow-DOF camera or tilt-shift effect.

### Requirements

- New `--bokeh` CLI flag to enable the effect (requires prior calibration or `--calibration`)
- Compute board distance from `tvec[2]` (z-component of the translation vector)
- Define a focal plane at the board's depth; blur increases with distance from this plane

### Algorithm

1. Detect board each frame, recover `tvec` for depth
2. Create a depth proxy map:
   - Board region (convex hull of detected corners) = focal distance (sharp)
   - Pixels outside the hull = estimated background distance (blurred)
   - Use `cv2.GaussianBlur` with kernel size proportional to depth difference
3. Blend sharp and blurred images using the depth mask:
   - `output = sharp * mask + blurred * (1 - mask)`
   - Feather the mask edges with a Gaussian blur on the mask itself for smooth transitions

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bokeh-strength` | `25` | Max Gaussian kernel size (px) for the blur |
| `--bokeh-feather` | `31` | Kernel size for mask edge feathering |
| `--bokeh-mode` | `gaussian` | Blur type: `gaussian` or `lens` (disk kernel for circular bokeh) |

### Lens Blur (Stretch)

- Circular disk kernel convolution for more realistic bokeh circles on highlights
- Hexagonal kernel option for aperture-blade simulation
- Bright-pixel emphasis: boost highlights before blur for glowing bokeh balls

### Rendering Modes

- **Board focus** (default) — board is sharp, everything else blurs
- **Tilt-shift** — horizontal band of sharpness across the board, top/bottom blur (miniature effect)
- Toggle between modes with a keyboard shortcut (`B`)

---

## Priority Order

1. **7** — AR object overlay (high visual impact, leverages existing pose estimation)
2. **8** — Synthetic bokeh / depth of field (fun visual effect, moderate scope)
3. **6** — Stereo calibration (large scope, separate milestone)
4. **2B** — CUDA acceleration (only if UMat insufficient)
