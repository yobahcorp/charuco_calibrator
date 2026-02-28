# Session Log

## 2026-02-28

### Changes

- Added `--image-folder` source option for streaming from a directory of images (sorted alphabetically)
  - New `ImageFolderSource` class in `image_source.py`
  - Updated `SourceConfig`, `create_source()` factory, CLI args, and end-of-stream handling
  - Commits: `e413a2d`, `25c4470`

- Added `show_heatmap` config option to start with heatmap enabled
  - New field in `AppConfig`, wired to `main.py` init
  - Updated default config: `auto_capture: true`, `show_heatmap: true`, board 9x7 DICT_7X7_50
  - Commit: `97ed0ae`

- Window mode changes
  - Tried maximized (non-fullscreen) with `WINDOW_KEEPRATIO` and letterboxing approaches
  - Reverted to fullscreen due to macOS compatibility issues
  - Commits: `1865003`, `7d2ba04`, `6e21ce3`, `e0f9dcd`, `e51cf03`, `b3b25c6`

- Updated all documentation for `--image-folder`
  - README.md, FEATURES.md, CLAUDE_SPEC.md, INSTALL_UBUNTU.md, docs/index.html
  - Commit: `25c4470`

- Built package v0.1.0 (sdist + wheel in `dist/`)

- Created `BACKLOG.md` with 15 planned features
  - Background calibration thread + progress indicator
  - GPU acceleration (UMat/CUDA)
  - Undistortion preview, undo, auto-save, stereo, board generator
  - FPS counter, per-view errors, capture pulse, guidance arrows, dark hint bar

- Implemented backlog Features 1A+1B+1C, 8, 10, 12 (first batch)
  - **Feature 1A+1C**: Background calibration thread — `calibrate_async()` with `threading.Thread`, lock-protected observation snapshot, `is_calibrating` property
  - **Feature 1B**: Calibrating progress indicator — pulsing "Calibrating..." overlay with animated dots
  - **Feature 12**: Dark help hint bar — semi-transparent strip behind bottom keyboard hints
  - **Feature 8**: FPS counter — rolling 30-frame average displayed in top-right of status panel
  - **Feature 10**: Capture visual pulse — green border flash on frame capture (`draw_border_flash()`)
  - Added 3 new tests for `calibrate_async()` (60 total tests, all passing)
  - Removed completed features from BACKLOG.md, updated priority order

- Implemented backlog Features 13, 4, 3, 5 (second batch)
  - **Feature 13**: Auto-prune outlier frames — `prune_outliers()` removes frames with per-view error > threshold * mean, re-calibrates iteratively. Config: `auto_prune: true`, `prune_threshold: 2.0`
  - **Feature 4**: Frame deletion / undo — `Z` key removes last observation via `pop_observation()`, flash message shows remaining count
  - **Feature 3**: Undistortion preview — `U` key toggles live `cv2.undistort()` overlay, "Undistort: ON" in status panel, only after calibration
  - **Feature 5**: Auto-save on quit — `Q`/`ESC` prompts "Save before quitting? Y/N" when unsaved calibration exists, tracks `_saved` flag
  - Added 4 new tests for `pop_observation()` and `prune_outliers()` (64 total, all passing)
  - Updated help hints: added U:undistort and Z:undo
  - Removed completed features from BACKLOG.md, updated priority order

### Decisions

- Image folder frames sorted alphabetically (standard `sorted()`)
- Source priority: `ros_topic` > `image_folder` > `video_path` > `camera_id`
- Fullscreen window mode kept (maximized alternatives didn't work reliably on macOS)
- GPU acceleration planned via UMat first (transparent), CUDA optional
- Background calibration uses snapshot of observations (copied under lock) to avoid contention with main thread
- Calibration completion detected via `_cal_flash_shown` flag to show RMS flash once
