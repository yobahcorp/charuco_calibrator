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

### Decisions

- Image folder frames sorted alphabetically (standard `sorted()`)
- Source priority: `ros_topic` > `image_folder` > `video_path` > `camera_id`
- Fullscreen window mode kept (maximized alternatives didn't work reliably on macOS)
- GPU acceleration planned via UMat first (transparent), CUDA optional
