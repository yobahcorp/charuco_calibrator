"""CLI entry point and main calibration loop."""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2

from .calibration import CalibrationManager
from .config import AppConfig, load_config, apply_cli_overrides
from .detector import CharucoDetectorWrapper, probe_aruco_dictionaries
from .heatmap import CornerHeatmap
from .image_source import create_source
from .scoring import CoverageState, score_frame
from .ui import Action, UIRenderer


WINDOW_NAME = "ChArUco Calibrator"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChArUco camera calibration with live feedback"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file to use instead of a live camera",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="Path to a folder of images to use as frames",
    )
    parser.add_argument(
        "--ros-topic",
        type=str,
        default=None,
        help="ROS 2 image topic (requires rclpy + cv_bridge)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for calibration output files",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help="Camera name for the calibration YAML",
    )
    parser.add_argument(
        "--print-board",
        type=str,
        default=None,
        metavar="OUTPUT",
        help="Generate a printable ChArUco board image and exit",
    )
    return parser.parse_args(argv)


def _init_gpu(enabled: bool) -> bool:
    """Initialize GPU acceleration if requested and available.

    Returns True if OpenCL was successfully enabled.
    """
    if not enabled:
        return False

    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        try:
            device = cv2.ocl.Device.getDefault()
            name = device.name() if device else "unknown"
        except Exception:
            name = "unknown"
        print(f"GPU: OpenCL enabled — {name}")
        return True

    print("GPU: OpenCL not available, falling back to CPU")
    return False


def run(cfg: AppConfig) -> int:
    """Run the calibration loop with the given config.

    This can be called directly with a programmatically-built AppConfig
    (e.g. from a ROS node) without going through CLI arg parsing.
    """
    # GPU initialization
    gpu_enabled = _init_gpu(cfg.gpu)

    # Initialize components
    detector = CharucoDetectorWrapper(cfg.board)
    coverage = CoverageState(
        grid_cols=cfg.coverage.grid_cols,
        grid_rows=cfg.coverage.grid_rows,
        scale_bins=cfg.coverage.scale_bins,
    )
    cal_manager = CalibrationManager()
    ui = UIRenderer()
    heatmap: CornerHeatmap | None = None
    show_heatmap = cfg.show_heatmap
    auto_capture = cfg.auto_capture
    show_undistort = False
    _saved = False

    # Image source
    source = create_source(cfg.source)
    if not source.is_open:
        print("ERROR: Could not open image source.", file=sys.stderr)
        return 1

    last_capture_time = 0.0
    _cal_flash_shown = True  # No pending flash at start
    _frame_times: collections.deque[float] = collections.deque(maxlen=30)
    no_detect_count = 0
    dict_probed = False
    suggested_dict: str | None = None
    tried_dicts: set[str] = {cfg.board.aruco_dict.upper()}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                # For video files / image folders, we've reached the end
                if cfg.source.video_path or cfg.source.image_folder:
                    break
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Lazy-init heatmap to frame size
            if heatmap is None:
                heatmap = CornerHeatmap(h, w)
                cal_manager.image_size = (w, h)

            now = time.time()
            _frame_times.append(now)

            # Compute rolling FPS
            if len(_frame_times) >= 2:
                fps = (len(_frame_times) - 1) / (_frame_times[-1] - _frame_times[0])
            else:
                fps = 0.0

            # Detect
            result = detector.detect(frame)

            # Dictionary auto-detection probe
            if result.valid:
                no_detect_count = 0
                dict_probed = False
                suggested_dict = None
                tried_dicts = {cfg.board.aruco_dict.upper()}
                ui.clear_prompt()
            else:
                no_detect_count += 1
                if no_detect_count >= 30 and not dict_probed:
                    dict_probed = True
                    probe_hits = probe_aruco_dictionaries(frame, cfg.board)
                    alt_hits = [
                        (name, cnt) for name, cnt in probe_hits
                        if name not in tried_dicts
                    ]
                    if alt_hits:
                        best_name, best_count = alt_hits[0]
                        suggested_dict = best_name
                        ui.set_prompt(
                            f"Found {best_name} ({best_count} markers). "
                            f"Switch dictionary? Y/N"
                        )
                    else:
                        suggested_dict = None
                        ui.set_prompt(
                            "No ArUco markers found with any dictionary [N to dismiss]"
                        )

            # Score
            frame_score = score_frame(
                result,
                frame,
                coverage,
                detector.total_corners,
                min_corners=cfg.thresholds.min_corners,
                min_blur_var=cfg.thresholds.min_blur_var,
                min_score=cfg.thresholds.min_score,
            )

            # Draw detection overlay
            vis = detector.draw_detection(frame, result)

            # Undistortion preview
            if show_undistort and cal_manager.result and cal_manager.result.valid:
                vis = cv2.undistort(
                    vis,
                    cal_manager.result.camera_matrix,
                    cal_manager.result.dist_coeffs,
                )

            # Heatmap blend
            if show_heatmap and heatmap is not None:
                vis = heatmap.blend_onto(vis)

            # Poll keyboard
            action = ui.poll_key(wait_ms=1)

            # Auto-capture check
            should_capture = False
            if action == Action.CAPTURE:
                should_capture = True
            elif auto_capture and not frame_score.rejected:
                cooldown = cfg.thresholds.capture_cooldown_ms / 1000.0
                if now - last_capture_time >= cooldown:
                    should_capture = True

            # Handle capture
            if should_capture and result.valid and not frame_score.rejected:
                obj_pts, img_pts = detector.match_image_points(
                    result.charuco_corners, result.charuco_ids
                )
                cal_manager.add_observation(obj_pts, img_pts, result.charuco_ids)
                coverage.update(result.charuco_corners, h, w)
                heatmap.add_corners(result.charuco_corners)
                last_capture_time = now
                ui.trigger_flash("ACCEPTED", now)
                ui.trigger_border_flash(now)

                # Auto-recalibrate
                if (
                    cal_manager.num_observations >= 4
                    and cal_manager.num_observations % cfg.recalibrate_every == 0
                ):
                    cal_manager.calibrate_async((w, h))
                    _cal_flash_shown = False

            # Handle dictionary switch confirmation
            if action == Action.CONFIRM and suggested_dict:
                cfg.board.aruco_dict = suggested_dict
                tried_dicts.add(suggested_dict.upper())
                detector.reinitialize(cfg.board)
                ui.clear_prompt()
                ui.trigger_flash(f"Switched to {suggested_dict}", now)
                no_detect_count = 0
                dict_probed = False
                suggested_dict = None
            elif action == Action.DENY:
                ui.clear_prompt()
                suggested_dict = None

            # Check if background calibration just finished
            if (
                not cal_manager.is_calibrating
                and cal_manager.result
                and cal_manager.result.valid
                and not _cal_flash_shown
            ):
                # Auto-prune outlier frames
                if cfg.auto_prune and cal_manager.num_observations >= 4:
                    pruned, old_rms, new_rms = cal_manager.prune_outliers(
                        threshold=cfg.prune_threshold,
                    )
                    if pruned > 0:
                        ui.trigger_flash(
                            f"Pruned {pruned} frames, RMS: {old_rms:.3f} -> {new_rms:.3f}",
                            now,
                            duration=1.5,
                        )
                    else:
                        ui.trigger_flash(
                            f"Calibrated! RMS={cal_manager.result.rms:.3f}", now
                        )
                else:
                    ui.trigger_flash(
                        f"Calibrated! RMS={cal_manager.result.rms:.3f}", now
                    )
                _cal_flash_shown = True
                _saved = False  # New calibration invalidates previous save

            # Handle other actions
            if action == Action.TOGGLE_AUTO:
                auto_capture = not auto_capture
            elif action == Action.CALIBRATE:
                if cal_manager.num_observations >= 4:
                    cal_manager.calibrate_async((w, h))
                    _cal_flash_shown = False
                else:
                    ui.trigger_flash(
                        f"Need >= 4 frames (have {cal_manager.num_observations})", now
                    )
            elif action == Action.RESET:
                coverage.reset()
                cal_manager.reset()
                heatmap.reset()
                ui.trigger_flash("RESET", now)
            elif action == Action.SAVE:
                if cal_manager.result and cal_manager.result.valid:
                    out_dir = Path(cfg.output.output_dir)
                    yaml_path = cal_manager.save_yaml(
                        out_dir / "calibration.yaml", cfg.output.camera_name
                    )
                    if cfg.output.save_observations:
                        cal_manager.save_observations_npz(out_dir / "observations.npz")
                    ui.trigger_flash(f"Saved to {yaml_path}", now, duration=1.5)
                    _saved = True
                else:
                    ui.trigger_flash("Calibrate first (press C)", now)
            elif action == Action.TOGGLE_HEATMAP:
                show_heatmap = not show_heatmap
            elif action == Action.UNDO:
                removed = cal_manager.pop_observation()
                if removed is not None:
                    ui.trigger_flash(
                        f"Removed frame ({cal_manager.num_observations} remaining)",
                        now,
                    )
                else:
                    ui.trigger_flash("No frames to undo", now)
            elif action == Action.UNDISTORT:
                if cal_manager.result and cal_manager.result.valid:
                    show_undistort = not show_undistort
                else:
                    ui.trigger_flash("Calibrate first (press C)", now)
            elif action == Action.QUIT:
                # Auto-save prompt if unsaved calibration exists
                if (
                    cal_manager.result
                    and cal_manager.result.valid
                    and not _saved
                ):
                    ui.set_prompt("Save before quitting? Y/N")
                    cv2.imshow(WINDOW_NAME, vis)
                    while True:
                        save_action = ui.poll_key(wait_ms=50)
                        if save_action == Action.CONFIRM:
                            out_dir = Path(cfg.output.output_dir)
                            yaml_path = cal_manager.save_yaml(
                                out_dir / "calibration.yaml",
                                cfg.output.camera_name,
                            )
                            if cfg.output.save_observations:
                                cal_manager.save_observations_npz(
                                    out_dir / "observations.npz"
                                )
                            ui.clear_prompt()
                            ui.trigger_flash(
                                f"Saved to {yaml_path}", now, duration=0.5
                            )
                            break
                        elif save_action == Action.DENY:
                            ui.clear_prompt()
                            break
                break

            # Draw UI overlays
            rms = cal_manager.result.rms if cal_manager.result and cal_manager.result.valid else None
            vis = ui.draw_status_panel(
                vis,
                num_accepted=cal_manager.num_observations,
                score=frame_score,
                rms=rms,
                auto_capture=auto_capture,
                show_heatmap=show_heatmap,
                aruco_dict=cfg.board.aruco_dict,
                fps=fps,
                show_undistort=show_undistort,
            )
            vis = ui.draw_prompt(vis)
            if cal_manager.result and cal_manager.result.valid:
                per_view = cal_manager.compute_per_view_errors_full()
                vis = ui.draw_per_view_errors(vis, per_view)
            vis = ui.draw_coverage_grid(vis, coverage)
            vis = ui.draw_guidance_arrow(vis, coverage)
            vis = ui.draw_quality_meter(vis, coverage.quality_meter)
            vis = ui.draw_accepted_flash(vis, now)
            vis = ui.draw_border_flash(vis, now)
            if cal_manager.is_calibrating:
                vis = ui.draw_calibrating_indicator(vis, now)
            vis = ui.draw_help_hint(vis)

            cv2.imshow(WINDOW_NAME, vis)

    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, load config, and run."""
    args = parse_args(argv)

    config_path = args.config or "config/default_config.yaml"
    cfg = load_config(config_path)
    cfg = apply_cli_overrides(cfg, args)

    # Board generator mode — print and exit
    if getattr(args, "print_board", None):
        detector = CharucoDetectorWrapper(cfg.board)
        board_img = detector.generate_board_image(800, 1000)
        out_path = Path(args.print_board)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), board_img)
        print(f"Board image saved to {out_path}")
        return 0

    return run(cfg)


def cli() -> None:
    """Console script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    cli()
