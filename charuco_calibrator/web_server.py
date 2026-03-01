"""FastAPI web server for browser-based calibration UI."""

from __future__ import annotations

import asyncio
import collections
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .calibration import CalibrationManager
from .config import AppConfig
from .detector import CharucoDetectorWrapper, probe_aruco_dictionaries
from .heatmap import CornerHeatmap
from .image_source import create_source
from .scoring import CoverageState, FrameScore, score_frame
from .ui import Action

_STATIC_DIR = Path(__file__).parent / "static"

# Action string -> Action enum mapping
_ACTION_MAP: dict[str, Action] = {
    "capture": Action.CAPTURE,
    "toggle_auto": Action.TOGGLE_AUTO,
    "calibrate": Action.CALIBRATE,
    "reset": Action.RESET,
    "save": Action.SAVE,
    "quit": Action.QUIT,
    "toggle_heatmap": Action.TOGGLE_HEATMAP,
    "undistort": Action.UNDISTORT,
    "undo": Action.UNDO,
    "confirm": Action.CONFIRM,
    "deny": Action.DENY,
}


def _action_from_string(action_str: str) -> Action:
    """Map a WebSocket action string to the Action enum."""
    return _ACTION_MAP.get(action_str.lower().strip(), Action.NONE)


class WebState:
    """Shared mutable state between the calibration loop thread and async handlers."""

    def __init__(self) -> None:
        self.latest_jpeg: bytes = b""
        self.latest_state: dict = {}
        self.action_queue: queue.Queue[str] = queue.Queue()
        self.video_clients: set[WebSocket] = set()
        self.state_clients: set[WebSocket] = set()
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()


def _build_state_dict(
    cal_manager: CalibrationManager,
    coverage: CoverageState,
    frame_score: Optional[FrameScore],
    auto_capture: bool,
    show_heatmap: bool,
    show_undistort: bool,
    fps: float,
    aruco_dict: str,
    is_calibrating: bool,
    flash_text: str,
    flash_until: float,
    prompt_text: str,
    stream_ended: bool,
) -> dict:
    """Build the JSON state dict from current calibration state."""
    now = time.time()

    rms = None
    per_view_errors: list[float] = []
    if cal_manager.result and cal_manager.result.valid:
        rms = round(cal_manager.result.rms, 4)
        pv = cal_manager.compute_per_view_errors_full()
        if len(pv) > 0:
            per_view_errors = [round(float(e), 4) for e in pv]

    score_dict = {
        "corner_ratio": 0.0,
        "hull_spread": 0.0,
        "new_coverage": 0.0,
        "blur_norm": 0.0,
        "total": 0.0,
        "rejected": True,
        "reject_reason": "no detection",
    }
    if frame_score is not None:
        score_dict = {
            "corner_ratio": round(frame_score.corner_ratio, 3),
            "hull_spread": round(frame_score.hull_spread, 3),
            "new_coverage": round(frame_score.new_coverage, 3),
            "blur_norm": round(frame_score.blur_norm, 3),
            "total": round(frame_score.total, 3),
            "rejected": frame_score.rejected,
            "reject_reason": frame_score.reject_reason,
        }

    coverage_grid = coverage.grid.tolist()

    return {
        "num_frames": cal_manager.num_observations,
        "auto_capture": auto_capture,
        "show_heatmap": show_heatmap,
        "show_undistort": show_undistort,
        "fps": round(fps, 1),
        "aruco_dict": aruco_dict,
        "rms": rms,
        "is_calibrating": is_calibrating,
        "stream_ended": stream_ended,
        "score": score_dict,
        "coverage_grid": coverage_grid,
        "grid_coverage_pct": round(coverage.grid_coverage * 100, 1),
        "quality_meter": round(coverage.quality_meter, 3),
        "per_view_errors": per_view_errors,
        "flash": {
            "text": flash_text,
            "active": now < flash_until,
        },
        "prompt": prompt_text,
    }


def _run_calibration_loop(cfg: AppConfig, web_state: WebState) -> None:
    """Calibration loop adapted from run() in main.py.

    Runs in a background thread. Writes JPEG frames and state dicts
    to web_state for the async broadcast tasks to pick up.
    """
    import sys

    # GPU initialization
    if cfg.gpu and cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)

    detector = CharucoDetectorWrapper(cfg.board)
    coverage = CoverageState(
        grid_cols=cfg.coverage.grid_cols,
        grid_rows=cfg.coverage.grid_rows,
        scale_bins=cfg.coverage.scale_bins,
    )
    cal_manager = CalibrationManager()
    heatmap: Optional[CornerHeatmap] = None
    show_heatmap = cfg.show_heatmap
    auto_capture = cfg.auto_capture
    show_undistort = False
    _saved = False

    source = create_source(cfg.source)
    if not source.is_open:
        print("ERROR: Could not open image source.", file=sys.stderr)
        web_state.shutdown_event.set()
        return

    last_capture_time = 0.0
    _cal_flash_shown = True
    _frame_times: collections.deque[float] = collections.deque(maxlen=30)
    no_detect_count = 0
    dict_probed = False
    suggested_dict: Optional[str] = None
    tried_dicts: set[str] = {cfg.board.aruco_dict.upper()}

    flash_text = ""
    flash_until = 0.0
    prompt_text = ""

    _last_frame = None
    _stream_ended = False

    def _trigger_flash(text: str, duration: float = 0.7) -> None:
        nonlocal flash_text, flash_until
        flash_text = text
        flash_until = time.time() + duration

    def _set_prompt(text: str) -> None:
        nonlocal prompt_text
        prompt_text = text

    def _clear_prompt() -> None:
        nonlocal prompt_text
        prompt_text = ""

    try:
        while not web_state.shutdown_event.is_set():
            if not _stream_ended:
                ok, frame = source.read()
                if not ok or frame is None:
                    if cfg.source.video_path or cfg.source.image_folder:
                        _stream_ended = True
                        frame = _last_frame
                        if frame is None:
                            break
                        _trigger_flash(
                            "Stream ended — use Save or Calibrate buttons",
                            duration=3.0,
                        )
                    else:
                        time.sleep(0.01)
                        continue
                _last_frame = frame
            else:
                frame = _last_frame

            h, w = frame.shape[:2]

            if heatmap is None:
                heatmap = CornerHeatmap(h, w)
                cal_manager.image_size = (w, h)

            now = time.time()
            _frame_times.append(now)

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
                _clear_prompt()
            else:
                no_detect_count += 1
                if no_detect_count >= 30 and not dict_probed:
                    dict_probed = True
                    probe_hits = probe_aruco_dictionaries(frame, cfg.board)
                    alt_hits = [
                        (name, cnt)
                        for name, cnt in probe_hits
                        if name not in tried_dicts
                    ]
                    if alt_hits:
                        best_name, best_count = alt_hits[0]
                        suggested_dict = best_name
                        _set_prompt(
                            f"Found {best_name} ({best_count} markers). "
                            f"Switch dictionary? Y/N"
                        )
                    else:
                        suggested_dict = None
                        _set_prompt(
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

            # Draw detection overlay (markers + corners)
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

            # Poll action queue
            action = Action.NONE
            try:
                action_str = web_state.action_queue.get_nowait()
                action = _action_from_string(action_str)
            except queue.Empty:
                pass

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
                _trigger_flash("ACCEPTED")

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
                _clear_prompt()
                _trigger_flash(f"Switched to {suggested_dict}")
                no_detect_count = 0
                dict_probed = False
                suggested_dict = None
            elif action == Action.DENY:
                _clear_prompt()
                suggested_dict = None

            # Check if background calibration just finished
            if (
                not cal_manager.is_calibrating
                and cal_manager.result
                and cal_manager.result.valid
                and not _cal_flash_shown
            ):
                if cfg.auto_prune and cal_manager.num_observations >= 4:
                    pruned, old_rms, new_rms = cal_manager.prune_outliers(
                        threshold=cfg.prune_threshold,
                    )
                    if pruned > 0:
                        _trigger_flash(
                            f"Pruned {pruned} frames, RMS: {old_rms:.3f} -> {new_rms:.3f}",
                            duration=1.5,
                        )
                    else:
                        _trigger_flash(
                            f"Calibrated! RMS={cal_manager.result.rms:.3f}"
                        )
                else:
                    _trigger_flash(
                        f"Calibrated! RMS={cal_manager.result.rms:.3f}"
                    )
                _cal_flash_shown = True
                _saved = False

            # Handle other actions
            if action == Action.TOGGLE_AUTO:
                auto_capture = not auto_capture
            elif action == Action.CALIBRATE:
                if cal_manager.num_observations >= 4:
                    cal_manager.calibrate_async((w, h))
                    _cal_flash_shown = False
                else:
                    _trigger_flash(
                        f"Need >= 4 frames (have {cal_manager.num_observations})"
                    )
            elif action == Action.RESET:
                coverage.reset()
                cal_manager.reset()
                heatmap.reset()
                _trigger_flash("RESET")
            elif action == Action.SAVE:
                if cal_manager.result and cal_manager.result.valid:
                    out_dir = Path(cfg.output.output_dir)
                    yaml_path = cal_manager.save_yaml(
                        out_dir / "calibration.yaml", cfg.output.camera_name
                    )
                    if cfg.output.save_observations:
                        cal_manager.save_observations_npz(
                            out_dir / "observations.npz"
                        )
                    _trigger_flash(f"Saved to {yaml_path}", duration=1.5)
                    _saved = True
                else:
                    _trigger_flash("Calibrate first (press C)")
            elif action == Action.TOGGLE_HEATMAP:
                show_heatmap = not show_heatmap
            elif action == Action.UNDO:
                removed = cal_manager.pop_observation()
                if removed is not None:
                    _trigger_flash(
                        f"Removed frame ({cal_manager.num_observations} remaining)"
                    )
                else:
                    _trigger_flash("No frames to undo")
            elif action == Action.UNDISTORT:
                if cal_manager.result and cal_manager.result.valid:
                    show_undistort = not show_undistort
                else:
                    _trigger_flash("Calibrate first (press C)")
            elif action == Action.QUIT:
                if (
                    cal_manager.result
                    and cal_manager.result.valid
                    and not _saved
                ):
                    _set_prompt("Save before quitting? Y/N")
                    # Build and push the prompt state, then wait for response
                    state = _build_state_dict(
                        cal_manager, coverage, frame_score, auto_capture,
                        show_heatmap, show_undistort, fps,
                        cfg.board.aruco_dict, cal_manager.is_calibrating,
                        flash_text, flash_until, prompt_text, _stream_ended,
                    )
                    _, jpeg_buf = cv2.imencode(
                        ".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    with web_state.lock:
                        web_state.latest_jpeg = jpeg_buf.tobytes()
                        web_state.latest_state = state
                    # Wait for confirm/deny
                    while not web_state.shutdown_event.is_set():
                        try:
                            resp = web_state.action_queue.get(timeout=0.1)
                            resp_action = _action_from_string(resp)
                            if resp_action == Action.CONFIRM:
                                out_dir = Path(cfg.output.output_dir)
                                cal_manager.save_yaml(
                                    out_dir / "calibration.yaml",
                                    cfg.output.camera_name,
                                )
                                if cfg.output.save_observations:
                                    cal_manager.save_observations_npz(
                                        out_dir / "observations.npz"
                                    )
                                _clear_prompt()
                                break
                            elif resp_action == Action.DENY:
                                _clear_prompt()
                                break
                        except queue.Empty:
                            continue
                web_state.shutdown_event.set()
                break

            # Build state and encode frame
            state = _build_state_dict(
                cal_manager, coverage, frame_score, auto_capture,
                show_heatmap, show_undistort, fps,
                cfg.board.aruco_dict, cal_manager.is_calibrating,
                flash_text, flash_until, prompt_text, _stream_ended,
            )
            _, jpeg_buf = cv2.imencode(
                ".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            with web_state.lock:
                web_state.latest_jpeg = jpeg_buf.tobytes()
                web_state.latest_state = state

    finally:
        source.release()


def create_app(cfg: AppConfig) -> FastAPI:
    """Create the FastAPI application wired to the given config."""
    app = FastAPI(title="ChArUco Calibrator")
    web_state = WebState()

    # Serve static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_path = _STATIC_DIR / "index.html"
        return HTMLResponse(index_path.read_text())

    @app.websocket("/ws/video")
    async def ws_video(ws: WebSocket):
        await ws.accept()
        web_state.video_clients.add(ws)
        try:
            while not web_state.shutdown_event.is_set():
                with web_state.lock:
                    jpeg = web_state.latest_jpeg
                if jpeg:
                    await ws.send_bytes(jpeg)
                await asyncio.sleep(0.033)  # ~30 fps
        except WebSocketDisconnect:
            pass
        finally:
            web_state.video_clients.discard(ws)

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        web_state.state_clients.add(ws)
        try:
            while not web_state.shutdown_event.is_set():
                with web_state.lock:
                    state = web_state.latest_state
                if state:
                    await ws.send_text(json.dumps(state))
                await asyncio.sleep(0.1)  # ~10 Hz
        except WebSocketDisconnect:
            pass
        finally:
            web_state.state_clients.discard(ws)

    @app.websocket("/ws/action")
    async def ws_action(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                data = await ws.receive_text()
                web_state.action_queue.put(data)
        except WebSocketDisconnect:
            pass

    @app.on_event("startup")
    async def startup():
        loop_thread = threading.Thread(
            target=_run_calibration_loop,
            args=(cfg, web_state),
            daemon=True,
        )
        loop_thread.start()

    return app


def start_web_server(
    cfg: AppConfig, host: str = "0.0.0.0", port: int = 8080
) -> int:
    """Launch uvicorn with the FastAPI app. Blocks until shutdown."""
    import uvicorn

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0
