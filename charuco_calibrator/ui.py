"""UI overlay rendering and keyboard dispatch."""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from .scoring import CoverageState, FrameScore


class Action(Enum):
    """User actions triggered by keyboard."""

    NONE = auto()
    CAPTURE = auto()
    TOGGLE_AUTO = auto()
    CALIBRATE = auto()
    RESET = auto()
    SAVE = auto()
    QUIT = auto()
    TOGGLE_HEATMAP = auto()
    CONFIRM = auto()
    DENY = auto()
    UNDO = auto()
    UNDISTORT = auto()


# Key mappings
_KEY_MAP = {
    ord(" "): Action.CAPTURE,
    ord("a"): Action.TOGGLE_AUTO,
    ord("A"): Action.TOGGLE_AUTO,
    ord("c"): Action.CALIBRATE,
    ord("C"): Action.CALIBRATE,
    ord("r"): Action.RESET,
    ord("R"): Action.RESET,
    ord("s"): Action.SAVE,
    ord("S"): Action.SAVE,
    ord("q"): Action.QUIT,
    ord("Q"): Action.QUIT,
    27: Action.QUIT,  # ESC
    ord("h"): Action.TOGGLE_HEATMAP,
    ord("H"): Action.TOGGLE_HEATMAP,
    ord("y"): Action.CONFIRM,
    ord("Y"): Action.CONFIRM,
    ord("n"): Action.DENY,
    ord("N"): Action.DENY,
    ord("z"): Action.UNDO,
    ord("Z"): Action.UNDO,
    ord("u"): Action.UNDISTORT,
    ord("U"): Action.UNDISTORT,
}


class UIRenderer:
    """Draws overlays, status panels, and handles keyboard input."""

    # Colors (BGR)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    CYAN = (255, 255, 0)
    ORANGE = (0, 165, 255)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    # Reference width the UI was originally designed for
    _REF_WIDTH = 640

    @staticmethod
    def _scale(frame: np.ndarray) -> float:
        """Compute UI scale factor based on frame width."""
        return frame.shape[1] / UIRenderer._REF_WIDTH

    def __init__(self) -> None:
        self._flash_until: float = 0.0
        self._flash_text: str = ""
        self._prompt_text: str = ""
        self._border_flash_until: float = 0.0

    # Height of the status panel (in scaled pixels) — used by other overlays to avoid overlap
    # y starts at 16, 3 lines at 18px spacing → last baseline at 52, +8 bottom pad = 60
    @staticmethod
    def _panel_height(s: float) -> int:
        return int(60 * s)

    def draw_status_panel(
        self,
        frame: np.ndarray,
        *,
        num_accepted: int,
        score: Optional[FrameScore],
        rms: Optional[float],
        auto_capture: bool,
        show_heatmap: bool,
        aruco_dict: str = "",
        fps: float = 0.0,
        show_undistort: bool = False,
    ) -> np.ndarray:
        """Draw the status information panel at the top of the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        s = self._scale(frame)
        font_main = 0.48 * s
        font_sm = 0.40 * s
        thick = max(1, round(s))
        pad = int(8 * s)
        line_h = int(18 * s)
        bar_h = self._panel_height(s)

        # Semi-transparent background bar
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        y = int(16 * s)
        # Line 1: frame count + auto + heatmap + undistort
        auto_str = "ON" if auto_capture else "OFF"
        hm_str = "ON" if show_heatmap else "OFF"
        ud_str = "  |  Undistort: ON" if show_undistort else ""
        line1 = f"Frames: {num_accepted}  |  Auto: {auto_str}  |  Heatmap: {hm_str}{ud_str}"
        cv2.putText(vis, line1, (pad, y), self.FONT, font_main, self.WHITE, thick)

        # FPS + dictionary label — top-right corner
        right_label = f"FPS: {fps:.1f}"
        if aruco_dict:
            right_label += f"  |  {aruco_dict}"
        text_size = cv2.getTextSize(right_label, self.FONT, font_sm, thick)[0]
        tx = w - text_size[0] - pad
        cv2.putText(vis, right_label, (tx, y), self.FONT, font_sm, self.CYAN, thick)

        # Line 2: current frame score
        y += line_h
        if score is not None and not score.rejected:
            color = self.GREEN if score.total >= 0.5 else self.YELLOW
            line2 = f"Score: {score.total:.2f}  (corners={score.corner_ratio:.2f}  hull={score.hull_spread:.2f}  new={score.new_coverage:.2f}  blur={score.blur_norm:.2f})"
            cv2.putText(vis, line2, (pad, y), self.FONT, font_sm, color, thick)
        elif score is not None and score.rejected:
            cv2.putText(vis, f"Rejected: {score.reject_reason}", (pad, y), self.FONT, font_sm, self.RED, thick)

        # Line 3: RMS
        y += line_h
        if rms is not None:
            rms_color = self.GREEN if rms < 1.0 else self.YELLOW if rms < 2.0 else self.RED
            cv2.putText(vis, f"RMS: {rms:.3f} px", (pad, y), self.FONT, font_main, rms_color, thick)
        else:
            cv2.putText(vis, "RMS: -- (calibrate with 'C')", (pad, y), self.FONT, font_main, self.WHITE, thick)

        return vis

    def set_prompt(self, text: str) -> None:
        """Set a persistent prompt message."""
        self._prompt_text = text

    def clear_prompt(self) -> None:
        """Clear the prompt message."""
        self._prompt_text = ""

    def draw_prompt(self, frame: np.ndarray) -> np.ndarray:
        """Draw a yellow prompt banner below the status panel."""
        if not self._prompt_text:
            return frame
        vis = frame.copy()
        h, w = vis.shape[:2]
        s = self._scale(frame)
        thick = max(1, round(s))
        banner_y = self._panel_height(s)
        banner_h = int(22 * s)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (0, 40, 80), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        cv2.putText(
            vis, self._prompt_text,
            (int(8 * s), banner_y + int(15 * s)), self.FONT, 0.4 * s, self.YELLOW, thick,
        )
        return vis

    def draw_coverage_grid(
        self,
        frame: np.ndarray,
        coverage: CoverageState,
    ) -> np.ndarray:
        """Draw the coverage grid in the bottom-right corner, above the quality bar."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        s = self._scale(frame)
        thick = max(1, round(s))

        cell_px = int(20 * s)
        grid_w = coverage.grid_cols * cell_px
        grid_h = coverage.grid_rows * cell_px
        margin = int(8 * s)

        # Quality bar block: 12px bar + 14px label + 8px gap = ~34px above hint bar
        hint_h = self._hint_bar_height(s)
        quality_block_h = int(34 * s)
        ox = fw - grid_w - margin
        oy = fh - grid_h - margin - quality_block_h - hint_h

        for r in range(coverage.grid_rows):
            for c in range(coverage.grid_cols):
                x0 = ox + c * cell_px
                y0 = oy + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                if coverage.grid[r, c] > 0:
                    fill = min(coverage.grid[r, c] * 30, 200)
                    color = (0, int(fill), 0)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(vis, (x0, y0), (x1, y1), self.WHITE, thick)

        # Label above grid
        pct = coverage.grid_coverage * 100
        cv2.putText(
            vis, f"Coverage: {pct:.0f}%",
            (ox, oy - int(4 * s)), self.FONT, 0.38 * s, self.WHITE, thick,
        )
        return vis

    def draw_guidance_arrow(
        self,
        frame: np.ndarray,
        coverage: CoverageState,
    ) -> np.ndarray:
        """Draw an arrow pointing toward the region with least coverage."""
        # Find empty cells; if none, skip
        empty = np.argwhere(coverage.grid == 0)
        if len(empty) == 0:
            return frame

        vis = frame.copy()
        fh, fw = vis.shape[:2]
        s = self._scale(frame)

        # Compute centroid of empty cells in normalized coords
        mean_row = float(np.mean(empty[:, 0]) + 0.5) / coverage.grid_rows
        mean_col = float(np.mean(empty[:, 1]) + 0.5) / coverage.grid_cols

        # Target point in frame coords
        tx = int(mean_col * fw)
        ty = int(mean_row * fh)

        # Arrow from frame center toward the target
        cx, cy = fw // 2, fh // 2
        dx, dy = tx - cx, ty - cy
        dist = max(np.sqrt(dx * dx + dy * dy), 1.0)

        # Arrow length proportional to frame, start from center
        arrow_len = int(60 * s)
        end_x = cx + int(dx / dist * arrow_len)
        end_y = cy + int(dy / dist * arrow_len)

        thick = max(2, int(3 * s))
        tip_len = int(15 * s)
        cv2.arrowedLine(
            vis, (cx, cy), (end_x, end_y),
            self.YELLOW, thick, tipLength=tip_len / max(arrow_len, 1),
        )

        return vis

    def draw_quality_meter(
        self,
        frame: np.ndarray,
        quality: float,
    ) -> np.ndarray:
        """Draw a quality meter bar at the bottom-right, above the hint bar."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        s = self._scale(frame)
        thick = max(1, round(s))

        bar_w = int(100 * s)
        bar_h = int(12 * s)
        margin = int(8 * s)
        hint_h = self._hint_bar_height(s)
        ox = fw - bar_w - margin
        oy = fh - bar_h - margin - hint_h

        # Background
        cv2.rectangle(vis, (ox, oy), (ox + bar_w, oy + bar_h), self.BLACK, -1)
        # Fill
        fill_w = int(bar_w * min(quality, 1.0))
        if quality >= 0.8:
            color = self.GREEN
        elif quality >= 0.5:
            color = self.YELLOW
        else:
            color = self.ORANGE
        cv2.rectangle(vis, (ox, oy), (ox + fill_w, oy + bar_h), color, -1)
        # Border
        cv2.rectangle(vis, (ox, oy), (ox + bar_w, oy + bar_h), self.WHITE, thick)
        # Label above bar
        cv2.putText(
            vis, f"Quality: {quality:.0%}",
            (ox, oy - int(5 * s)), self.FONT, 0.35 * s, self.WHITE, thick,
        )
        return vis

    def draw_accepted_flash(
        self, frame: np.ndarray, current_time: float
    ) -> np.ndarray:
        """Draw a brief 'ACCEPTED' flash overlay."""
        if current_time < self._flash_until:
            vis = frame.copy()
            h, w = vis.shape[:2]
            s = self._scale(frame)
            thick = max(2, round(2 * s))
            font_scale = 1.2 * s
            text = self._flash_text
            text_size = cv2.getTextSize(text, self.FONT, font_scale, thick)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2
            # Background rect
            pad = int(15 * s)
            cv2.rectangle(
                vis,
                (tx - pad, ty - text_size[1] - pad),
                (tx + text_size[0] + pad, ty + pad),
                self.BLACK,
                -1,
            )
            cv2.putText(vis, text, (tx, ty), self.FONT, font_scale, self.GREEN, thick)
            return vis
        return frame

    def trigger_flash(self, text: str, current_time: float, duration: float = 0.7) -> None:
        """Trigger a flash message."""
        self._flash_text = text
        self._flash_until = current_time + duration

    def draw_per_view_errors(
        self,
        frame: np.ndarray,
        errors: np.ndarray,
    ) -> np.ndarray:
        """Draw a small per-view error bar chart on the left side."""
        if len(errors) == 0:
            return frame
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        s = self._scale(frame)
        thick = max(1, round(s))
        pad = int(8 * s)

        max_bars = min(len(errors), 20)
        bar_h = int(6 * s)
        gap = int(2 * s)
        max_bar_w = int(80 * s)
        chart_h = max_bars * (bar_h + gap)

        # Position: left side, below status panel with gap
        ox = pad
        oy = self._panel_height(s) + int(18 * s)

        # Clamp: don't let bars extend past the help hint bar at the bottom
        hint_bar_h = int(24 * s)
        max_chart_bottom = fh - hint_bar_h - pad
        if oy + chart_h > max_chart_bottom:
            avail = max_chart_bottom - oy
            max_bars = max(1, avail // (bar_h + gap))
            chart_h = max_bars * (bar_h + gap)

        # Background
        overlay = vis.copy()
        cv2.rectangle(
            overlay,
            (ox - 2, oy - int(14 * s)),
            (ox + max_bar_w + int(35 * s), oy + chart_h + 2),
            self.BLACK, -1,
        )
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

        # Label
        cv2.putText(
            vis, "Per-view error",
            (ox, oy - int(3 * s)), self.FONT, 0.3 * s, self.WHITE, thick,
        )

        max_err = float(np.max(errors)) if len(errors) > 0 else 1.0
        mean_err = float(np.mean(errors))

        for i in range(max_bars):
            err = float(errors[i])
            bar_w = int((err / max(max_err, 1e-6)) * max_bar_w)
            y0 = oy + i * (bar_h + gap)

            if err > 2.0 * mean_err:
                color = self.RED
            elif err > 1.5 * mean_err:
                color = self.YELLOW
            else:
                color = self.GREEN

            cv2.rectangle(vis, (ox, y0), (ox + bar_w, y0 + bar_h), color, -1)
            cv2.putText(
                vis, f"{err:.2f}",
                (ox + max_bar_w + int(3 * s), y0 + bar_h),
                self.FONT, 0.25 * s, self.WHITE, thick,
            )

        return vis

    def trigger_border_flash(self, current_time: float, duration: float = 0.2) -> None:
        """Trigger a green border flash on frame capture."""
        self._border_flash_until = current_time + duration

    def draw_border_flash(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Draw a bright green border around the frame on capture."""
        if current_time >= self._border_flash_until:
            return frame
        vis = frame.copy()
        h, w = vis.shape[:2]
        s = self._scale(frame)
        border = max(4, int(6 * s))
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), self.GREEN, border)
        return vis

    def draw_calibrating_indicator(
        self, frame: np.ndarray, current_time: float
    ) -> np.ndarray:
        """Draw a pulsing 'Calibrating...' indicator at the center of the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        s = self._scale(frame)
        thick = max(2, round(2 * s))
        font_scale = 0.9 * s

        # Animated dots: cycle through 1-3 dots
        n_dots = int(current_time * 3) % 3 + 1
        text = "Calibrating" + "." * n_dots

        text_size = cv2.getTextSize(text, self.FONT, font_scale, thick)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2 + int(60 * s)

        # Pulsing alpha via sine wave
        alpha = 0.5 + 0.3 * np.sin(current_time * 4)

        pad = int(12 * s)
        overlay = vis.copy()
        cv2.rectangle(
            overlay,
            (tx - pad, ty - text_size[1] - pad),
            (tx + text_size[0] + pad, ty + pad),
            self.BLACK,
            -1,
        )
        cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)
        cv2.putText(vis, text, (tx, ty), self.FONT, font_scale, self.CYAN, thick)
        return vis

    # Height of the bottom hint bar (in scaled pixels)
    @staticmethod
    def _hint_bar_height(s: float) -> int:
        return int(24 * s)

    def draw_help_hint(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard shortcut hints on a dark strip at the bottom."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        s = self._scale(frame)
        thick = max(1, round(s))
        bar_h = self._hint_bar_height(s)
        pad = int(8 * s)

        # Semi-transparent dark background strip
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, fh - bar_h), (fw, fh), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        hints = "SPACE:capture  A:auto  C:calibrate  R:reset  S:save  H:heatmap  U:undistort  Z:undo  Q:quit"
        cv2.putText(vis, hints, (pad, fh - int(8 * s)), self.FONT, 0.35 * s, self.WHITE, thick)
        return vis

    @staticmethod
    def poll_key(wait_ms: int = 1) -> Action:
        """Poll for a keyboard event and return the corresponding Action."""
        key = cv2.waitKey(wait_ms) & 0xFF
        return _KEY_MAP.get(key, Action.NONE)
