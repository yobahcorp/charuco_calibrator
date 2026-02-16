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
    ) -> np.ndarray:
        """Draw the status information panel at the top of the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        s = self._scale(frame)
        font_main = 0.55 * s
        font_sm = 0.45 * s
        thick = max(1, round(s))
        pad = int(10 * s)
        line_h = int(22 * s)
        bar_h = int(90 * s)

        # Semi-transparent background bar
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        y = int(20 * s)
        # Line 1: frame count + auto + heatmap status
        auto_str = "ON" if auto_capture else "OFF"
        hm_str = "ON" if show_heatmap else "OFF"
        line1 = f"Frames: {num_accepted}  |  Auto: {auto_str}  |  Heatmap: {hm_str}"
        cv2.putText(vis, line1, (pad, y), self.FONT, font_main, self.WHITE, thick)

        # Dictionary label — top-right corner
        if aruco_dict:
            text_size = cv2.getTextSize(aruco_dict, self.FONT, font_sm, thick)[0]
            tx = w - text_size[0] - pad
            cv2.putText(vis, aruco_dict, (tx, y), self.FONT, font_sm, self.CYAN, thick)

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
        banner_y = int(90 * s)
        banner_h = int(28 * s)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (0, 40, 80), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        cv2.putText(
            vis, self._prompt_text,
            (int(10 * s), banner_y + int(19 * s)), self.FONT, 0.48 * s, self.YELLOW, thick,
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

        cell_px = int(25 * s)
        grid_w = coverage.grid_cols * cell_px
        grid_h = coverage.grid_rows * cell_px
        margin = int(10 * s)

        # Quality bar occupies: 14px bar + 16px label above + 10px bottom margin = 40px
        quality_block_h = int(40 * s)
        ox = fw - grid_w - margin
        oy = fh - grid_h - margin - quality_block_h

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
            (ox, oy - int(5 * s)), self.FONT, 0.45 * s, self.WHITE, thick,
        )
        return vis

    def draw_quality_meter(
        self,
        frame: np.ndarray,
        quality: float,
    ) -> np.ndarray:
        """Draw a quality meter bar at the very bottom-right, below the coverage grid."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        s = self._scale(frame)
        thick = max(1, round(s))

        bar_w = int(120 * s)
        bar_h = int(14 * s)
        margin = int(10 * s)
        ox = fw - bar_w - margin
        oy = fh - bar_h - margin

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
            (ox, oy - int(6 * s)), self.FONT, 0.4 * s, self.WHITE, thick,
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

    def draw_help_hint(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard shortcut hints at the bottom-left."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        s = self._scale(frame)
        thick = max(1, round(s))
        hints = "SPACE:capture  A:auto  C:calibrate  R:reset  S:save  H:heatmap  Q:quit"
        cv2.putText(vis, hints, (int(10 * s), fh - int(10 * s)), self.FONT, 0.4 * s, self.WHITE, thick)
        return vis

    @staticmethod
    def poll_key(wait_ms: int = 1) -> Action:
        """Poll for a keyboard event and return the corresponding Action."""
        key = cv2.waitKey(wait_ms) & 0xFF
        return _KEY_MAP.get(key, Action.NONE)
