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
    FONT_SCALE = 0.55
    THICKNESS = 1

    def __init__(self) -> None:
        self._flash_until: float = 0.0
        self._flash_text: str = ""

    def draw_status_panel(
        self,
        frame: np.ndarray,
        *,
        num_accepted: int,
        score: Optional[FrameScore],
        rms: Optional[float],
        auto_capture: bool,
        show_heatmap: bool,
    ) -> np.ndarray:
        """Draw the status information panel at the top of the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Semi-transparent background bar
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        y = 20
        # Line 1: frame count + auto + heatmap status
        auto_str = "ON" if auto_capture else "OFF"
        hm_str = "ON" if show_heatmap else "OFF"
        line1 = f"Frames: {num_accepted}  |  Auto: {auto_str}  |  Heatmap: {hm_str}"
        cv2.putText(vis, line1, (10, y), self.FONT, self.FONT_SCALE, self.WHITE, self.THICKNESS)

        # Line 2: current frame score
        y += 22
        if score is not None and not score.rejected:
            color = self.GREEN if score.total >= 0.5 else self.YELLOW
            line2 = f"Score: {score.total:.2f}  (corners={score.corner_ratio:.2f}  hull={score.hull_spread:.2f}  new={score.new_coverage:.2f}  blur={score.blur_norm:.2f})"
            cv2.putText(vis, line2, (10, y), self.FONT, 0.45, color, self.THICKNESS)
        elif score is not None and score.rejected:
            cv2.putText(vis, f"Rejected: {score.reject_reason}", (10, y), self.FONT, 0.45, self.RED, self.THICKNESS)

        # Line 3: RMS
        y += 22
        if rms is not None:
            rms_color = self.GREEN if rms < 1.0 else self.YELLOW if rms < 2.0 else self.RED
            cv2.putText(vis, f"RMS: {rms:.3f} px", (10, y), self.FONT, self.FONT_SCALE, rms_color, self.THICKNESS)
        else:
            cv2.putText(vis, "RMS: -- (calibrate with 'C')", (10, y), self.FONT, self.FONT_SCALE, self.WHITE, self.THICKNESS)

        return vis

    def draw_coverage_grid(
        self,
        frame: np.ndarray,
        coverage: CoverageState,
    ) -> np.ndarray:
        """Draw the coverage grid in the bottom-right corner."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]

        cell_px = 25
        grid_w = coverage.grid_cols * cell_px
        grid_h = coverage.grid_rows * cell_px
        margin = 10
        ox = fw - grid_w - margin
        oy = fh - grid_h - margin - 20  # space for quality bar

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
                cv2.rectangle(vis, (x0, y0), (x1, y1), self.WHITE, 1)

        # Label
        pct = coverage.grid_coverage * 100
        cv2.putText(
            vis, f"Coverage: {pct:.0f}%",
            (ox, oy - 5), self.FONT, 0.45, self.WHITE, 1,
        )
        return vis

    def draw_quality_meter(
        self,
        frame: np.ndarray,
        quality: float,
    ) -> np.ndarray:
        """Draw a quality meter bar at the bottom-right."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]

        bar_w = 120
        bar_h = 14
        margin = 10
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
        cv2.rectangle(vis, (ox, oy), (ox + bar_w, oy + bar_h), self.WHITE, 1)
        # Label
        cv2.putText(
            vis, f"Quality: {quality:.0%}",
            (ox, oy - 4), self.FONT, 0.4, self.WHITE, 1,
        )
        return vis

    def draw_accepted_flash(
        self, frame: np.ndarray, current_time: float
    ) -> np.ndarray:
        """Draw a brief 'ACCEPTED' flash overlay."""
        if current_time < self._flash_until:
            vis = frame.copy()
            h, w = vis.shape[:2]
            text = self._flash_text
            text_size = cv2.getTextSize(text, self.FONT, 1.2, 2)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2
            # Background rect
            pad = 15
            cv2.rectangle(
                vis,
                (tx - pad, ty - text_size[1] - pad),
                (tx + text_size[0] + pad, ty + pad),
                self.BLACK,
                -1,
            )
            cv2.putText(vis, text, (tx, ty), self.FONT, 1.2, self.GREEN, 2)
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
        hints = "SPACE:capture  A:auto  C:calibrate  R:reset  S:save  H:heatmap  Q:quit"
        cv2.putText(vis, hints, (10, fh - 10), self.FONT, 0.4, self.WHITE, 1)
        return vis

    @staticmethod
    def poll_key(wait_ms: int = 1) -> Action:
        """Poll for a keyboard event and return the corresponding Action."""
        key = cv2.waitKey(wait_ms) & 0xFF
        return _KEY_MAP.get(key, Action.NONE)
