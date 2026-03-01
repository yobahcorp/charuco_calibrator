// ChArUco Calibrator — Web UI Client

(function () {
  "use strict";

  let wsVideo = null;
  let wsState = null;
  let wsAction = null;
  let flashTimeout = null;

  // ---- WebSocket connections ----

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const base = `${proto}://${location.host}`;

    // Video WebSocket (binary)
    wsVideo = new WebSocket(`${base}/ws/video`);
    wsVideo.binaryType = "arraybuffer";
    wsVideo.onmessage = onVideoFrame;
    wsVideo.onclose = function () { setTimeout(connect, 2000); };

    // State WebSocket (text JSON)
    wsState = new WebSocket(`${base}/ws/state`);
    wsState.onmessage = function (evt) {
      updateUI(JSON.parse(evt.data));
    };
    wsState.onclose = function () { setTimeout(connect, 2000); };

    // Action WebSocket (text, client -> server)
    wsAction = new WebSocket(`${base}/ws/action`);
    wsAction.onclose = function () { setTimeout(connect, 2000); };
  }

  // ---- Video frame handling ----

  const videoImg = () => document.getElementById("video-stream");

  function onVideoFrame(evt) {
    const blob = new Blob([evt.data], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    const img = videoImg();
    const oldUrl = img.src;
    img.src = url;
    if (oldUrl && oldUrl.startsWith("blob:")) {
      URL.revokeObjectURL(oldUrl);
    }
  }

  // ---- Send action to server ----

  function sendAction(actionStr) {
    if (wsAction && wsAction.readyState === WebSocket.OPEN) {
      wsAction.send(actionStr);
    }
  }

  // ---- UI update from state JSON ----

  function updateUI(state) {
    // Top bar stats
    setText("stat-frames", "Frames: " + state.num_frames);
    setText("stat-auto", "Auto: " + (state.auto_capture ? "ON" : "OFF"));
    setText("stat-fps", "FPS: " + state.fps.toFixed(1));
    setText("stat-dict", state.aruco_dict);

    // RMS with color
    var rmsEl = document.getElementById("stat-rms");
    if (state.rms !== null) {
      rmsEl.textContent = "RMS: " + state.rms.toFixed(3);
      rmsEl.className = state.rms < 1.0 ? "good" : state.rms < 2.0 ? "ok" : "bad";
    } else {
      rmsEl.textContent = "RMS: --";
      rmsEl.className = "";
    }

    // Calibrating indicator
    toggleHidden("stat-calibrating", !state.is_calibrating);

    // Toggle button active states
    toggleActive("btn-auto", state.auto_capture);
    toggleActive("btn-heatmap", state.show_heatmap);
    toggleActive("btn-undistort", state.show_undistort);

    // Coverage grid
    drawCoverageGrid(state.coverage_grid, state.grid_coverage_pct);

    // Quality meter
    updateQualityMeter(state.quality_meter);

    // Score breakdown
    updateScore(state.score);

    // Per-view errors
    drawErrorBars(state.per_view_errors);

    // Flash message
    if (state.flash && state.flash.active) {
      showFlash(state.flash.text);
    }

    // Prompt
    if (state.prompt) {
      showPrompt(state.prompt);
    } else {
      hidePrompt();
    }
  }

  // ---- Helper functions ----

  function setText(id, text) {
    document.getElementById(id).textContent = text;
  }

  function toggleHidden(id, hide) {
    document.getElementById(id).classList.toggle("hidden", hide);
  }

  function toggleActive(id, active) {
    var el = document.getElementById(id);
    if (el) el.classList.toggle("active", active);
  }

  // ---- Coverage grid (Canvas 2D) ----

  function drawCoverageGrid(grid, pct) {
    var canvas = document.getElementById("coverage-grid");
    var ctx = canvas.getContext("2d");
    if (!grid || grid.length === 0) return;

    var rows = grid.length;
    var cols = grid[0].length;
    var cellW = canvas.width / cols;
    var cellH = canvas.height / rows;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var val = grid[r][c];
        if (val > 0) {
          var fill = Math.min(val * 30, 200);
          ctx.fillStyle = "rgb(0, " + fill + ", 0)";
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 1;
        ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);
      }
    }

    setText("coverage-pct", pct.toFixed(0) + "%");
  }

  // ---- Quality meter ----

  function updateQualityMeter(quality) {
    var fill = document.getElementById("quality-fill");
    var pct = Math.round(quality * 100);
    fill.style.width = pct + "%";
    fill.className = quality >= 0.8 ? "" : quality >= 0.5 ? "ok" : "low";
    setText("quality-pct", pct + "%");
  }

  // ---- Score breakdown ----

  function updateScore(score) {
    if (!score) return;

    var rejEl = document.getElementById("score-rejected");

    if (score.rejected) {
      setText("score-corners", "--");
      setText("score-hull", "--");
      setText("score-newcov", "--");
      setText("score-blur", "--");
      setText("score-total", "--");
      rejEl.textContent = score.reject_reason;
      rejEl.classList.remove("hidden");
    } else {
      setText("score-corners", score.corner_ratio.toFixed(2));
      setText("score-hull", score.hull_spread.toFixed(2));
      setText("score-newcov", score.new_coverage.toFixed(2));
      setText("score-blur", score.blur_norm.toFixed(2));

      var totalEl = document.getElementById("score-total");
      totalEl.textContent = score.total.toFixed(2);
      totalEl.style.color = score.total >= 0.5 ? "#00cc66" : "#ffcc00";

      rejEl.classList.add("hidden");
    }
  }

  // ---- Per-view error bars (Canvas 2D) ----

  function drawErrorBars(errors) {
    var canvas = document.getElementById("error-bars");
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!errors || errors.length === 0) {
      ctx.fillStyle = "#666";
      ctx.font = "12px sans-serif";
      ctx.fillText("No calibration data", 10, canvas.height / 2);
      return;
    }

    var maxBars = Math.min(errors.length, 20);
    var barH = Math.floor(canvas.height / maxBars);
    var maxErr = Math.max.apply(null, errors);
    var sum = 0;
    for (var k = 0; k < errors.length; k++) sum += errors[k];
    var meanErr = sum / errors.length;
    var labelW = 45;
    var barAreaW = canvas.width - labelW;

    for (var i = 0; i < maxBars; i++) {
      var err = errors[i];
      var barW = (err / Math.max(maxErr, 0.001)) * barAreaW;
      var y = i * barH;

      // Color based on relationship to mean
      if (err > 2.0 * meanErr) {
        ctx.fillStyle = "#ff4444";
      } else if (err > 1.5 * meanErr) {
        ctx.fillStyle = "#ffaa00";
      } else {
        ctx.fillStyle = "#44cc44";
      }

      ctx.fillRect(0, y + 1, barW, barH - 2);

      // Error value label
      ctx.fillStyle = "#ccc";
      ctx.font = "10px monospace";
      ctx.fillText(err.toFixed(2), barW + 4, y + barH * 0.75);
    }
  }

  // ---- Flash message ----

  function showFlash(text) {
    var el = document.getElementById("flash-overlay");
    el.textContent = text;
    el.classList.remove("hidden", "fade-out");
    clearTimeout(flashTimeout);
    flashTimeout = setTimeout(function () {
      el.classList.add("fade-out");
      setTimeout(function () { el.classList.add("hidden"); }, 300);
    }, 500);
  }

  // ---- Prompt overlay ----

  function showPrompt(text) {
    setText("prompt-text", text);
    toggleHidden("prompt-overlay", false);
  }

  function hidePrompt() {
    toggleHidden("prompt-overlay", true);
  }

  // ---- Keyboard shortcuts ----

  document.addEventListener("keydown", function (evt) {
    if (evt.target.tagName === "INPUT" || evt.target.tagName === "TEXTAREA") return;

    var keyMap = {
      " ": "capture",
      "a": "toggle_auto",
      "c": "calibrate",
      "r": "reset",
      "s": "save",
      "h": "toggle_heatmap",
      "u": "undistort",
      "z": "undo",
      "y": "confirm",
      "n": "deny",
      "Escape": "quit"
    };

    var key = evt.key.length === 1 ? evt.key.toLowerCase() : evt.key;
    var action = keyMap[key];
    if (action) {
      evt.preventDefault();
      sendAction(action);
    }
  });

  // ---- Button click handlers (event delegation) ----

  document.addEventListener("click", function (evt) {
    var btn = evt.target.closest("button[data-action]");
    if (btn) {
      sendAction(btn.dataset.action);
    }
  });

  // ---- Initialize ----

  window.addEventListener("load", connect);
})();
