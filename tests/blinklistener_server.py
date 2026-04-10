"""
BlinkListener — Python Backend Server
======================================
A Flask REST API that controls the BlinkListener FMCW acoustic eye-blink
detector and streams live state to the Streamlit frontend.

Run with:
    python blinklistener_server.py

Endpoints:
    POST /start          — start detection with a JSON config body
    POST /stop           — stop detection
    POST /reset          — stop + clear all state
    GET  /status         — returns live metrics + signal history (JSON)
    GET  /devices        — list available audio devices

Dependencies:
    pip install flask numpy scipy pyaudio
"""

import threading
import queue
import time
from collections import deque
from typing import Optional

import numpy as np
import pyaudio
from flask import Flask, jsonify, request
from scipy import signal
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
SPEED_OF_SOUND = 343.0

DEFAULTS = dict(
    sample_rate    = 44100,
    f_start        = 18000,
    f_end          = 22000,
    chirp_duration = 0.04,
    face_dist_min  = 0.3,
    face_dist_max  = 0.9,
    threshold_mult = 2.5,
    refractory_s   = 0.35,
    calibration_s  = 3.0,
    smooth_window  = 3,
)


# ── chirp helper ──────────────────────────────────────────────────────────────

def make_chirp(f_start, f_end, duration, sample_rate):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    c = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method="linear")
    c *= np.hanning(len(c))
    return c.astype(np.float32)


# ── BlinkListener core ────────────────────────────────────────────────────────

class BlinkListener:
    """
    Active acoustic FMCW eye-blink detector.
    All mutable state is thread-safe for reads by the /status endpoint.
    """

    def __init__(self, cfg: dict):
        self.sr             = cfg["sample_rate"]
        self.f_start        = cfg["f_start"]
        self.f_end          = cfg["f_end"]
        self.chirp_dur      = cfg["chirp_duration"]
        self.chirp_n        = int(cfg["chirp_duration"] * cfg["sample_rate"])
        self.threshold_mult = cfg["threshold_mult"]
        self.refractory_s   = cfg["refractory_s"]
        self.calibration_s  = cfg["calibration_s"]
        self.smooth_window  = cfg["smooth_window"]
        self.face_dist_min  = cfg["face_dist_min"]
        self.face_dist_max  = cfg["face_dist_max"]

        self.tx_chirp     = make_chirp(self.f_start, self.f_end, self.chirp_dur, self.sr)
        self._range_bins  = self._compute_range_bins()
        self.face_bin: Optional[int] = None

        maxlen = 500
        self.amp_history    = deque(maxlen=maxlen)
        self.phase_history  = deque(maxlen=maxlen)
        self.deriv_history  = deque(maxlen=maxlen)
        self.thresh_history = deque(maxlen=maxlen)

        self.blink_count       = 0
        self.status_msg        = "idle"
        self._background_level = None

        refractory_chirps        = int(self.refractory_s / self.chirp_dur)
        self._chirps_since_blink = refractory_chirps
        self._calibration_n      = int(self.calibration_s / self.chirp_dur)
        self._chirp_index        = 0

        self.audio     = pyaudio.PyAudio()
        self.rx_queue  = queue.Queue()
        self._running  = False

        self._tx_buffer = np.tile(self.tx_chirp, 8)
        self._tx_pos    = 0
        self._tx_lock   = threading.Lock()
        self._rx_accum  = np.zeros(0, dtype=np.float32)

        self._lock = threading.Lock()   # protects public state reads

    # ── range helpers ──────────────────────────────────────────────────────

    def _compute_range_bins(self):
        bw = self.f_end - self.f_start
        k  = np.arange(self.chirp_n // 2 + 1)
        fb = k / self.chirp_dur
        return fb * SPEED_OF_SOUND * self.chirp_dur / (2 * bw)

    def _face_bin_range(self):
        d  = self._range_bins
        lo = int(np.searchsorted(d, self.face_dist_min))
        hi = int(np.searchsorted(d, self.face_dist_max))
        return max(lo, 1), min(hi, len(d) - 1)

    # ── FMCW processing ────────────────────────────────────────────────────

    def _process_chirp(self, rx_chunk):
        if len(rx_chunk) < self.chirp_n:
            return None
        rx   = rx_chunk[: self.chirp_n].astype(np.float32)
        beat = rx * self.tx_chirp
        spec = np.fft.rfft(beat, n=self.chirp_n)

        if self.face_bin is None and self._chirp_index >= self._calibration_n:
            lo, hi = self._face_bin_range()
            energy = np.abs(spec[lo:hi])
            with self._lock:
                self.face_bin = lo + int(np.argmax(energy))
                dist = (self._range_bins[self.face_bin]
                        if self.face_bin < len(self._range_bins) else 0)
                self.status_msg = f"face_locked:{dist:.2f}"

        if self.face_bin is None:
            return None
        return complex(spec[self.face_bin])

    def _detect_blink(self, face_signal) -> bool:
        amp   = abs(face_signal)
        phase = np.angle(face_signal)

        with self._lock:
            self.amp_history.append(amp)
            self.phase_history.append(phase)

            if len(self.amp_history) < 3:
                return False

            d_amp   = abs(self.amp_history[-1] - self.amp_history[-2])
            d_phase = abs(self.phase_history[-1] - self.phase_history[-2])
            if d_phase > np.pi:
                d_phase = abs(d_phase - 2 * np.pi)

            amp_mean = float(np.mean(self.amp_history))
            raw      = d_amp + amp_mean * 0.8 * d_phase

            self.deriv_history.append(raw)
            if len(self.deriv_history) >= self.smooth_window:
                feature = float(np.mean(list(self.deriv_history)[-self.smooth_window:]))
            else:
                feature = raw

            if self._background_level is None:
                self._background_level = feature
            elif feature < self._background_level * self.threshold_mult:
                self._background_level = 0.99 * self._background_level + 0.01 * feature

            threshold = self._background_level * self.threshold_mult
            self.thresh_history.append(threshold)

            refractory_chirps = int(self.refractory_s / self.chirp_dur)
            self._chirps_since_blink += 1
            if feature > threshold and self._chirps_since_blink >= refractory_chirps:
                self._chirps_since_blink = 0
                self.blink_count += 1
                return True
        return False

    # ── audio callbacks ────────────────────────────────────────────────────

    def _tx_callback(self, in_data, frame_count, time_info, status):
        rx = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.rx_queue.put(rx)

        with self._tx_lock:
            needed = frame_count
            out    = np.zeros(needed, dtype=np.float32)
            pos    = self._tx_pos % len(self._tx_buffer)
            end    = pos + needed
            if end <= len(self._tx_buffer):
                out[:] = self._tx_buffer[pos:end]
            else:
                first       = len(self._tx_buffer) - pos
                out[:first] = self._tx_buffer[pos:]
                out[first:] = self._tx_buffer[: needed - first]
            self._tx_pos = (self._tx_pos + needed) % len(self._tx_buffer)

        return ((out * 32767).astype(np.int16).tobytes(), pyaudio.paContinue)

    def _processing_thread(self):
        while self._running or not self.rx_queue.empty():
            try:
                chunk = self.rx_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._rx_accum = np.concatenate([self._rx_accum, chunk])
            while len(self._rx_accum) >= self.chirp_n:
                chirp_rx       = self._rx_accum[: self.chirp_n]
                self._rx_accum = self._rx_accum[self.chirp_n:]
                self._chirp_index += 1

                face_sig = self._process_chirp(chirp_rx)
                if face_sig is None:
                    continue
                self._detect_blink(face_sig)

    # ── public start / stop ────────────────────────────────────────────────

    def start(self, dev_index: Optional[int] = None):
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sr,
                input=True,
                output=True,
                frames_per_buffer=self.chirp_n,
                input_device_index=dev_index,
                output_device_index=dev_index,
                stream_callback=self._tx_callback,
            )
        except OSError:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sr,
                input=True,
                output=True,
                frames_per_buffer=self.chirp_n,
                stream_callback=self._tx_callback,
            )

        self._running   = True
        self.status_msg = f"calibrating:{self.calibration_s:.0f}"
        t = threading.Thread(target=self._processing_thread, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        with self._lock:
            self.status_msg = "stopped"

    # ── snapshot for /status ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of current state."""
        with self._lock:
            n = min(len(self.deriv_history), len(self.thresh_history),
                    len(self.amp_history),   len(self.phase_history))

            face_dist = None
            if self.face_bin is not None and self.face_bin < len(self._range_bins):
                face_dist = round(float(self._range_bins[self.face_bin]), 3)

            refractory_chirps = int(self.refractory_s / self.chirp_dur)
            peaks = []
            if n >= 5:
                deriv_arr  = np.array(list(self.deriv_history)[-n:])
                thresh_arr = np.array(list(self.thresh_history)[-n:])
                pk, _      = find_peaks(deriv_arr, height=thresh_arr,
                                        distance=refractory_chirps)
                peaks = pk.tolist()

            return {
                "running":        self._running,
                "status_msg":     self.status_msg,
                "blink_count":    self.blink_count,
                "chirp_index":    self._chirp_index,
                "face_bin":       self.face_bin,
                "face_dist_m":    face_dist,
                "chirp_dur":      self.chirp_dur,
                "history": {
                    "amp":    list(self.amp_history)[-n:],
                    "phase":  list(self.phase_history)[-n:],
                    "deriv":  list(self.deriv_history)[-n:],
                    "thresh": list(self.thresh_history)[-n:],
                    "blink_peaks": peaks,
                },
            }


# ── global listener instance (one session at a time) ─────────────────────────
_listener: Optional[BlinkListener] = None
_listener_lock = threading.Lock()


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/devices", methods=["GET"])
def devices():
    """List all audio devices."""
    p      = pyaudio.PyAudio()
    result = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        result.append({
            "index":    i,
            "name":     info["name"],
            "inputs":   int(info["maxInputChannels"]),
            "outputs":  int(info["maxOutputChannels"]),
        })
    p.terminate()
    return jsonify(result)


@app.route("/start", methods=["POST"])
def start():
    """
    Start detection.
    Body (all optional — falls back to DEFAULTS):
        {
          "f_start": 18000, "f_end": 22000, "chirp_duration": 0.04,
          "face_dist_min": 0.3, "face_dist_max": 0.9,
          "threshold_mult": 2.5, "refractory_s": 0.35,
          "calibration_s": 3.0, "smooth_window": 3,
          "device_index": null
        }
    """
    global _listener
    with _listener_lock:
        if _listener is not None and _listener._running:
            return jsonify({"error": "already running"}), 409

        body = request.get_json(silent=True) or {}
        cfg  = {**DEFAULTS, **{k: v for k, v in body.items() if k in DEFAULTS}}
        dev  = body.get("device_index", None)

        _listener = BlinkListener(cfg)
        _listener.start(dev_index=dev)

    return jsonify({"status": "started", "config": cfg})


@app.route("/stop", methods=["POST"])
def stop():
    """Stop detection."""
    global _listener
    with _listener_lock:
        if _listener is None or not _listener._running:
            return jsonify({"error": "not running"}), 409
        _listener.stop()
    return jsonify({"status": "stopped", "blink_count": _listener.blink_count})


@app.route("/reset", methods=["POST"])
def reset():
    """Stop detection and clear all state."""
    global _listener
    with _listener_lock:
        if _listener is not None and _listener._running:
            _listener.stop()
        _listener = None
    return jsonify({"status": "reset"})


@app.route("/status", methods=["GET"])
def status():
    """Return live metrics and signal history."""
    with _listener_lock:
        if _listener is None:
            return jsonify({"running": False, "status_msg": "idle",
                            "blink_count": 0, "chirp_index": 0,
                            "face_bin": None, "face_dist_m": None,
                            "history": {}})
        return jsonify(_listener.snapshot())


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  BlinkListener — Backend Server")
    print("=" * 50)
    print("  Listening on http://localhost:5000")
    print("  Endpoints: /start  /stop  /reset  /status  /devices")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
