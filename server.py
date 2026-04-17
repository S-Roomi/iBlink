"""
Bank Robbery Alert — FastAPI Server
====================================
Exposes the acoustic FMCW eye-blink detector as a REST + WebSocket API so
any front-end (TypeScript, React, etc.) can drive it.

Endpoints
---------
REST
  GET  /api/devices                  List available duplex audio devices
  POST /api/session/start            Create & start a detection session
  POST /api/session/stop             Stop the current session
  POST /api/session/dismiss          Dismiss a triggered alert
  GET  /api/session/status           Current snapshot (poll-friendly)
  GET  /api/session/chart            PNG chart of the last N samples

WebSocket
  WS   /ws/live                      Push status frames ~every 250 ms

Run with:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Dependencies:
  pip install fastapi uvicorn[standard] numpy scipy pyaudio matplotlib
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
import warnings
from collections import deque
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from scipy import signal
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bank Robbery Alert API",
    description="Acoustic FMCW eye-blink silent duress system",
    version="1.0.0",
)

# Allow all origins in development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global session holder ─────────────────────────────────────────────────────
# One session at a time.  The TypeScript front-end starts/stops it.
_session: Optional["BlinkListener"] = None
_session_lock = threading.Lock()

# ── Constants ─────────────────────────────────────────────────────────────────

SPEED_OF_SOUND = 343.0

DEFAULTS = dict(
    sample_rate     = 44100,
    f_start         = 18000,
    f_end           = 22000,
    chirp_duration  = 0.04,      # seconds
    face_dist_min   = 0.3,       # metres
    face_dist_max   = 0.9,
    threshold_mult  = 2.8,
    refractory_s    = 0.35,
    calibration_s   = 3.0,
    smooth_window   = 3,
    alert_window_s  = 5,
    alert_threshold = 5,
    sensing_delay_s = 2.0,
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SessionConfig(BaseModel):
    sample_rate:     int   = Field(DEFAULTS["sample_rate"],     ge=8000, le=96000)
    f_start:         int   = Field(DEFAULTS["f_start"],         ge=16000, le=20000)
    f_end:           int   = Field(DEFAULTS["f_end"],           ge=16500, le=22050)
    chirp_duration:  float = Field(DEFAULTS["chirp_duration"],  ge=0.01, le=0.1)
    face_dist_min:   float = Field(DEFAULTS["face_dist_min"],   ge=0.1, le=1.0)
    face_dist_max:   float = Field(DEFAULTS["face_dist_max"],   ge=0.2, le=2.0)
    threshold_mult:  float = Field(DEFAULTS["threshold_mult"],  ge=1.0, le=6.0)
    refractory_s:    float = Field(DEFAULTS["refractory_s"],    ge=0.1, le=1.0)
    calibration_s:   float = Field(DEFAULTS["calibration_s"],   ge=1.0, le=10.0)
    smooth_window:   int   = Field(DEFAULTS["smooth_window"],   ge=1, le=10)
    alert_window_s:  int   = Field(DEFAULTS["alert_window_s"],  ge=1, le=30)
    alert_threshold: int   = Field(DEFAULTS["alert_threshold"], ge=1, le=20)
    sensing_delay_s: float = Field(DEFAULTS["sensing_delay_s"], ge=0.0, le=10.0)
    device_index:    Optional[int] = None   # None → auto-select


class StartResponse(BaseModel):
    ok:      bool
    message: str


class StatusResponse(BaseModel):
    running:            bool
    status_msg:         str
    blink_count:        int
    chirps_processed:   int
    face_bin:           Optional[int]
    face_distance_m:    Optional[float]
    blinks_in_window:   int
    alert_threshold:    int
    alert_window_s:     int
    seconds_until_reset: float
    alert_triggered:    bool
    alert_time:         Optional[float]


class DeviceInfo(BaseModel):
    index: int
    name:  str
    input_channels:  int
    output_channels: int
    is_duplex: bool


# ── Audio helpers ─────────────────────────────────────────────────────────────

def make_chirp(f_start: int, f_end: int, duration: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    c = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method="linear")
    c *= np.hanning(len(c))
    return c.astype(np.float32)


def list_audio_devices() -> list[DeviceInfo]:
    pa = pyaudio.PyAudio()
    devices: list[DeviceInfo] = []
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            devices.append(DeviceInfo(
                index           = i,
                name            = info.get("name", f"Device {i}"),
                input_channels  = int(info.get("maxInputChannels",  0)),
                output_channels = int(info.get("maxOutputChannels", 0)),
                is_duplex       = (info.get("maxInputChannels",  0) > 0 and
                                   info.get("maxOutputChannels", 0) > 0),
            ))
        except Exception:
            pass
    pa.terminate()
    return devices


# ── Core BlinkListener (algorithm unchanged from original) ────────────────────

class BlinkListener:
    """
    Acoustic FMCW blink detector — identical signal pipeline to the original
    Streamlit version, stripped of all UI code.

    Pipeline (Liu et al. IMWUT 2021):
      1. TX continuous ultrasonic FMCW chirp via speaker
      2. RX via microphone
      3. Beat = TX × RX   (de-chirp)
      4. Range FFT → complex spectrum; pick face bin
      5. Static clutter removal (calibration mean)
      6. Extract unwrapped phase per chirp
      7. Breathing cancellation (2nd-order HP Butterworth, >1 Hz)
      8. |Δphase| feature + smoothing
      9. Adaptive EMA threshold
     10. Refractory gate → blink event
     11. Sliding-window alert logic
    """

    BREATH_CUTOFF_HZ = 1.0
    NOISE_GATE_MULT = 2.8
    MIN_DERIV_GATE = 0.005

    def __init__(self, cfg: SessionConfig):
        self.sr             = cfg.sample_rate
        self.f_start        = cfg.f_start
        self.f_end          = cfg.f_end
        self.chirp_dur      = cfg.chirp_duration
        self.chirp_n        = int(cfg.chirp_duration * cfg.sample_rate)
        self.threshold_mult = cfg.threshold_mult
        self.refractory_s   = cfg.refractory_s
        self.calibration_s  = cfg.calibration_s
        self.smooth_window  = cfg.smooth_window
        self.face_dist_min  = cfg.face_dist_min
        self.face_dist_max  = cfg.face_dist_max
        self.alert_window_s  = cfg.alert_window_s
        self.alert_threshold = cfg.alert_threshold
        self.sensing_delay_s = cfg.sensing_delay_s

        self.tx_chirp    = make_chirp(self.f_start, self.f_end, self.chirp_dur, self.sr)
        self._range_bins = self._compute_range_bins()
        self.face_bin: Optional[int] = None

        self._calib_spectra: list = []
        self._static_clutter: Optional[np.ndarray] = None
        self._calibration_n = int(self.calibration_s / self.chirp_dur)

        maxlen = 500
        self._phase_raw: deque = deque(maxlen=maxlen)
        self._unwrap_accum: float = 0.0

        self.amp_history    = deque(maxlen=maxlen)
        self.phase_history  = deque(maxlen=maxlen)
        self.deriv_history  = deque(maxlen=maxlen)
        self.thresh_history = deque(maxlen=maxlen)

        self._bg_level: Optional[float] = None
        self._bg_alpha = 0.02
        self._above_threshold = False
        self._peak_feature = 0.0

        self.blink_count = 0
        self.status_msg  = "Idle"
        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        self._chirps_since_blink = refractory_chirps
        self._chirp_index        = 0

        self.audio      = pyaudio.PyAudio()
        self.rx_queue   = queue.Queue()
        self._running   = False
        self._tx_buffer = np.tile(self.tx_chirp, 8)
        self._tx_pos    = 0
        self._tx_lock   = threading.Lock()
        self._rx_accum  = np.zeros(0, dtype=np.float32)

        self._blink_times: deque = deque()
        self.alert_triggered     = False
        self.alert_time: Optional[float] = None
        self._sensing_enabled    = False
        self._start_wall_time: Optional[float] = None
        self.blinks_in_window: int = 0

    # ── range helpers ─────────────────────────────────────────────────────────

    def _compute_range_bins(self) -> np.ndarray:
        bw = self.f_end - self.f_start
        n  = self.chirp_n
        k  = np.arange(n // 2 + 1)
        fb = k / self.chirp_dur
        return fb * SPEED_OF_SOUND * self.chirp_dur / (2 * bw)

    def _face_bin_range(self):
        d  = self._range_bins
        lo = int(np.searchsorted(d, self.face_dist_min))
        hi = int(np.searchsorted(d, self.face_dist_max))
        return max(lo, 1), min(hi, len(d) - 1)

    # ── FMCW de-chirp + clutter removal ──────────────────────────────────────

    def _process_chirp(self, rx_chunk: np.ndarray) -> Optional[complex]:
        if len(rx_chunk) < self.chirp_n:
            return None

        rx   = rx_chunk[: self.chirp_n].astype(np.float32)
        beat = rx * self.tx_chirp
        spec = np.fft.rfft(beat, n=self.chirp_n)

        if self._static_clutter is None:
            self._calib_spectra.append(spec.copy())
            if len(self._calib_spectra) >= self._calibration_n:
                self._static_clutter = np.mean(self._calib_spectra, axis=0)
                lo, hi   = self._face_bin_range()
                clean    = spec - self._static_clutter
                energy   = np.abs(clean[lo:hi])
                self.face_bin = lo + int(np.argmax(energy))
                dist = (self._range_bins[self.face_bin]
                        if self.face_bin < len(self._range_bins) else 0)
                self.status_msg = f"Face locked at {dist:.2f} m — monitoring…"
            return None

        clean = spec - self._static_clutter
        return complex(clean[self.face_bin])

    # ── phase unwrapping ──────────────────────────────────────────────────────

    def _unwrap_push(self, wrapped_phase: float) -> float:
        if not self._phase_raw:
            self._unwrap_accum = wrapped_phase
            return wrapped_phase
        prev_wrapped = np.angle(np.exp(1j * self._phase_raw[-1]))
        diff = wrapped_phase - prev_wrapped
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        self._unwrap_accum += diff
        return self._unwrap_accum

    # ── breathing cancellation ────────────────────────────────────────────────

    def _remove_breathing(self, phase_seq: np.ndarray) -> np.ndarray:
        chirp_rate  = 1.0 / self.chirp_dur
        nyq         = chirp_rate / 2.0
        cutoff_norm = self.BREATH_CUTOFF_HZ / nyq
        min_len = 18
        if len(phase_seq) < min_len or cutoff_norm >= 1.0:
            return phase_seq - np.linspace(phase_seq[0], phase_seq[-1], len(phase_seq))
        try:
            b, a    = signal.butter(2, cutoff_norm, btype="high")
            cleaned = signal.filtfilt(b, a, phase_seq)
        except Exception:
            cleaned = phase_seq - np.linspace(phase_seq[0], phase_seq[-1], len(phase_seq))
        return cleaned

    # ── blink detection ───────────────────────────────────────────────────────

    def _detect_blink(self, face_phasor: complex) -> bool:
        amp          = abs(face_phasor)
        wrapped_ph   = np.angle(face_phasor)
        unwrapped_ph = self._unwrap_push(wrapped_ph)

        self.amp_history.append(amp)
        self.phase_history.append(unwrapped_ph)
        self._phase_raw.append(unwrapped_ph)

        if len(self._phase_raw) < 4:
            return False

        phase_arr = np.array(self._phase_raw)
        clean_ph  = self._remove_breathing(phase_arr)
        d_phase   = abs(clean_ph[-1] - clean_ph[-2])
        self.deriv_history.append(d_phase)

        if len(self.deriv_history) >= self.smooth_window:
            feature = float(np.mean(list(self.deriv_history)[-self.smooth_window:]))
        else:
            feature = d_phase

        if self._bg_level is None:
            self._bg_level = feature
        else:
            threshold_now = self._bg_level * self.threshold_mult
            if feature < threshold_now:
                self._bg_level = (
                    (1.0 - self._bg_alpha) * self._bg_level
                    + self._bg_alpha * feature
                )

        threshold = self._bg_level * self.threshold_mult

        # Robustly gate against idle noise using median + k * MAD from
        # recent derivative history. This significantly reduces false blinks
        # when user is still.
        recent = np.array(list(self.deriv_history)[-30:], dtype=np.float32)
        noise_gate = 0.0
        if len(recent) >= 10:
            med = float(np.median(recent))
            mad = float(np.median(np.abs(recent - med)))
            robust_sigma = 1.4826 * mad
            noise_gate = med + self.NOISE_GATE_MULT * robust_sigma

        detection_gate = max(threshold, noise_gate, self.MIN_DERIV_GATE)
        self.thresh_history.append(detection_gate)

        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        self._chirps_since_blink += 1

        # Count one blink per above-threshold excursion to avoid overcounting
        # when the feature stays elevated across consecutive chirps.
        if feature > detection_gate:
            self._above_threshold = True
            self._peak_feature = max(self._peak_feature, feature)
            return False

        if self._above_threshold:
            self._above_threshold = False
            peak_feature = self._peak_feature
            self._peak_feature = 0.0
            if peak_feature > detection_gate and self._chirps_since_blink >= refractory_chirps:
                self._chirps_since_blink = 0
                return True
        return False

    # ── alert logic ───────────────────────────────────────────────────────────

    def _check_alert(self):
        now    = time.time()
        cutoff = now - self.alert_window_s
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        self.blinks_in_window = len(self._blink_times)

        if self.blinks_in_window >= self.alert_threshold and not self.alert_triggered:
            self.alert_triggered = True
            self.alert_time      = now
            self.status_msg      = "ROBBERY ALERT TRIGGERED"
            # Reset the blink window immediately after threshold hit so
            # subsequent counting starts fresh.
            self._blink_times.clear()
            self.blinks_in_window = 0

    def seconds_until_window_reset(self) -> float:
        if not self._blink_times:
            return float(self.alert_window_s)
        oldest    = self._blink_times[0]
        expires   = oldest + self.alert_window_s
        remaining = expires - time.time()
        return max(0.0, remaining)

    # ── audio callbacks ───────────────────────────────────────────────────────

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
                first        = len(self._tx_buffer) - pos
                out[:first]  = self._tx_buffer[pos:]
                out[first:]  = self._tx_buffer[: needed - first]
            self._tx_pos = (self._tx_pos + needed) % len(self._tx_buffer)

        return ((out * 32767).astype(np.int16).tobytes(), pyaudio.paContinue)

    def _processing_thread(self):
        self._start_wall_time = time.time()
        self._sensing_enabled = False
        self.status_msg = f"Starting up — sensing begins in {self.sensing_delay_s:.0f}s…"

        while self._running or not self.rx_queue.empty():
            if not self._sensing_enabled:
                elapsed = time.time() - self._start_wall_time
                if elapsed >= self.sensing_delay_s:
                    self._sensing_enabled = True
                    # Re-baseline detector at sensing start so startup noise
                    # does not contribute to blink counting.
                    self._phase_raw.clear()
                    self.deriv_history.clear()
                    self.thresh_history.clear()
                    self._bg_level = None
                    self._above_threshold = False
                    self._peak_feature = 0.0
                    if self._static_clutter is None:
                        self.status_msg = (
                            f"Calibrating for {self.calibration_s:.0f}s — sit still…"
                        )

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

                is_blink = self._detect_blink(face_sig)

                if is_blink and self._sensing_enabled:
                    self.blink_count += 1
                    self._blink_times.append(time.time())
                    self._check_alert()

    # ── public interface ──────────────────────────────────────────────────────

    def start(self, device_index: Optional[int] = None):
        kwargs = dict(
            format            = pyaudio.paInt16,
            channels          = 1,
            rate              = self.sr,
            input             = True,
            output            = True,
            frames_per_buffer = self.chirp_n,
            stream_callback   = self._tx_callback,
        )
        if device_index is not None:
            kwargs["input_device_index"]  = device_index
            kwargs["output_device_index"] = device_index

        try:
            self.stream = self.audio.open(**kwargs)
        except OSError:
            # Re-try without specifying device (let OS choose)
            kwargs.pop("input_device_index",  None)
            kwargs.pop("output_device_index", None)
            self.stream = self.audio.open(**kwargs)

        self._running = True
        t = threading.Thread(target=self._processing_thread, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.status_msg = "Stopped"

    def dismiss_alert(self):
        self.alert_triggered  = False
        self.alert_time       = None
        self._blink_times.clear()
        self.blinks_in_window = 0

    def snapshot(self) -> StatusResponse:
        dist: Optional[float] = None
        if self.face_bin is not None and self.face_bin < len(self._range_bins):
            dist = float(self._range_bins[self.face_bin])

        return StatusResponse(
            running            = self._running,
            status_msg         = self.status_msg,
            blink_count        = self.blink_count,
            chirps_processed   = self._chirp_index,
            face_bin           = self.face_bin,
            face_distance_m    = dist,
            blinks_in_window   = self.blinks_in_window,
            alert_threshold    = self.alert_threshold,
            alert_window_s     = self.alert_window_s,
            seconds_until_reset = self.seconds_until_window_reset(),
            alert_triggered    = self.alert_triggered,
            alert_time         = self.alert_time,
        )

    def build_png_chart(self, dpi: int = 100) -> Optional[bytes]:
        """Render the 3-panel diagnostic chart as PNG bytes."""
        n = min(
            len(self.deriv_history),
            len(self.thresh_history),
            len(self.amp_history),
            len(self.phase_history),
        )
        if n < 5:
            return None

        deriv  = list(self.deriv_history)[-n:]
        thresh = list(self.thresh_history)[-n:]
        amps   = list(self.amp_history)[-n:]
        phases = np.array(list(self.phase_history)[-n:])
        t      = np.arange(n) * self.chirp_dur

        clean_phases = self._remove_breathing(phases)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig.patch.set_facecolor("#0a0c10")
        for ax in axes:
            ax.set_facecolor("#111620")
            ax.tick_params(colors="#aabbcc")
            ax.yaxis.label.set_color("#aabbcc")
            ax.xaxis.label.set_color("#aabbcc")
            ax.title.set_color("#ccddee")
            for spine in ax.spines.values():
                spine.set_color("#2a3040")

        axes[0].plot(t, amps, lw=0.9, color="#4fc3f7", label="Amplitude")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Face Bin — Amplitude")
        axes[0].legend(facecolor="#1a1d23", labelcolor="white")
        axes[0].grid(True, alpha=0.15)

        axes[1].plot(t, phases,       lw=0.7, color="#888", alpha=0.5, label="Raw phase")
        axes[1].plot(t, clean_phases, lw=0.9, color="#ffb74d", label="Breathing removed")
        axes[1].set_ylabel("Phase (rad)")
        axes[1].set_title("Face Bin — Phase (breathing cancelled)")
        axes[1].legend(facecolor="#1a1d23", labelcolor="white")
        axes[1].grid(True, alpha=0.15)

        deriv_arr  = np.array(deriv)
        thresh_arr = np.array(thresh)
        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        peaks, _ = find_peaks(deriv_arr, height=thresh_arr, distance=refractory_chirps)

        axes[2].plot(t, deriv_arr,  lw=0.9, color="#a5d6a7", label="|Δphase| feature")
        axes[2].plot(t, thresh_arr, lw=1.2, linestyle="--", color="#ef5350",
                     label="Adaptive threshold")
        if len(peaks):
            axes[2].plot(t[peaks], deriv_arr[peaks], "x", color="#ffff00",
                         ms=8, mew=2, label=f"Blinks ({len(peaks)})")
        axes[2].set_ylabel("Phase derivative (rad)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Blink Detection — Phase Derivative")
        axes[2].legend(facecolor="#1a1d23", labelcolor="white")
        axes[2].grid(True, alpha=0.15)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/devices", response_model=list[DeviceInfo], tags=["Devices"])
def get_devices():
    """List all audio devices visible to PyAudio, flagging duplex-capable ones."""
    return list_audio_devices()


@app.post("/api/session/start", response_model=StartResponse, tags=["Session"])
def start_session(cfg: SessionConfig = SessionConfig()):
    """
    Start a new detection session.  If one is already running it is stopped first.
    All configuration parameters are optional — omit them to use defaults.
    """
    global _session
    with _session_lock:
        if _session is not None and _session._running:
            _session.stop()
        _session = BlinkListener(cfg)
        try:
            _session.start(device_index=cfg.device_index)
        except Exception as exc:
            _session = None
            raise HTTPException(status_code=500, detail=f"Failed to open audio stream: {exc}")
    return StartResponse(ok=True, message="Session started")


@app.post("/api/session/stop", response_model=StartResponse, tags=["Session"])
def stop_session():
    """Stop the current detection session."""
    global _session
    with _session_lock:
        if _session is None:
            raise HTTPException(status_code=404, detail="No active session")
        _session.stop()
    return StartResponse(ok=True, message="Session stopped")


@app.post("/api/session/dismiss", response_model=StartResponse, tags=["Session"])
def dismiss_alert():
    """Dismiss the triggered alert and reset the blink window."""
    global _session
    with _session_lock:
        if _session is None:
            raise HTTPException(status_code=404, detail="No active session")
        _session.dismiss_alert()
    return StartResponse(ok=True, message="Alert dismissed")


@app.get("/api/session/status", response_model=StatusResponse, tags=["Session"])
def get_status():
    """
    Poll-friendly status snapshot.  Suitable for simple polling loops
    (recommended interval: 250–500 ms).  For real-time push use /ws/live.
    """
    global _session
    if _session is None:
        raise HTTPException(status_code=404, detail="No active session")
    return _session.snapshot()


@app.get("/api/session/chart", tags=["Session"])
def get_chart(dpi: int = 100):
    """
    Returns the 3-panel diagnostic chart (amplitude / phase / blink feature)
    as a PNG image.  Returns 204 if not enough data has been collected yet.
    """
    global _session
    if _session is None:
        raise HTTPException(status_code=404, detail="No active session")
    png = _session.build_png_chart(dpi=dpi)
    if png is None:
        return Response(status_code=204)
    return Response(content=png, media_type="image/png")


# ── WebSocket live feed ───────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Push a StatusResponse JSON frame every ~250 ms.

    Message types sent to client
    ─────────────────────────────
    { "type": "status",  "data": <StatusResponse> }
    { "type": "error",   "message": "..." }
    { "type": "no_session" }

    The client can send:
    { "action": "dismiss" }   — dismiss the current alert
    { "action": "ping" }      — server echoes { "type": "pong" }
    """
    await websocket.accept()

    async def receive_loop():
        """Process inbound client messages without blocking the push loop."""
        global _session
        try:
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")
                if action == "dismiss" and _session is not None:
                    _session.dismiss_alert()
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
        except (WebSocketDisconnect, Exception):
            pass

    asyncio.create_task(receive_loop())

    try:
        while True:
            global _session
            if _session is None:
                await websocket.send_json({"type": "no_session"})
            else:
                snap = _session.snapshot()
                await websocket.send_json({
                    "type": "status",
                    "data": snap.model_dump(),
                })
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, Exception):
        pass


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)