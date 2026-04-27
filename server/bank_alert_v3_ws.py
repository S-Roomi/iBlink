"""
iBlink FastAPI Server
=====================
REST + WebSocket API exposing BlinkListenerST to the React frontend.

Endpoints:
  GET  /api/devices            — list audio devices
  POST /api/session/start      — start a detection session
  POST /api/session/stop       — stop the current session
  POST /api/session/dismiss    — dismiss the active alert
  GET  /api/session/status     — current session state (404 if no session)
  GET  /api/session/chart      — PNG chart (204 if not enough data yet)
  WS   /ws/live                — live status push every ~0.5 s

Run with:
    uvicorn server.bank_alert_v3_ws:app --reload --port 8000
"""

import asyncio
import io
import json
import time
import threading
import queue
from collections import deque
from typing import Optional

import numpy as np
import pyaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import warnings

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

warnings.filterwarnings("ignore")

DEFAULTS = dict(
    sample_rate     = 44100,
    f_start         = 18000,
    f_end           = 22000,
    chirp_duration  = 0.04,
    face_dist_min   = 0.3,
    face_dist_max   = 0.9,
    threshold_mult  = 3.0,
    refractory_s    = 0.35,
    calibration_s   = 3.0,
    smooth_window   = 3,
    alert_window_s  = 5,
    alert_threshold = 5,
    sensing_delay_s = 2.0,
)

SPEED_OF_SOUND = 343.0

def make_chirp(f_start, f_end, duration, sample_rate):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    c = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method="linear")
    c *= np.hanning(len(c))
    return c.astype(np.float32)


def get_audio_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            in_ch  = int(info.get("maxInputChannels",  0))
            out_ch = int(info.get("maxOutputChannels", 0))
            devices.append({
                "index":             i,
                "name":              info.get("name", f"Device {i}"),
                "input_channels":    in_ch,
                "output_channels":   out_ch,
                "is_duplex":         in_ch > 0 and out_ch > 0,
                "default_sample_rate": info.get("defaultSampleRate"),
                "host_api":          None,
            })
        except Exception:
            pass
    p.terminate()
    return devices

# ── BlinkListenerST ────────────────────────────────────────────────────────────
# Extracted from bank_alert_v3.py so this file has no Streamlit dependency.

class BlinkListenerST:
    BREATH_CUTOFF_HZ = 1.0

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
        self.alert_window_s  = cfg["alert_window_s"]
        self.alert_threshold = cfg["alert_threshold"]
        self.sensing_delay_s = cfg.get("sensing_delay_s", 2.0)

        self.tx_chirp    = make_chirp(self.f_start, self.f_end, self.chirp_dur, self.sr)
        self._range_bins = self._compute_range_bins()
        self.face_bin: Optional[int] = None

        self._calib_spectra: list = []
        self._static_clutter: Optional[np.ndarray] = None
        self._calibration_n = int(self.calibration_s / self.chirp_dur)

        maxlen = 500
        self._phase_raw: deque = deque(maxlen=maxlen)
        self._unwrap_accum: float = 0.0

        self._phasor_buffer: deque = deque(maxlen=maxlen)
        self._viewing_center: Optional[complex] = None
        self._prev_amp_from_center: Optional[float] = None
        self.UPDATE_INTERVAL = max(50, int(2.0 / self.chirp_dur))
        self._chirps_since_update: int = 0
        self._calib_spectra_full: list = []

        self.amp_history    = deque(maxlen=maxlen)
        self.phase_history  = deque(maxlen=maxlen)
        self.deriv_history  = deque(maxlen=maxlen)
        self.thresh_history = deque(maxlen=maxlen)

        self._bg_level: Optional[float] = None
        self._bg_alpha = 0.02

        self.blink_count  = 0
        self.status_msg   = "Idle"
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

    # ── range ──────────────────────────────────────────────────────────────────

    def _compute_range_bins(self):
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

    # ── FMCW processing ────────────────────────────────────────────────────────

    def _process_chirp(self, rx_chunk) -> Optional[complex]:
        if len(rx_chunk) < self.chirp_n:
            return None
        rx   = rx_chunk[: self.chirp_n].astype(np.float32)
        beat = rx * self.tx_chirp
        spec = np.fft.rfft(beat, n=self.chirp_n)

        if self._static_clutter is None:
            self._calib_spectra.append(spec.copy())
            self._calib_spectra_full.append(spec.copy())
            if len(self._calib_spectra) >= self._calibration_n:
                self._static_clutter = np.mean(self._calib_spectra, axis=0)
                lo, hi = self._face_bin_range()
                stacked = np.array(self._calib_spectra_full)
                best_bin, best_var = lo, -1.0
                for b in range(lo, hi):
                    col = stacked[:, b]
                    var_iq = float(np.var(col.real) + np.var(col.imag))
                    if var_iq > best_var:
                        best_var, best_bin = var_iq, b
                self.face_bin = best_bin
                dist = (self._range_bins[self.face_bin]
                        if self.face_bin < len(self._range_bins) else 0)
                self.status_msg = f"✅ Face locked at {dist:.2f} m — monitoring…"
                for s in self._calib_spectra_full:
                    self._phasor_buffer.append(
                        complex(s[self.face_bin] - self._static_clutter[self.face_bin])
                    )
            return None

        clean = spec - self._static_clutter
        return complex(clean[self.face_bin])

    # ── phase helpers ──────────────────────────────────────────────────────────

    def _unwrap_push(self, wrapped_phase: float) -> float:
        if not self._phase_raw:
            self._unwrap_accum = wrapped_phase
            return wrapped_phase
        prev_wrapped = np.angle(np.exp(1j * self._phase_raw[-1]))
        diff = wrapped_phase - prev_wrapped
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        self._unwrap_accum += diff
        return self._unwrap_accum

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

    # ── I-Q arc fitting (Pratt) ────────────────────────────────────────────────

    @staticmethod
    def _fit_arc_center(phasors) -> Optional[complex]:
        pts = np.array([[p.real, p.imag] for p in phasors], dtype=np.float64)
        if len(pts) < 6:
            return None
        x, y = pts[:, 0], pts[:, 1]
        A   = np.column_stack([x, y, np.ones(len(x))])
        b_v = x ** 2 + y ** 2
        try:
            result, _, _, sv = np.linalg.lstsq(A, b_v, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if sv[-1] < 1e-10:
            return None
        return complex(result[0] / 2.0, result[1] / 2.0)

    def _update_viewing_position(self):
        recent = list(self._phasor_buffer)[-self.UPDATE_INTERVAL:]
        new_center = self._fit_arc_center(recent)
        if new_center is not None:
            self._viewing_center = new_center

    # ── blink detection ────────────────────────────────────────────────────────

    def _detect_blink(self, face_phasor: complex) -> bool:
        amp_raw      = abs(face_phasor)
        wrapped_ph   = np.angle(face_phasor)
        unwrapped_ph = self._unwrap_push(wrapped_ph)

        self._phase_raw.append(unwrapped_ph)
        self.phase_history.append(unwrapped_ph)
        self._phasor_buffer.append(face_phasor)

        if len(self._phasor_buffer) < 4 or len(self._phase_raw) < 4:
            self.amp_history.append(amp_raw)
            self.deriv_history.append(0.0)
            self.thresh_history.append(1.0)
            return False

        self._chirps_since_update += 1
        if self._chirps_since_update >= self.UPDATE_INTERVAL:
            self._update_viewing_position()
            self._chirps_since_update = 0

        if self._viewing_center is None and len(self._phasor_buffer) >= self.UPDATE_INTERVAL:
            self._update_viewing_position()

        if self._viewing_center is not None:
            amp_from_vp = abs(face_phasor - self._viewing_center)
        else:
            amp_from_vp = amp_raw

        self.amp_history.append(amp_from_vp)

        if self._prev_amp_from_center is None:
            self._prev_amp_from_center = amp_from_vp
        d_feature = abs(amp_from_vp - self._prev_amp_from_center)
        self._prev_amp_from_center = amp_from_vp

        phase_arr = np.array(self._phase_raw)
        clean_ph  = self._remove_breathing(phase_arr)
        d_phase   = abs(clean_ph[-1] - clean_ph[-2])

        feature_raw = 0.7 * d_feature + 0.3 * d_phase
        self.deriv_history.append(feature_raw)

        if len(self.deriv_history) >= self.smooth_window:
            feature = float(np.mean(list(self.deriv_history)[-self.smooth_window:]))
        else:
            feature = feature_raw

        threshold_now = (self._bg_level * self.threshold_mult
                         if self._bg_level is not None else 1.0)
        if feature > 3.0 * threshold_now and self._bg_level is not None:
            self._phase_raw.clear()
            self._phasor_buffer.clear()
            self._viewing_center       = None
            self._prev_amp_from_center = None
            self._chirps_since_update  = 0
            self._bg_level             = None
            self._phase_raw.append(unwrapped_ph)
            self._phasor_buffer.append(face_phasor)
            self.status_msg = "⚠️ Large movement — re-initialising VP…"
            self.thresh_history.append(threshold_now)
            return False

        if self._bg_level is None:
            self._bg_level = feature
        else:
            if feature < threshold_now:
                self._bg_level = ((1.0 - self._bg_alpha) * self._bg_level
                                  + self._bg_alpha * feature)

        threshold = self._bg_level * self.threshold_mult
        self.thresh_history.append(threshold)

        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        self._chirps_since_blink += 1
        if feature > threshold and self._chirps_since_blink >= refractory_chirps:
            self._chirps_since_blink = 0
            return True
        return False

    # ── alert ──────────────────────────────────────────────────────────────────

    def _check_alert(self):
        now    = time.time()
        cutoff = now - self.alert_window_s
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        self.blinks_in_window = len(self._blink_times)
        if self.blinks_in_window >= self.alert_threshold and not self.alert_triggered:
            self.alert_triggered = True
            self.alert_time      = now
            self.status_msg      = "🚨 ROBBERY ALERT TRIGGERED"

    def seconds_until_window_reset(self) -> float:
        if not self._blink_times:
            return float(self.alert_window_s)
        expires   = self._blink_times[0] + self.alert_window_s
        remaining = expires - time.time()
        return max(0.0, remaining)

    # ── audio callbacks ────────────────────────────────────────────────────────

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
        self._start_wall_time = time.time()
        self._sensing_enabled = False
        self.status_msg = f"⏳ Starting up — sensing begins in {self.sensing_delay_s:.0f} s…"

        while self._running or not self.rx_queue.empty():
            if not self._sensing_enabled:
                elapsed = time.time() - self._start_wall_time
                if elapsed >= self.sensing_delay_s:
                    self._sensing_enabled = True
                    if self._static_clutter is None:
                        self.status_msg = f"🔊 Calibrating for {self.calibration_s:.0f} s — sit still…"

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

    # ── public interface ───────────────────────────────────────────────────────

    def start(self, dev_index=None):
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16, channels=1, rate=self.sr,
                input=True, output=True, frames_per_buffer=self.chirp_n,
                input_device_index=dev_index, output_device_index=dev_index,
                stream_callback=self._tx_callback,
            )
        except OSError:
            self.stream = self.audio.open(
                format=pyaudio.paInt16, channels=1, rate=self.sr,
                input=True, output=True, frames_per_buffer=self.chirp_n,
                stream_callback=self._tx_callback,
            )
        self._running   = True
        self.status_msg = f"⏳ Starting up — sensing begins in {self.sensing_delay_s:.0f} s…"
        threading.Thread(target=self._processing_thread, daemon=True).start()

    def stop(self):
        self._running = False
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.status_msg = "⏹️ Stopped"

    def dismiss_alert(self):
        self.alert_triggered  = False
        self.alert_time       = None
        self._blink_times.clear()
        self.blinks_in_window = 0

    def build_figure(self):
        n = min(len(self.deriv_history), len(self.thresh_history),
                len(self.amp_history),   len(self.phase_history))
        if n < 5:
            return None

        deriv  = list(self.deriv_history)[-n:]
        thresh = list(self.thresh_history)[-n:]
        amps   = list(self.amp_history)[-n:]
        phases = np.array(list(self.phase_history)[-n:])
        t      = np.arange(n) * self.chirp_dur
        clean_phases = self._remove_breathing(phases)

        iq_pts  = list(self._phasor_buffer)[-min(200, len(self._phasor_buffer)):]
        iq_real = [p.real for p in iq_pts]
        iq_imag = [p.imag for p in iq_pts]
        vp      = self._viewing_center

        fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                                 gridspec_kw={"width_ratios": [2, 1]})
        fig.patch.set_facecolor("#0a0c10")
        for ax in axes.flat:
            ax.set_facecolor("#111620")
            ax.tick_params(colors="#aabbcc")
            ax.yaxis.label.set_color("#aabbcc")
            ax.xaxis.label.set_color("#aabbcc")
            ax.title.set_color("#ccddee")
            for spine in ax.spines.values():
                spine.set_color("#2a3040")

        ax0, ax_iq = axes[0, 0], axes[0, 1]
        ax1, ax2   = axes[1, 0], axes[1, 1]

        ax0.plot(t, amps, lw=0.9, color="#4fc3f7", label="Amplitude from VP")
        ax0.set_ylabel("Amplitude (from VP)")
        ax0.set_title("Face Bin — Amplitude from Optimal Viewing Position")
        ax0.legend(facecolor="#1a1d23", labelcolor="white", fontsize=8)
        ax0.grid(True, alpha=0.15)

        ax_iq.scatter(iq_real, iq_imag, s=3, color="#4fc3f7", alpha=0.5, label="I-Q samples")
        if vp is not None:
            ax_iq.plot(vp.real, vp.imag, "r*", ms=12,
                       label=f"VP ({vp.real:.2f}, {vp.imag:.2f})")
            r_vals = [abs(complex(ix, iq) - vp) for ix, iq in zip(iq_real, iq_imag)]
            r_med  = float(np.median(r_vals)) if r_vals else 0.0
            theta  = np.linspace(0, 2 * np.pi, 200)
            ax_iq.plot(vp.real + r_med * np.cos(theta),
                       vp.imag + r_med * np.sin(theta),
                       "--", color="#ff8800", lw=1, alpha=0.6, label="Fit arc")
        ax_iq.set_xlabel("I (real)")
        ax_iq.set_ylabel("Q (imag)")
        ax_iq.set_title(f"I-Q Space — {'VP ready' if vp is not None else 'VP fitting…'}")
        ax_iq.legend(facecolor="#1a1d23", labelcolor="white", fontsize=7)
        ax_iq.grid(True, alpha=0.15)
        ax_iq.set_aspect("equal", "datalim")

        ax1.plot(t, phases,       lw=0.7, color="#888", alpha=0.5, label="Raw phase")
        ax1.plot(t, clean_phases, lw=0.9, color="#ffb74d", label="Breathing removed")
        ax1.set_ylabel("Phase (rad)")
        ax1.set_title("Face Bin — Phase (breathing cancelled)")
        ax1.legend(facecolor="#1a1d23", labelcolor="white", fontsize=8)
        ax1.grid(True, alpha=0.15)

        deriv_arr  = np.array(deriv)
        thresh_arr = np.array(thresh)
        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        peaks, _ = find_peaks(deriv_arr, height=thresh_arr, distance=refractory_chirps)
        t2 = np.arange(len(deriv_arr)) * self.chirp_dur
        ax2.plot(t2, deriv_arr,  lw=0.9, color="#a5d6a7", label="Blink feature")
        ax2.plot(t2, thresh_arr, lw=1.2, linestyle="--", color="#ef5350", label="Threshold")
        if len(peaks):
            ax2.plot(t2[peaks], deriv_arr[peaks], "x", color="#ffff00",
                     ms=8, mew=2, label=f"Blinks ({len(peaks)})")
        ax2.set_ylabel("Feature amplitude")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Blink Detection — VP Amplitude Derivative")
        ax2.legend(facecolor="#1a1d23", labelcolor="white", fontsize=7)
        ax2.grid(True, alpha=0.15)

        plt.tight_layout()
        return fig


app = FastAPI(title="iBlink API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── global session state ───────────────────────────────────────────────────────

_session_lock          = threading.Lock()
_listener: Optional[BlinkListenerST] = None
_selected_device_index: Optional[int]  = None
_selected_device_name:  Optional[str]  = None


class SessionConfig(BaseModel):
    sample_rate:     int   = DEFAULTS["sample_rate"]
    f_start:         int   = DEFAULTS["f_start"]
    f_end:           int   = DEFAULTS["f_end"]
    chirp_duration:  float = DEFAULTS["chirp_duration"]
    face_dist_min:   float = DEFAULTS["face_dist_min"]
    face_dist_max:   float = DEFAULTS["face_dist_max"]
    threshold_mult:  float = DEFAULTS["threshold_mult"]
    refractory_s:    float = DEFAULTS["refractory_s"]
    calibration_s:   float = DEFAULTS["calibration_s"]
    smooth_window:   int   = DEFAULTS["smooth_window"]
    alert_window_s:  int   = DEFAULTS["alert_window_s"]
    alert_threshold: int   = DEFAULTS["alert_threshold"]
    sensing_delay_s: float = DEFAULTS["sensing_delay_s"]
    device_index:    Optional[int] = None


def _build_status(bl: BlinkListenerST) -> dict:
    face_dist = None
    if bl.face_bin is not None and bl.face_bin < len(bl._range_bins):
        face_dist = round(float(bl._range_bins[bl.face_bin]), 3)

    feature_value   = float(list(bl.deriv_history)[-1])  if bl.deriv_history  else None
    threshold_value = float(list(bl.thresh_history)[-1]) if bl.thresh_history else None
    bg_level        = float(bl._bg_level) if bl._bg_level is not None else None

    calib_remaining = None
    if bl._static_clutter is None and bl._start_wall_time is not None:
        elapsed          = time.time() - bl._start_wall_time
        sensing_elapsed  = max(0.0, elapsed - bl.sensing_delay_s)
        calib_remaining  = round(max(0.0, bl.calibration_s - sensing_elapsed), 2)

    return {
        "running":               bl._running,
        "status_msg":            bl.status_msg,
        "blink_count":           bl.blink_count,
        "chirps_processed":      bl._chirp_index,
        "face_bin":              bl.face_bin,
        "face_distance_m":       face_dist,
        "blinks_in_window":      bl.blinks_in_window,
        "alert_threshold":       bl.alert_threshold,
        "alert_window_s":        bl.alert_window_s,
        "seconds_until_reset":   round(bl.seconds_until_window_reset(), 2),
        "alert_triggered":       bl.alert_triggered,
        "alert_time":            bl.alert_time,
        "sensing_enabled":       bl._sensing_enabled,
        "calibration_remaining_s": calib_remaining,
        "selected_device_index": _selected_device_index,
        "selected_device_name":  _selected_device_name,
        "feature_value":         feature_value,
        "threshold_value":       threshold_value,
        "background_level":      bg_level,
    }


@app.get("/api/devices")
def list_devices():
    return get_audio_devices()


@app.post("/api/session/start")
def start_session(cfg: SessionConfig):
    global _listener, _selected_device_index, _selected_device_name
    with _session_lock:
        if _listener is not None and _listener._running:
            raise HTTPException(status_code=409, detail="Session already running")

        config = cfg.model_dump(exclude={"device_index"})
        _selected_device_index = cfg.device_index

        if cfg.device_index is not None:
            devices = get_audio_devices()
            dev = next((d for d in devices if d["index"] == cfg.device_index), None)
            _selected_device_name = dev["name"] if dev else None
        else:
            _selected_device_name = None

        bl = BlinkListenerST(config)
        bl.start(dev_index=cfg.device_index)
        _listener = bl

    return {"ok": True, "message": "Session started"}


@app.post("/api/session/stop")
def stop_session():
    global _listener
    with _session_lock:
        if _listener is None:
            raise HTTPException(status_code=404, detail="No active session")
        _listener.stop()
        _listener = None
    return {"ok": True, "message": "Session stopped"}


@app.post("/api/session/dismiss")
def dismiss_alert():
    with _session_lock:
        if _listener is None:
            raise HTTPException(status_code=404, detail="No active session")
        _listener.dismiss_alert()
    return {"ok": True, "message": "Alert dismissed"}


@app.get("/api/session/status")
def get_status():
    if _listener is None:
        raise HTTPException(status_code=404, detail="No active session")
    return _build_status(_listener)


@app.get("/api/session/chart")
def get_chart():
    if _listener is None:
        raise HTTPException(status_code=404, detail="No active session")
    fig = _listener.build_figure()
    if fig is None:
        raise HTTPException(status_code=204, detail="Insufficient data for chart")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()

    async def push_status():
        try:
            while True:
                if _listener is None:
                    payload = {"type": "no_session"}
                else:
                    payload = {"type": "status", "data": _build_status(_listener)}
                await websocket.send_json(payload)
                await asyncio.sleep(0.5)
        except Exception:
            pass

    push_task = asyncio.create_task(push_status())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if isinstance(msg, dict) and msg.get("action") == "dismiss" and _listener is not None:
                    _listener.dismiss_alert()
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        push_task.cancel()
        try:
            await push_task
        except asyncio.CancelledError:
            pass
