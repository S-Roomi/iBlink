"""
Bank Robbery Alert — Streamlit Web App
=======================================
Acoustic FMCW eye-blink detector repurposed as a silent duress system.
If a teller blinks ≥ N times within a T-second window, a bank-robbery
alert fires on screen.

Run with:  streamlit run bank_robbery_alert.py

Dependencies:
    pip install streamlit numpy scipy pyaudio matplotlib sounddevice
"""

import time
import threading
import queue
from collections import deque
from typing import Optional

import numpy as np
import pyaudio
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Robbery Alert",
    page_icon="🏦",
    layout="wide",
)

# ── custom CSS — bank / dark theme ────────────────────────────────────────────
st.markdown("""
<style>
/* ── global background ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0c10 !important;
    color: #e8e8e8;
}
[data-testid="stSidebar"] {
    background-color: #0f1318 !important;
    border-right: 1px solid #1e2530;
}

/* ── title bar styling ── */
.bank-header {
    background: linear-gradient(135deg, #1a0000 0%, #3d0000 50%, #1a0000 100%);
    border: 2px solid #8b0000;
    border-radius: 10px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 18px;
}
.bank-header h1 {
    color: #ff2222;
    font-size: 2.2rem;
    font-weight: 900;
    margin: 0;
    letter-spacing: 2px;
    text-shadow: 0 0 18px #ff0000aa;
}
.bank-header .subtitle {
    color: #ffaaaa;
    font-size: 0.85rem;
    margin-top: 4px;
    letter-spacing: 1px;
}

/* ── ALERT BANNER ── */
.alert-banner {
    background: linear-gradient(90deg, #8b0000, #cc0000, #8b0000);
    background-size: 200% 100%;
    animation: alertPulse 0.8s ease-in-out infinite alternate;
    border: 3px solid #ff0000;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    margin: 16px 0;
}
.alert-banner h2 {
    color: #ffffff;
    font-size: 2.6rem;
    font-weight: 900;
    margin: 0 0 8px 0;
    letter-spacing: 4px;
    text-shadow: 0 0 20px #fff;
    animation: textFlicker 0.8s ease-in-out infinite alternate;
}
.alert-banner p {
    color: #ffdddd;
    font-size: 1.1rem;
    margin: 0;
}
@keyframes alertPulse {
    from { background-position: 0% 50%; box-shadow: 0 0 20px #ff000066; }
    to   { background-position: 100% 50%; box-shadow: 0 0 50px #ff0000cc; }
}
@keyframes textFlicker {
    from { opacity: 1; }
    to   { opacity: 0.85; }
}

/* ── countdown ring ── */
.countdown-wrap {
    display: flex;
    justify-content: center;
    margin: 12px 0;
}
.countdown-circle {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    border: 4px solid #cc9900;
    background: #1a1500;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 18px #cc990066;
}
.countdown-number {
    font-size: 2.2rem;
    font-weight: 900;
    color: #ffcc00;
    line-height: 1;
}
.countdown-label {
    font-size: 0.6rem;
    color: #aa8800;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── blink counter card ── */
.blink-card {
    background: #0f1a0f;
    border: 2px solid #1a4a1a;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.blink-number {
    font-size: 3rem;
    font-weight: 900;
    color: #44ff44;
    line-height: 1;
}
.blink-label {
    font-size: 0.75rem;
    color: #77aa77;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}
.blink-warning {
    border-color: #884400 !important;
    background: #1a0f00 !important;
}
.blink-warning .blink-number { color: #ffaa00 !important; }
.blink-warning .blink-label  { color: #aa7700 !important; }

/* ── status pill ── */
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.status-idle       { background: #1c2030; color: #8899bb; border: 1px solid #334; }
.status-calibrating{ background: #1c1800; color: #ddbb44; border: 1px solid #553; }
.status-active     { background: #0f1f0f; color: #44dd44; border: 1px solid #363; }
.status-alert      { background: #2a0000; color: #ff4444; border: 1px solid #833; }

/* ── metric cards override ── */
[data-testid="stMetric"] {
    background: #111620;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #8899bb !important; }
[data-testid="stMetricValue"] { color: #e8e8e8 !important; font-weight: 700; }

/* ── sidebar sliders ── */
[data-testid="stSlider"] > div > div > div > div { background: #8b0000 !important; }

/* ── buttons ── */
[data-testid="stButton"] button[kind="primary"] {
    background: #8b0000 !important;
    border: 1px solid #cc2222 !important;
    color: white !important;
    font-weight: 700;
    letter-spacing: 1px;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: #cc0000 !important;
    box-shadow: 0 0 12px #ff000066;
}
</style>
""", unsafe_allow_html=True)

# ── default parameter values ──────────────────────────────────────────────────
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
    # Alert-specific defaults
    alert_window_s  = 5,
    alert_threshold = 5,
)

SPEED_OF_SOUND = 343.0


# ── helpers ───────────────────────────────────────────────────────────────────

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
            devices.append({
                "index": i,
                "name": info.get("name", f"Device {i}"),
                "in":   int(info.get("maxInputChannels", 0)),
                "out":  int(info.get("maxOutputChannels", 0)),
            })
        except Exception:
            # Skip devices that PyAudio cannot query (e.g. virtual/broken drivers)
            pass
    p.terminate()
    return devices


# ── BlinkListener core (adapted for Streamlit) ────────────────────────────────

class BlinkListenerST:
    """
    BlinkListener — paper-accurate implementation for Streamlit + Bank Robbery Alert.

    Signal pipeline (per the BlinkListener paper, Liu et al. IMWUT 2021):
    ┌──────────────────────────────────────────────────────────────────┐
    │ 1. TX: continuous ultrasonic FMCW chirp via speaker             │
    │ 2. RX: microphone captures direct + reflected signal            │
    │ 3. Beat signal = TX_chirp × RX_chunk  (de-chirp / mix-down)     │
    │ 4. Range FFT → complex spectrum; pick face bin                  │
    │ 5. Static clutter removal: subtract calibration mean spectrum   │
    │ 6. Extract PHASE of face bin per chirp                          │
    │ 7. Accumulate N samples → unwrap phase sequence                 │
    │ 8. Breathing cancellation: remove low-frequency trend           │
    │    (high-pass filter; breathing ≈ 0.1–0.5 Hz; blink ≈ 4–10 Hz) │
    │ 9. Blink feature = absolute phase derivative of cleaned signal  │
    │10. Smooth feature; adaptive threshold; refractory gate          │
    └──────────────────────────────────────────────────────────────────┘
    """

    # Breathing is ~0.1–0.5 Hz.  We high-pass above 1 Hz to kill it entirely
    # while preserving blink transients (typical blink ~150–400 ms → 2.5–6 Hz).
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

        # Alert parameters
        self.alert_window_s  = cfg["alert_window_s"]
        self.alert_threshold = cfg["alert_threshold"]
        self.sensing_delay_s = cfg.get("sensing_delay_s", 2.0)

        # ── chirp template ────────────────────────────────────────────────────
        self.tx_chirp    = make_chirp(self.f_start, self.f_end, self.chirp_dur, self.sr)
        self._range_bins = self._compute_range_bins()
        self.face_bin: Optional[int] = None

        # ── calibration / static clutter ──────────────────────────────────────
        # Accumulate mean spectrum during calibration to subtract static reflectors
        self._calib_spectra: list  = []          # raw complex spectra during calibration
        self._static_clutter: Optional[np.ndarray] = None  # mean calibration spectrum
        self._calibration_n = int(self.calibration_s / self.chirp_dur)

        # ── phase tracking ────────────────────────────────────────────────────
        # We keep a rolling buffer of raw unwrapped phase values.
        # Buffer length: enough for the high-pass filter to be stable.
        # At ~25 chirps/s (40 ms each), 200 samples ≈ 8 s of history.
        maxlen = 500
        self._phase_raw: deque = deque(maxlen=maxlen)   # unwrapped, before HP filter
        self._unwrap_accum: float = 0.0                  # running unwrap offset

        # ── I-Q viewing position scheme (BlinkListener §6.2) ─────────────────
        # Store the raw complex phasors at the face bin so we can fit an arc.
        # The arc is driven by breathing/heartbeat involuntary head movement;
        # its centre is the optimal viewing position.
        self._phasor_buffer: deque = deque(maxlen=maxlen)   # complex phasors
        self._viewing_center: Optional[complex] = None       # arc centre (optimal VP)
        self._prev_amp_from_center: Optional[float] = None  # for derivative feature
        # Update the viewing position every UPDATE_INTERVAL chirps (~2 s at 40 ms)
        self.UPDATE_INTERVAL = max(50, int(2.0 / self.chirp_dur))
        self._chirps_since_update: int = 0

        # Per-bin spectra kept during calibration — needed for 2D I-Q variance
        # face-bin selection (paper §5.2) instead of naïve 1D energy.
        self._calib_spectra_full: list = []   # list of full complex rfft vectors

        # ── feature / threshold histories (for plotting) ─────────────────────
        self.amp_history    = deque(maxlen=maxlen)   # amplitude from viewing centre
        self.phase_history  = deque(maxlen=maxlen)   # kept for the chart only
        self.deriv_history  = deque(maxlen=maxlen)   # blink feature
        self.thresh_history = deque(maxlen=maxlen)   # adaptive threshold

        # ── adaptive background (EMA on quiet frames only) ───────────────────
        self._bg_level: Optional[float] = None
        self._bg_alpha  = 0.02   # slow adaptation: ~50 frames to fully update

        # ── refractory / chirp bookkeeping ───────────────────────────────────
        self.blink_count  = 0
        self.status_msg   = "Idle"
        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        self._chirps_since_blink = refractory_chirps
        self._chirp_index        = 0

        # ── audio plumbing ────────────────────────────────────────────────────
        self.audio     = pyaudio.PyAudio()
        self.rx_queue  = queue.Queue()
        self._running  = False
        self._tx_buffer = np.tile(self.tx_chirp, 8)
        self._tx_pos    = 0
        self._tx_lock   = threading.Lock()
        self._rx_accum  = np.zeros(0, dtype=np.float32)

        # ── alert state ───────────────────────────────────────────────────────
        self._blink_times: deque  = deque()
        self.alert_triggered      = False
        self.alert_time: Optional[float] = None
        self._sensing_enabled     = False
        self._start_wall_time: Optional[float] = None
        self.blinks_in_window: int = 0

    # ── range helpers ─────────────────────────────────────────────────────────

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

    # ── FMCW de-chirp + static clutter removal ────────────────────────────────

    def _process_chirp(self, rx_chunk) -> Optional[complex]:
        """
        De-chirp one RX chunk, subtract static clutter, return the complex
        phasor at the face bin.  Returns None until calibration is complete.
        """
        if len(rx_chunk) < self.chirp_n:
            return None

        rx   = rx_chunk[: self.chirp_n].astype(np.float32)
        beat = rx * self.tx_chirp
        spec = np.fft.rfft(beat, n=self.chirp_n)   # complex spectrum

        # ── Phase 1: collect calibration spectra ─────────────────────────────
        if self._static_clutter is None:
            self._calib_spectra.append(spec.copy())
            self._calib_spectra_full.append(spec.copy())   # keep full spectrum for IQ variance
            if len(self._calib_spectra) >= self._calibration_n:
                # Build mean static clutter template
                self._static_clutter = np.mean(self._calib_spectra, axis=0)
                # ── Paper §5.2: select face bin by maximum 2D I-Q variance ──
                # Breathing/heartbeat embedded interference creates an arc
                # trajectory in I-Q space; that bin has the highest 2D spread.
                # The naïve 1D energy method (used for breathing detection) fails
                # for eye blinks because the eye-reflection amplitude is tiny.
                lo, hi = self._face_bin_range()
                stacked = np.array(self._calib_spectra_full)   # (N, bins)
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
                # Seed the phasor buffer with calibration samples at the face bin
                # so arc fitting can begin immediately after calibration.
                for s in self._calib_spectra_full:
                    self._phasor_buffer.append(complex(s[self.face_bin] - self._static_clutter[self.face_bin]))
            return None  # still calibrating

        # ── Phase 2: subtract static clutter and return face-bin phasor ──────
        clean = spec - self._static_clutter
        return complex(clean[self.face_bin])

    # ── phase unwrapping helper ───────────────────────────────────────────────

    def _unwrap_push(self, wrapped_phase: float) -> float:
        """
        Extend the continuous unwrapped phase sequence by one sample.
        Maintains a running offset so the sequence never jumps by ±π.
        """
        if not self._phase_raw:
            self._unwrap_accum = wrapped_phase
            return wrapped_phase

        prev_wrapped = np.angle(np.exp(1j * self._phase_raw[-1]))  # re-wrap last stored
        diff = wrapped_phase - prev_wrapped
        # Wrap diff into (−π, π]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        self._unwrap_accum += diff
        return self._unwrap_accum

    # ── breathing cancellation via high-pass filter ───────────────────────────

    def _remove_breathing(self, phase_seq: np.ndarray) -> np.ndarray:
        """
        High-pass filter to remove slow breathing-induced phase drift.
        Breathing: ~0.1–0.5 Hz.  Blink transient: ~2.5–10 Hz.
        Chirp rate = 1/chirp_dur (e.g. 25 Hz for 40 ms chirps).
        We use a simple 2nd-order Butterworth IIR HP filter.
        Falls back to a mean-subtraction if the sequence is too short.
        """
        chirp_rate = 1.0 / self.chirp_dur   # e.g. 25 Hz
        nyq        = chirp_rate / 2.0
        cutoff_norm = self.BREATH_CUTOFF_HZ / nyq

        # Need at least ~3x filter order samples for padlen in filtfilt
        min_len = 18  # 2nd order → padlen=6 → need ≥7; use 18 for safety
        if len(phase_seq) < min_len or cutoff_norm >= 1.0:
            # Fallback: just remove the linear trend (handles breathing well enough)
            return phase_seq - np.linspace(phase_seq[0], phase_seq[-1], len(phase_seq))

        try:
            b, a    = signal.butter(2, cutoff_norm, btype="high")
            cleaned = signal.filtfilt(b, a, phase_seq)
        except Exception:
            cleaned = phase_seq - np.linspace(phase_seq[0], phase_seq[-1], len(phase_seq))
        return cleaned

    # ── I-Q arc fitting (Pratt method) ───────────────────────────────────────

    @staticmethod
    def _fit_arc_center(phasors) -> Optional[complex]:
        """
        Fit a circle to a sequence of complex I-Q samples using the Pratt
        algebraic method (BlinkListener §6.2; Pratt 1987).

        The embedded interference (breathing + heartbeat) moves the composite
        signal along an arc in I-Q space.  The centre of that arc is the
        **optimal viewing position**: measuring signal amplitude *from* that
        point maximises the blink-induced transient and minimises the
        continuous breathing-drift component.

        Returns None if there are too few points or the fit is degenerate.
        """
        pts = np.array([[p.real, p.imag] for p in phasors], dtype=np.float64)
        if len(pts) < 6:
            return None

        x, y = pts[:, 0], pts[:, 1]
        # Pratt: solve  [x  y  1] * [a b c]^T  =  x²+y²
        A   = np.column_stack([x, y, np.ones(len(x))])
        b_v = x ** 2 + y ** 2
        try:
            result, residuals, rank, sv = np.linalg.lstsq(A, b_v, rcond=None)
        except np.linalg.LinAlgError:
            return None

        # Degenerate check: if singular values are tiny the points are collinear
        if sv[-1] < 1e-10:
            return None

        cx = result[0] / 2.0
        cy = result[1] / 2.0
        return complex(cx, cy)

    def _update_viewing_position(self):
        """
        Re-fit the arc centre using the most recent UPDATE_INTERVAL phasors.
        Called every UPDATE_INTERVAL chirps so the optimal viewing position
        tracks slow head-position drift (paper §6.3 Step 3).
        Falls back gracefully to the old centre if fitting fails.
        """
        recent = list(self._phasor_buffer)[-self.UPDATE_INTERVAL:]
        new_center = self._fit_arc_center(recent)
        if new_center is not None:
            self._viewing_center = new_center

    # ── main blink detector ───────────────────────────────────────────────────

    def _detect_blink(self, face_phasor: complex) -> bool:
        """
        Paper-accurate blink detection (BlinkListener §6, Liu et al. 2021):

          1.  Buffer the raw complex phasor.
          2.  Maintain a continuously-updated optimal viewing position (VP)
              by fitting a circle (Pratt method) to the I-Q arc formed by
              breathing/heartbeat embedded interference (§6.2).
          3.  Compute amplitude from the VP — this maximises the blink
              transient and suppresses the breathing arc variation.
          4.  Large-movement guard: if the feature is >> 3× threshold,
              reset and restart (§6.3 Step 4).
          5.  Adaptive background EMA on quiet frames only.
          6.  Refractory gate to prevent double-counting.

        Phase-derivative is kept as a fall-back while the VP is still being
        established (fewer than UPDATE_INTERVAL chirps since calibration).
        """
        amp_raw      = abs(face_phasor)
        wrapped_ph   = np.angle(face_phasor)
        unwrapped_ph = self._unwrap_push(wrapped_ph)

        self._phase_raw.append(unwrapped_ph)
        self.phase_history.append(unwrapped_ph)
        self._phasor_buffer.append(face_phasor)

        # Need at least a few samples in BOTH buffers before computing derivatives.
        # _phasor_buffer is seeded from calibration data so it may already be large,
        # but _phase_raw only fills during live _detect_blink calls — after a cold
        # start or a large-movement restart it can have just 1 entry even when
        # _phasor_buffer is full, causing clean_ph[-2] to crash.
        if len(self._phasor_buffer) < 4 or len(self._phase_raw) < 4:
            self.amp_history.append(amp_raw)
            self.deriv_history.append(0.0)
            self.thresh_history.append(1.0)
            return False

        # ── Step 1: periodic viewing-position update ──────────────────────────
        self._chirps_since_update += 1
        if self._chirps_since_update >= self.UPDATE_INTERVAL:
            self._update_viewing_position()
            self._chirps_since_update = 0

        # Bootstrap VP on the very first full window if not yet set
        if self._viewing_center is None and len(self._phasor_buffer) >= self.UPDATE_INTERVAL:
            self._update_viewing_position()

        # ── Step 2: compute amplitude feature ────────────────────────────────
        # Paper §6.2: measure distance from the optimal viewing position
        # (arc centre) rather than the coordinate origin.  This maximises the
        # blink bump and makes breathing/heartbeat appear as a nearly-constant
        # radius, so it does not corrupt the derivative.
        if self._viewing_center is not None:
            amp_from_vp = abs(face_phasor - self._viewing_center)
        else:
            # VP not yet ready — fall back to origin-referenced amplitude
            amp_from_vp = amp_raw

        self.amp_history.append(amp_from_vp)

        # Instantaneous derivative of amplitude-from-VP
        if self._prev_amp_from_center is None:
            self._prev_amp_from_center = amp_from_vp
        d_feature = abs(amp_from_vp - self._prev_amp_from_center)
        self._prev_amp_from_center = amp_from_vp

        # ── Step 3: fuse with breathing-cancelled phase derivative ────────────
        # The paper notes that a blink causes BOTH an amplitude change (large)
        # and a phase change (small).  Using both improves robustness.
        phase_arr = np.array(self._phase_raw)
        clean_ph  = self._remove_breathing(phase_arr)
        d_phase   = abs(clean_ph[-1] - clean_ph[-2])

        # Weighted combination — amplitude-from-VP is the primary signal
        feature_raw = 0.7 * d_feature + 0.3 * d_phase
        self.deriv_history.append(feature_raw)

        # ── Step 4: smooth ────────────────────────────────────────────────────
        if len(self.deriv_history) >= self.smooth_window:
            feature = float(np.mean(list(self.deriv_history)[-self.smooth_window:]))
        else:
            feature = feature_raw

        # ── Step 5: large-movement restart (§6.3 Step 4) ─────────────────────
        # If the signal spike is more than 3× the current threshold it is not
        # a blink but a large body movement; reset and reinitialise.
        threshold_now = (self._bg_level * self.threshold_mult
                         if self._bg_level is not None else 1.0)
        if feature > 3.0 * threshold_now and self._bg_level is not None:
            # Flush state so we re-establish the VP cleanly
            self._phase_raw.clear()
            self._phasor_buffer.clear()
            self._viewing_center       = None
            self._prev_amp_from_center = None
            self._chirps_since_update  = 0
            self._bg_level             = None
            # Re-seed with the current sample so the buffers are never left
            # completely empty — the < 4 guard above will keep us safe until
            # enough new samples accumulate.
            self._phase_raw.append(unwrapped_ph)
            self._phasor_buffer.append(face_phasor)
            self.status_msg = "⚠️ Large movement — re-initialising VP…"
            self.thresh_history.append(threshold_now)
            return False

        # ── Step 6: adaptive background EMA ──────────────────────────────────
        if self._bg_level is None:
            self._bg_level = feature
        else:
            if feature < threshold_now:
                self._bg_level = ((1.0 - self._bg_alpha) * self._bg_level
                                  + self._bg_alpha * feature)

        threshold = self._bg_level * self.threshold_mult
        self.thresh_history.append(threshold)

        # ── Step 7: refractory gate + threshold check ─────────────────────────
        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        self._chirps_since_blink += 1
        if feature > threshold and self._chirps_since_blink >= refractory_chirps:
            self._chirps_since_blink = 0
            return True
        return False

    # ── alert logic ───────────────────────────────────────────────────────────

    def _check_alert(self):
        """Maintain a sliding window of blink timestamps; fire alert if threshold met."""
        now = time.time()
        cutoff = now - self.alert_window_s
        # Prune old blinks
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        self.blinks_in_window = len(self._blink_times)

        if self.blinks_in_window >= self.alert_threshold and not self.alert_triggered:
            self.alert_triggered = True
            self.alert_time = now
            self.status_msg = "🚨 ROBBERY ALERT TRIGGERED"

    # ── countdown helper ──────────────────────────────────────────────────────

    def seconds_until_window_reset(self) -> float:
        """Seconds remaining before the oldest blink in the window expires."""
        if not self._blink_times:
            return self.alert_window_s
        oldest  = self._blink_times[0]
        expires = oldest + self.alert_window_s
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
            # Check if startup delay has elapsed
            if not self._sensing_enabled:
                elapsed = time.time() - self._start_wall_time
                if elapsed >= self.sensing_delay_s:
                    self._sensing_enabled = True
                    # Status msg will be updated by _process_chirp once calibration
                    # finishes, so only update it here if still in startup phase
                    if self._static_clutter is None:
                        self.status_msg = f"🔊 Calibrating for {self.calibration_s:.0f} s — sit still…"

            try:
                chunk = self.rx_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Full pipeline runs always — warms up clutter model, phase history,
            # background EMA — blink counting only gated by sensing_enabled.
            self._rx_accum = np.concatenate([self._rx_accum, chunk])
            while len(self._rx_accum) >= self.chirp_n:
                chirp_rx       = self._rx_accum[: self.chirp_n]
                self._rx_accum = self._rx_accum[self.chirp_n:]
                self._chirp_index += 1

                face_sig = self._process_chirp(chirp_rx)
                if face_sig is None:
                    continue

                is_blink = self._detect_blink(face_sig)

                # Only count blinks toward the alert after the startup delay
                if is_blink and self._sensing_enabled:
                    self.blink_count += 1
                    self._blink_times.append(time.time())
                    self._check_alert()

    # ── public interface ──────────────────────────────────────────────────────

    def start(self, dev_index=None):
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
        self.status_msg = f"⏳ Starting up — sensing begins in {self.sensing_delay_s:.0f} s…"
        t = threading.Thread(target=self._processing_thread, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.status_msg = "⏹️ Stopped"

    def dismiss_alert(self):
        self.alert_triggered = False
        self.alert_time      = None
        self._blink_times.clear()
        self.blinks_in_window = 0

    def build_figure(self):
        n = min(len(self.deriv_history), len(self.thresh_history),
                len(self.amp_history),   len(self.phase_history))
        if n < 5:
            return None

        deriv  = list(self.deriv_history)[-n:]
        thresh = list(self.thresh_history)[-n:]
        amps   = list(self.amp_history)[-n:]      # amplitude-from-VP
        phases = np.array(list(self.phase_history)[-n:])
        t      = np.arange(n) * self.chirp_dur

        # Breathing-cancelled phase for display
        clean_phases = self._remove_breathing(phases)

        # ── I-Q scatter data ──────────────────────────────────────────────────
        iq_pts   = list(self._phasor_buffer)[-min(200, len(self._phasor_buffer)):]
        iq_real  = [p.real for p in iq_pts]
        iq_imag  = [p.imag for p in iq_pts]
        vp       = self._viewing_center

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

        # ── Panel 1 (top-left): amplitude from optimal viewing position ───────
        ax0.plot(t, amps, lw=0.9, color="#4fc3f7",
                 label="Amplitude from VP (optimal viewing position)")
        ax0.set_ylabel("Amplitude (from VP)")
        ax0.set_title("Face Bin — Amplitude from Optimal Viewing Position")
        ax0.legend(facecolor="#1a1d23", labelcolor="white", fontsize=8)
        ax0.grid(True, alpha=0.15)

        # ── Panel 2 (top-right): I-Q scatter + arc centre ─────────────────────
        ax_iq.scatter(iq_real, iq_imag, s=3, color="#4fc3f7", alpha=0.5,
                      label="I-Q samples")
        if vp is not None:
            ax_iq.plot(vp.real, vp.imag, "r*", ms=12,
                       label=f"VP ({vp.real:.2f}, {vp.imag:.2f})")
            # Draw approximate arc circle
            r_vals = [abs(complex(ix, iq) - vp) for ix, iq in zip(iq_real, iq_imag)]
            r_med  = float(np.median(r_vals)) if r_vals else 0.0
            theta  = np.linspace(0, 2 * np.pi, 200)
            ax_iq.plot(vp.real + r_med * np.cos(theta),
                       vp.imag + r_med * np.sin(theta),
                       "--", color="#ff8800", lw=1, alpha=0.6, label="Fit arc")
        ax_iq.set_xlabel("I (real)")
        ax_iq.set_ylabel("Q (imag)")
        vp_status = "VP ready" if vp is not None else "VP fitting…"
        ax_iq.set_title(f"I-Q Space — {vp_status}")
        ax_iq.legend(facecolor="#1a1d23", labelcolor="white", fontsize=7)
        ax_iq.grid(True, alpha=0.15)
        ax_iq.set_aspect("equal", "datalim")

        # ── Panel 3 (bottom-left): breathing-cancelled phase ──────────────────
        ax1.plot(t, phases,       lw=0.7, color="#888", alpha=0.5,
                 label="Raw phase (unwrapped)")
        ax1.plot(t, clean_phases, lw=0.9, color="#ffb74d",
                 label="Phase (breathing removed)")
        ax1.set_ylabel("Phase (rad)")
        ax1.set_title("Face Bin — Phase (breathing cancelled)")
        ax1.legend(facecolor="#1a1d23", labelcolor="white", fontsize=8)
        ax1.grid(True, alpha=0.15)

        # ── Panel 4 (bottom-right): blink detection feature ───────────────────
        deriv_arr  = np.array(deriv)
        thresh_arr = np.array(thresh)
        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        peaks, _ = find_peaks(deriv_arr, height=thresh_arr, distance=refractory_chirps)
        t2 = np.arange(len(deriv_arr)) * self.chirp_dur

        ax2.plot(t2, deriv_arr,  lw=0.9, color="#a5d6a7",
                 label="Blink feature (VP amp + phase)")
        ax2.plot(t2, thresh_arr, lw=1.2, linestyle="--", color="#ef5350",
                 label="Adaptive threshold")
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


# ── session-state initialisation ──────────────────────────────────────────────

def init_state():
    defaults = {
        "listener": None,
        "running":  False,
        "cfg":      dict(DEFAULTS),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── sidebar — configuration ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏦 Bank Robbery Alert")
    st.caption("Configure detection parameters below.")

    # ── Alert thresholds ──────────────────────────────────────────────────────
    st.subheader("🚨 Alert Settings")
    alert_window = st.slider(
        "Detection window (s)", 1, 30,
        DEFAULTS["alert_window_s"], 1,
        disabled=st.session_state.running,
        help="Rolling time window to count blinks in.",
    )
    alert_thresh = st.slider(
        "Blink threshold (count)", 1, 20,
        DEFAULTS["alert_threshold"], 1,
        disabled=st.session_state.running,
        help="Number of blinks within the window to trigger alert.",
    )
    sensing_delay = st.slider(
        "Startup sensing delay (s)", 1, 10, 2, 1,
        disabled=st.session_state.running,
        help="Seconds to wait before blink counting begins after activation.",
    )

    st.divider()

    # ── Chirp parameters ──────────────────────────────────────────────────────
    st.subheader("Chirp Signal")
    f_start   = st.slider("Start frequency (Hz)", 16000, 20000,
                          DEFAULTS["f_start"], 500,
                          disabled=st.session_state.running)
    f_end     = st.slider("End frequency (Hz)", f_start + 500, 22050,
                          DEFAULTS["f_end"], 500,
                          disabled=st.session_state.running)
    chirp_dur = st.slider("Chirp duration (ms)", 10, 100,
                          int(DEFAULTS["chirp_duration"] * 1000), 5,
                          disabled=st.session_state.running)

    st.subheader("Face Distance Window")
    face_min = st.slider("Min distance (m)", 0.1, 1.0,
                         DEFAULTS["face_dist_min"], 0.05,
                         disabled=st.session_state.running)
    face_max = st.slider("Max distance (m)", face_min + 0.1, 2.0,
                         DEFAULTS["face_dist_max"], 0.05,
                         disabled=st.session_state.running)

    st.subheader("Detection")
    thresh_mult = st.slider("Threshold multiplier", 1.0, 6.0,
                            DEFAULTS["threshold_mult"], 0.1,
                            disabled=st.session_state.running)
    refractory  = st.slider("Refractory period (s)", 0.1, 1.0,
                            DEFAULTS["refractory_s"], 0.05,
                            disabled=st.session_state.running)
    calib_s     = st.slider("Calibration time (s)", 1, 10,
                            int(DEFAULTS["calibration_s"]), 1,
                            disabled=st.session_state.running)
    smooth_win  = st.slider("Smoothing window (chirps)", 1, 10,
                            DEFAULTS["smooth_window"], 1,
                            disabled=st.session_state.running)

    st.divider()
    st.subheader("Audio Device")
    devices     = get_audio_devices()
    duplex_devs = [d for d in devices if d["in"] > 0 and d["out"] > 0]
    dev_labels  = [f"[{d['index']}] {d['name']}" for d in duplex_devs]
    dev_label   = st.selectbox("Duplex device", ["Auto"] + dev_labels,
                               disabled=st.session_state.running)
    if dev_label == "Auto":
        selected_dev_idx = None
    else:
        selected_dev_idx = int(dev_label.split("]")[0].strip("["))

    with st.expander("All audio devices"):
        for d in devices:
            st.text(f"[{d['index']}] in={d['in']} out={d['out']}  {d['name']}")

# ── collect config ─────────────────────────────────────────────────────────────
cfg = dict(
    sample_rate     = DEFAULTS["sample_rate"],
    f_start         = f_start,
    f_end           = f_end,
    chirp_duration  = chirp_dur / 1000.0,
    face_dist_min   = face_min,
    face_dist_max   = face_max,
    threshold_mult  = thresh_mult,
    refractory_s    = refractory,
    calibration_s   = float(calib_s),
    smooth_window   = smooth_win,
    alert_window_s  = alert_window,
    alert_threshold = alert_thresh,
    sensing_delay_s = float(sensing_delay),
)

# ── main page ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="bank-header">
    <div>
        <h1>🏦 BANK ROBBERY ALERT</h1>
        <div class="subtitle">
            ACOUSTIC BLINK DETECTION · SILENT DURESS SYSTEM
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── control row ────────────────────────────────────────────────────────────────
col_start, col_stop, col_reset = st.columns([1, 1, 1])

with col_start:
    if st.button("▶ Activate System", type="primary",
                 disabled=st.session_state.running, use_container_width=True):
        bl = BlinkListenerST(cfg)
        bl.start(dev_index=selected_dev_idx)
        st.session_state.listener = bl
        st.session_state.running  = True
        st.rerun()

with col_stop:
    if st.button("⏹ Deactivate", disabled=not st.session_state.running,
                 use_container_width=True):
        if st.session_state.listener:
            st.session_state.listener.stop()
        st.session_state.running = False
        st.rerun()

with col_reset:
    if st.button("🔄 Reset", disabled=st.session_state.running,
                 use_container_width=True):
        st.session_state.listener = None
        st.rerun()

st.divider()

# ── live dashboard ─────────────────────────────────────────────────────────────
bl: Optional[BlinkListenerST] = st.session_state.listener

if bl is not None:
    # Handle dismiss via session_state — checked at the top of every script run
    if st.session_state.get("_dismiss_alert"):
        bl.dismiss_alert()
        st.session_state["_dismiss_alert"] = False

    # Placeholders for live updates
    alert_placeholder   = st.empty()
    dismiss_placeholder = st.empty()
    status_placeholder  = st.empty()
    metrics_placeholder = st.empty()
    counter_placeholder = st.empty()
    chart_placeholder   = st.empty()

    # Dismiss button rendered exactly ONCE per script run — never duplicated
    if bl.alert_triggered:
        with dismiss_placeholder.container():
            if st.button("✅  Dismiss Alert", key="dismiss_alert_btn"):
                st.session_state["_dismiss_alert"] = True
                st.rerun()
    else:
        dismiss_placeholder.empty()

    def render_frame(bl):
        """Render one frame of the live dashboard."""

        # ── ALERT BANNER ────────────────────────────────────────────────────
        if bl.alert_triggered:
            with alert_placeholder.container():
                st.markdown("""
<div class="alert-banner">
    <h2>🚨 ROBBERY IN PROGRESS 🚨</h2>
    <p>Silent duress signal detected — blink threshold exceeded — contact authorities immediately</p>
</div>
""", unsafe_allow_html=True)
        else:
            alert_placeholder.empty()

        # ── STATUS PILL ─────────────────────────────────────────────────────
        if bl.alert_triggered:
            pill_cls, pill_txt = "status-alert", "🚨 ALERT — NOTIFY AUTHORITIES"
        elif "Calibrating" in bl.status_msg or "Calibrat" in bl.status_msg:
            pill_cls, pill_txt = "status-calibrating", bl.status_msg
        elif bl.face_bin is not None:
            pill_cls, pill_txt = "status-active", bl.status_msg
        else:
            pill_cls, pill_txt = "status-idle", bl.status_msg

        status_placeholder.markdown(
            f'<span class="status-pill {pill_cls}">{pill_txt}</span>',
            unsafe_allow_html=True,
        )

        # ── METRIC ROW ──────────────────────────────────────────────────────
        with metrics_placeholder.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Blinks", bl.blink_count)
            m2.metric("Chirps Processed", bl._chirp_index)
            m3.metric("Face Bin", bl.face_bin if bl.face_bin is not None else "—")
            face_dist_display = (
                f"{bl._range_bins[bl.face_bin]:.2f} m"
                if bl.face_bin is not None and bl.face_bin < len(bl._range_bins)
                else "—"
            )
            m4.metric("Estimated Distance", face_dist_display)
            vp = bl._viewing_center
            vp_display = f"({vp.real:.2f}, {vp.imag:.2f})" if vp is not None else "Fitting…"
            m5.metric("Viewing Position (I, Q)", vp_display)

        # ── LIVE COUNTER + COUNTDOWN ─────────────────────────────────────────
        with counter_placeholder.container():
            st.markdown("#### 🔢 Live Alert Counter")
            c1, c2, c3 = st.columns([1, 1, 2])

            # Blinks in window
            blinks_now = bl.blinks_in_window
            pct        = min(blinks_now / bl.alert_threshold, 1.0)
            warning    = pct >= 0.6
            card_class = "blink-card blink-warning" if warning else "blink-card"

            with c1:
                st.markdown(f"""
<div class="{card_class}">
  <div class="blink-number">{blinks_now}</div>
  <div class="blink-label">Blinks in window</div>
</div>
""", unsafe_allow_html=True)

            with c2:
                secs_left = bl.seconds_until_window_reset()
                st.markdown(f"""
<div class="countdown-wrap">
  <div class="countdown-circle">
    <div class="countdown-number">{secs_left:.1f}</div>
    <div class="countdown-label">window (s)</div>
  </div>
</div>
""", unsafe_allow_html=True)

            with c3:
                st.markdown(
                    f"**Window:** {bl.alert_window_s} s &nbsp;·&nbsp; "
                    f"**Threshold:** {bl.alert_threshold} blinks &nbsp;·&nbsp; "
                    f"**Progress:** {blinks_now} / {bl.alert_threshold}"
                )
                progress_val = int(pct * 100)
                bar_color    = "#cc0000" if pct >= 1.0 else ("#ff8800" if warning else "#1a6a1a")
                st.markdown(f"""
<div style="background:#1a1d23;border-radius:6px;height:22px;overflow:hidden;border:1px solid #2a3040;">
  <div style="background:{bar_color};height:100%;width:{progress_val}%;
              transition:width 0.3s ease;border-radius:6px;"></div>
</div>
""", unsafe_allow_html=True)

    # ── auto-refresh loop ─────────────────────────────────────────────────────
    if st.session_state.running:
        while st.session_state.running:
            render_frame(bl)
            fig = bl.build_figure()
            if fig:
                chart_placeholder.pyplot(fig)
                plt.close(fig)
            else:
                chart_placeholder.info("📊 Collecting signal data — calibrating…")
            time.sleep(0.5)
    else:
        render_frame(bl)
        fig = bl.build_figure()
        if fig:
            chart_placeholder.pyplot(fig)
            plt.close(fig)

else:
    # ── landing / how-it-works ────────────────────────────────────────────────
    st.info("👆 Configure parameters in the sidebar, then press **▶ Activate System** to begin monitoring.")

    st.subheader("How Bank Robbery Alert Works")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**1. Transmit** — Your speaker plays a continuous ultrasonic chirp (18–22 kHz, inaudible to most adults).

**2. Receive** — The microphone captures the room, including eyelid reflections at close range.

**3. Mix (FMCW)** — TX × RX produces a *beat* frequency proportional to reflector distance.
        """)
    with col_b:
        st.markdown("""
**4. Range FFT + Face Bin** — Identifies the bin corresponding to your eyes using **2D I-Q variance** (breathing arcs) — more accurate than the 1D energy method used in breathing detectors.

**5. Optimal Viewing Position** — Fits a circle (Pratt arc method) to the I-Q trajectory formed by breathing/heartbeat interference. The arc centre is the **optimal viewing position**; measuring amplitude from this point maximises the blink transient and cancels breathing drift.

**6. Alert** — If the teller blinks ≥ N times within T seconds, the silent alarm fires.
        """)

    st.subheader("Silent Duress Protocol")
    st.markdown(f"""
- Default trigger: **{DEFAULTS['alert_threshold']} blinks** within **{DEFAULTS['alert_window_s']} seconds**
- Both the count and the window are adjustable in the sidebar
- The alert fires silently — no visible or audible cue for the robber
- Position laptop **~30–60 cm** from the teller's face at camera height
- Sit still during calibration, then blink naturally
    """)

    st.subheader("Setup Tips")
    st.markdown("""
- Use a quiet environment for best detection accuracy
- Adjust the **Face Distance Window** if your face is not being locked
- Reduce **Threshold multiplier** if blinks are missed; increase it to reduce false positives
- The ultrasonic chirp is mostly inaudible to adults under normal conditions
    """)