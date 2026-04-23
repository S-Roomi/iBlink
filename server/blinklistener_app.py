"""
BlinkListener — Streamlit Web App
==================================
A web interface for the BlinkListener acoustic FMCW eye-blink detector.
Run with:  streamlit run blinklistener_app.py

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
    page_title="BlinkListener",
    page_icon="👁️",
    layout="wide",
)

# ── default parameter values ──────────────────────────────────────────────────
DEFAULTS = dict(
    sample_rate=44100,
    f_start=18000,
    f_end=22000,
    chirp_duration=0.04,
    face_dist_min=0.3,
    face_dist_max=0.9,
    threshold_mult=3.0,
    refractory_s=0.35,
    calibration_s=3.0,
    smooth_window=3,
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
        info = p.get_device_info_by_index(i)
        devices.append({
            "index": i,
            "name": info["name"],
            "in": int(info["maxInputChannels"]),
            "out": int(info["maxOutputChannels"]),
        })
    p.terminate()
    return devices


# ── BlinkListener core (adapted for Streamlit) ────────────────────────────────

class BlinkListenerST:
    """BlinkListener adapted for Streamlit: exposes state for live display."""

    def __init__(self, cfg: dict):
        self.sr            = cfg["sample_rate"]
        self.f_start       = cfg["f_start"]
        self.f_end         = cfg["f_end"]
        self.chirp_dur     = cfg["chirp_duration"]
        self.chirp_n       = int(cfg["chirp_duration"] * cfg["sample_rate"])
        self.threshold_mult = cfg["threshold_mult"]
        self.refractory_s  = cfg["refractory_s"]
        self.calibration_s = cfg["calibration_s"]
        self.smooth_window = cfg["smooth_window"]
        self.face_dist_min = cfg["face_dist_min"]
        self.face_dist_max = cfg["face_dist_max"]

        self.tx_chirp      = make_chirp(self.f_start, self.f_end,
                                        self.chirp_dur, self.sr)
        self._range_bins   = self._compute_range_bins()
        self.face_bin: Optional[int] = None

        maxlen = 500
        self.amp_history    = deque(maxlen=maxlen)
        self.phase_history  = deque(maxlen=maxlen)
        self.deriv_history  = deque(maxlen=maxlen)
        self.thresh_history = deque(maxlen=maxlen)

        self.blink_count  = 0
        self.status_msg   = "Idle"
        self._background_level = None

        refractory_chirps        = int(self.refractory_s / self.chirp_dur)
        self._chirps_since_blink = refractory_chirps
        self._calibration_n      = int(self.calibration_s / self.chirp_dur)
        self._chirp_index        = 0

        self.audio      = pyaudio.PyAudio()
        self.rx_queue   = queue.Queue()
        self._running   = False

        self._tx_buffer = np.tile(self.tx_chirp, 8)
        self._tx_pos    = 0
        self._tx_lock   = threading.Lock()
        self._rx_accum  = np.zeros(0, dtype=np.float32)

    # ── range helpers ──────────────────────────────────────────────────────

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

    # ── FMCW processing ────────────────────────────────────────────────────

    def _process_chirp(self, rx_chunk):
        if len(rx_chunk) < self.chirp_n:
            return None
        rx      = rx_chunk[: self.chirp_n].astype(np.float32)
        beat    = rx * self.tx_chirp
        spec    = np.fft.rfft(beat, n=self.chirp_n)

        if self.face_bin is None and self._chirp_index >= self._calibration_n:
            lo, hi = self._face_bin_range()
            energy = np.abs(spec[lo:hi])
            self.face_bin = lo + int(np.argmax(energy))
            dist = self._range_bins[self.face_bin] if self.face_bin < len(self._range_bins) else 0
            self.status_msg = f"✅ Face locked at {dist:.2f} m — blink away!"

        if self.face_bin is None:
            return None
        return complex(spec[self.face_bin])

    def _detect_blink(self, face_signal):
        amp   = abs(face_signal)
        phase = np.angle(face_signal)

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
                first   = len(self._tx_buffer) - pos
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
                if self._detect_blink(face_sig):
                    self.blink_count += 1

    # ── public interface ───────────────────────────────────────────────────

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
        self.status_msg = f"🔊 Calibrating for {self.calibration_s:.0f} s — sit still…"
        t = threading.Thread(target=self._processing_thread, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.status_msg = "⏹️ Stopped"

    def build_figure(self):
        n = min(len(self.deriv_history), len(self.thresh_history),
                len(self.amp_history),   len(self.phase_history))
        if n < 5:
            return None

        deriv  = list(self.deriv_history)[-n:]
        thresh = list(self.thresh_history)[-n:]
        amps   = list(self.amp_history)[-n:]
        phases = list(self.phase_history)[-n:]
        t      = np.arange(n) * self.chirp_dur

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig.patch.set_facecolor("#0e1117")
        for ax in axes:
            ax.set_facecolor("#1a1d23")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.spines[:].set_color("#444")

        axes[0].plot(t, amps, lw=0.9, color="#4fc3f7", label="Amplitude")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Face Bin — Amplitude")
        axes[0].legend(facecolor="#333", labelcolor="white")
        axes[0].grid(True, alpha=0.2)

        axes[1].plot(t, np.unwrap(phases), lw=0.9, color="#ffb74d", label="Phase (unwrapped)")
        axes[1].set_ylabel("Phase (rad)")
        axes[1].set_title("Face Bin — Phase")
        axes[1].legend(facecolor="#333", labelcolor="white")
        axes[1].grid(True, alpha=0.2)

        deriv_arr  = np.array(deriv)
        thresh_arr = np.array(thresh)
        refractory_chirps = int(self.refractory_s / self.chirp_dur)
        peaks, _ = find_peaks(deriv_arr, height=thresh_arr, distance=refractory_chirps)

        axes[2].plot(t, deriv_arr,  lw=0.9, color="#a5d6a7", label="Feature")
        axes[2].plot(t, thresh_arr, lw=1.2, linestyle="--", color="#ef5350", label="Threshold")
        if len(peaks):
            axes[2].plot(t[peaks], deriv_arr[peaks], "x", color="yellow",
                         ms=8, mew=2, label=f"Blinks ({len(peaks)})")
        axes[2].set_ylabel("Feature value")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Blink Detection Feature")
        axes[2].legend(facecolor="#333", labelcolor="white")
        axes[2].grid(True, alpha=0.2)

        plt.tight_layout()
        return fig


# ── session-state initialisation ─────────────────────────────────────────────

def init_state():
    if "listener" not in st.session_state:
        st.session_state.listener  = None
    if "running" not in st.session_state:
        st.session_state.running   = False
    if "cfg" not in st.session_state:
        st.session_state.cfg       = dict(DEFAULTS)


init_state()

# ── sidebar — configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    st.caption("Adjust parameters before starting detection.")

    st.subheader("Chirp Signal")
    f_start = st.slider("Start frequency (Hz)", 16000, 20000,
                        DEFAULTS["f_start"], 500, disabled=st.session_state.running)
    f_end   = st.slider("End frequency (Hz)",   f_start + 500, 22050,
                        DEFAULTS["f_end"],   500, disabled=st.session_state.running)
    chirp_dur = st.slider("Chirp duration (ms)", 10, 100,
                          int(DEFAULTS["chirp_duration"] * 1000), 5,
                          disabled=st.session_state.running)

    st.subheader("Face Distance Window")
    face_min = st.slider("Min distance (m)", 0.1, 1.0,
                         DEFAULTS["face_dist_min"], 0.05, disabled=st.session_state.running)
    face_max = st.slider("Max distance (m)", face_min + 0.1, 2.0,
                         DEFAULTS["face_dist_max"], 0.05, disabled=st.session_state.running)

    st.subheader("Detection")
    thresh_mult  = st.slider("Threshold multiplier", 1.0, 6.0,
                             DEFAULTS["threshold_mult"], 0.1,
                             disabled=st.session_state.running)
    refractory   = st.slider("Refractory period (s)", 0.1, 1.0,
                             DEFAULTS["refractory_s"], 0.05,
                             disabled=st.session_state.running)
    calib_s      = st.slider("Calibration time (s)", 1, 10,
                             int(DEFAULTS["calibration_s"]), 1,
                             disabled=st.session_state.running)
    smooth_win   = st.slider("Smoothing window (chirps)", 1, 10,
                             DEFAULTS["smooth_window"], 1,
                             disabled=st.session_state.running)

    st.divider()
    st.subheader("Audio Device")
    devices      = get_audio_devices()
    duplex_devs  = [d for d in devices if d["in"] > 0 and d["out"] > 0]
    dev_labels   = [f"[{d['index']}] {d['name']}" for d in duplex_devs]
    dev_label    = st.selectbox("Duplex device", ["Auto"] + dev_labels,
                                disabled=st.session_state.running)
    if dev_label == "Auto":
        selected_dev_idx = None
    else:
        selected_dev_idx = int(dev_label.split("]")[0].strip("["))

    st.divider()
    with st.expander("All audio devices"):
        for d in devices:
            st.text(f"[{d['index']}] in={d['in']} out={d['out']}  {d['name']}")

# ── collect config ────────────────────────────────────────────────────────────
cfg = dict(
    sample_rate    = DEFAULTS["sample_rate"],
    f_start        = f_start,
    f_end          = f_end,
    chirp_duration = chirp_dur / 1000.0,
    face_dist_min  = face_min,
    face_dist_max  = face_max,
    threshold_mult = thresh_mult,
    refractory_s   = refractory,
    calibration_s  = float(calib_s),
    smooth_window  = smooth_win,
)

# ── main page ─────────────────────────────────────────────────────────────────

st.title("👁️ BlinkListener")
st.caption("Active acoustic FMCW eye-blink detector · Liu et al., IMWUT 2021")

# ── control row ───────────────────────────────────────────────────────────────
col_start, col_stop, col_reset = st.columns([1, 1, 1])

with col_start:
    if st.button("▶ Start Detection", type="primary",
                 disabled=st.session_state.running, use_container_width=True):
        bl = BlinkListenerST(cfg)
        bl.start(dev_index=selected_dev_idx)
        st.session_state.listener = bl
        st.session_state.running  = True
        st.rerun()

with col_stop:
    if st.button("⏹ Stop", disabled=not st.session_state.running,
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

# ── live metrics ──────────────────────────────────────────────────────────────
bl: Optional[BlinkListenerST] = st.session_state.listener

if bl is not None:
    status_box = st.info(bl.status_msg)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Blinks Detected", bl.blink_count)
    m2.metric("Chirps Processed", bl._chirp_index)
    m3.metric("Face Bin", bl.face_bin if bl.face_bin is not None else "—")
    face_dist_display = (
        f"{bl._range_bins[bl.face_bin]:.2f} m"
        if bl.face_bin is not None and bl.face_bin < len(bl._range_bins)
        else "—"
    )
    m4.metric("Estimated Distance", face_dist_display)

    st.divider()

    chart_placeholder = st.empty()

    # ── auto-refresh loop ────────────────────────────────────────────────────
    if st.session_state.running:
        while st.session_state.running:
            fig = bl.build_figure()
            if fig:
                chart_placeholder.pyplot(fig)
                plt.close(fig)
            else:
                chart_placeholder.info("📊 Collecting data…")
            time.sleep(0.5)
            # Update metrics in place
            status_box.info(bl.status_msg)
    else:
        # Session stopped — show final plot
        fig = bl.build_figure()
        if fig:
            chart_placeholder.pyplot(fig)
            plt.close(fig)

else:
    st.info("👆 Configure the parameters in the sidebar and press **▶ Start Detection** to begin.")

    st.subheader("How it works")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**1. Transmit** — Your speaker plays a continuous ultrasonic chirp (18–22 kHz, mostly inaudible).

**2. Receive** — The microphone records the room, capturing direct sound and reflections.

**3. Mix (FMCW)** — TX chirp × RX signal produces a *beat* frequency proportional to reflector distance.
        """)
    with col_b:
        st.markdown("""
**4. Range FFT** — An FFT over each chirp period reveals the distance profile. The peak in the face-distance window is your *face bin*.

**5. Phase & amplitude tracking** — An eye-blink causes a sub-mm eyelid displacement, producing a transient change in both amplitude and phase.

**6. Detect** — Threshold the combined derivative; a refractory gate prevents double-counting.
        """)

    st.subheader("Setup tips")
    st.markdown("""
- Place your laptop **~30–60 cm** from your face, roughly at camera height.
- The ultrasonic chirp is mostly inaudible to adults.
- Sit **still** during the calibration period, then blink naturally.
- If detection is unreliable, adjust the **Face Distance Window** in the sidebar.
    """)
