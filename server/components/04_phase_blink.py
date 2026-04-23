"""
Component 4: Phase Extraction & Blink Detection
================================================
Builds on Components 2 & 3.  After calibration locks the face bin,
this script extracts the PHASE of the face-bin phasor on every chirp,
unwraps it, high-pass filters out breathing, computes the derivative,
applies an adaptive threshold, and detects blinks with a refractory gate.

What you'll learn:
  - Why phase (not amplitude) detects micro-movements (sub-mm eyelid shift)
  - Phase unwrapping: why raw phase wraps at ±π and how to fix it
  - Breathing cancellation: high-pass Butterworth filter
  - Adaptive threshold EMA: track the quiet baseline without false positives
  - Refractory gate: don't double-count one blink

Live plot has 3 panels:
  Panel 1 — Face-bin amplitude
  Panel 2 — Unwrapped phase (raw + breathing removed)
  Panel 3 — Phase derivative + adaptive threshold + detected blinks

Run:  python 04_phase_blink.py
Close the plot window to stop.
"""

import time
import queue
import threading
from collections import deque

import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 44100
F_START          = 18000
F_END            = 22000
CHIRP_DURATION   = 0.04
FACE_DIST_MIN    = 0.3
FACE_DIST_MAX    = 0.9
CALIBRATION_S    = 3.0
THRESHOLD_MULT   = 3.0    # how many × baseline to call a blink
REFRACTORY_S     = 0.35   # minimum gap between blinks (seconds)
SMOOTH_WINDOW    = 3      # chirps to smooth the derivative feature
BREATH_CUTOFF_HZ = 1.0    # high-pass cutoff; breathing < 0.5 Hz
SPEED_OF_SOUND   = 343.0
PLOT_HISTORY     = 200    # how many chirps to show in live plot


def make_chirp(f_start, f_end, duration, sample_rate):
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    c = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method="linear")
    c *= np.hanning(n)
    return c.astype(np.float32)


def compute_range_bins(f_start, f_end, chirp_duration, chirp_n):
    bw = f_end - f_start
    k  = np.arange(chirp_n // 2 + 1)
    fb = k / chirp_duration
    return fb * SPEED_OF_SOUND * chirp_duration / (2 * bw)


def remove_breathing(phase_seq, chirp_duration, cutoff_hz=1.0):
    """
    High-pass filter the phase sequence.

    Breathing moves the chest/face by several mm at ~0.1–0.5 Hz.
    A blink shifts the eyelid by ~0.1 mm over ~150–400 ms (2.5–6 Hz).
    High-pass at 1 Hz kills breathing while keeping the blink transient.
    """
    chirp_rate  = 1.0 / chirp_duration
    nyq         = chirp_rate / 2.0
    cutoff_norm = cutoff_hz / nyq

    if len(phase_seq) < 18 or cutoff_norm >= 1.0:
        # Fallback: remove linear trend (handles breathing roughly)
        return phase_seq - np.linspace(phase_seq[0], phase_seq[-1], len(phase_seq))

    try:
        b, a    = signal.butter(2, cutoff_norm, btype="high")
        cleaned = signal.filtfilt(b, a, phase_seq)
    except Exception:
        cleaned = phase_seq - np.linspace(phase_seq[0], phase_seq[-1], len(phase_seq))
    return cleaned


class PhaseBlinkDetector:

    def __init__(self):
        self.chirp   = make_chirp(F_START, F_END, CHIRP_DURATION, SAMPLE_RATE)
        self.chirp_n = len(self.chirp)
        self.bins    = compute_range_bins(F_START, F_END, CHIRP_DURATION, self.chirp_n)

        lo = int(np.searchsorted(self.bins, FACE_DIST_MIN))
        hi = int(np.searchsorted(self.bins, FACE_DIST_MAX))
        self.win_lo = max(lo, 1)
        self.win_hi = min(hi, len(self.bins) - 1)

        # Calibration
        self._calib_spectra  = []
        self._calib_n        = int(CALIBRATION_S / CHIRP_DURATION)
        self._static_clutter = None
        self.face_bin        = None

        # Phase tracking (rolling buffers)
        maxlen = 500
        self._phase_raw      = deque(maxlen=maxlen)
        self._unwrap_accum   = 0.0

        # History for plotting (shorter window)
        self.amp_history    = deque(maxlen=PLOT_HISTORY)
        self.phase_history  = deque(maxlen=PLOT_HISTORY)
        self.deriv_history  = deque(maxlen=PLOT_HISTORY)
        self.thresh_history = deque(maxlen=PLOT_HISTORY)
        self.blink_indices  = deque(maxlen=PLOT_HISTORY)  # indices into history

        # Adaptive background EMA
        self._bg_level = None
        self._bg_alpha = 0.02

        # Refractory / counting
        self.blink_count = 0
        refractory_chirps = int(REFRACTORY_S / CHIRP_DURATION)
        self._chirps_since_blink = refractory_chirps
        self._chirp_idx = 0

        # Status
        self.status = "⏳ Calibrating — sit still…"

        # Thread-safety
        self._lock = threading.Lock()

        # Audio
        self.rx_queue  = queue.Queue()
        self._running  = False
        self._rx_accum = np.zeros(0, dtype=np.float32)
        tx_buf         = np.tile(self.chirp * 0.8, 8)
        self._tx_buf   = tx_buf
        self._tx_pos   = [0]

    # ── Audio start/stop ──────────────────────────────────────────────────────

    def start(self):
        self._running = True
        pa = pyaudio.PyAudio()
        self._pa = pa

        def cb(in_data, frame_count, time_info, status):
            rx = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.rx_queue.put(rx)

            pos = self._tx_pos[0] % len(self._tx_buf)
            end = pos + frame_count
            out = np.zeros(frame_count, dtype=np.float32)
            if end <= len(self._tx_buf):
                out[:] = self._tx_buf[pos:end]
            else:
                first       = len(self._tx_buf) - pos
                out[:first] = self._tx_buf[pos:]
                out[first:] = self._tx_buf[: frame_count - first]
            self._tx_pos[0] = (self._tx_pos[0] + frame_count) % len(self._tx_buf)
            return (out * 32767).astype(np.int16).tobytes(), pyaudio.paContinue

        self._stream = pa.open(
            format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
            input=True, output=True, frames_per_buffer=self.chirp_n,
            stream_callback=cb,
        )
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    # ── Processing loop ───────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                chunk = self.rx_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._rx_accum = np.concatenate([self._rx_accum, chunk])
            while len(self._rx_accum) >= self.chirp_n:
                rx_chunk       = self._rx_accum[: self.chirp_n]
                self._rx_accum = self._rx_accum[self.chirp_n:]
                self._process(rx_chunk)

    def _process(self, rx_chunk):
        beat = rx_chunk[: self.chirp_n] * self.chirp
        spec = np.fft.rfft(beat, n=self.chirp_n)

        # ── Phase 1: calibration ───────────────────────────────────────────────
        if self._static_clutter is None:
            self._calib_spectra.append(spec.copy())
            if len(self._calib_spectra) >= self._calib_n:
                self._static_clutter = np.mean(self._calib_spectra, axis=0)
                clean  = spec - self._static_clutter
                energy = np.abs(clean[self.win_lo: self.win_hi])
                fb     = self.win_lo + int(np.argmax(energy))
                with self._lock:
                    self.face_bin = fb
                    self.status   = (f"✅ Face locked at bin {fb} "
                                     f"({self.bins[fb]:.2f} m)")
            return

        # ── Phase 2: track face-bin phasor ────────────────────────────────────
        clean     = spec - self._static_clutter
        phasor    = complex(clean[self.face_bin])
        is_blink  = self._detect_blink(phasor)

        if is_blink:
            self.blink_count += 1
            with self._lock:
                self.status = f"👁️  Blink #{self.blink_count} detected!"
                self.blink_indices.append(len(self.deriv_history) - 1)

    def _unwrap_push(self, wrapped_phase: float) -> float:
        """Maintain a continuously unwrapped phase (no ±π jumps)."""
        if not self._phase_raw:
            self._unwrap_accum = wrapped_phase
            return wrapped_phase
        prev = np.angle(np.exp(1j * self._phase_raw[-1]))
        diff = wrapped_phase - prev
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        self._unwrap_accum += diff
        return self._unwrap_accum

    def _detect_blink(self, phasor: complex) -> bool:
        amp        = abs(phasor)
        wrapped_ph = np.angle(phasor)
        unwrapped  = self._unwrap_push(wrapped_ph)

        with self._lock:
            self.amp_history.append(amp)
            self.phase_history.append(unwrapped)
        self._phase_raw.append(unwrapped)

        if len(self._phase_raw) < 4:
            return False

        # Step 1: breathing removal
        phase_arr = np.array(self._phase_raw)
        clean_ph  = remove_breathing(phase_arr, CHIRP_DURATION, BREATH_CUTOFF_HZ)

        # Step 2: absolute first-order derivative
        d_phase = abs(clean_ph[-1] - clean_ph[-2])

        # Step 3: smooth
        with self._lock:
            self.deriv_history.append(d_phase)

        if len(self.deriv_history) >= SMOOTH_WINDOW:
            feature = float(np.mean(list(self.deriv_history)[-SMOOTH_WINDOW:]))
        else:
            feature = d_phase

        # Step 4: adaptive background EMA
        if self._bg_level is None:
            self._bg_level = feature
        else:
            threshold_now = self._bg_level * THRESHOLD_MULT
            if feature < threshold_now:
                self._bg_level = ((1 - self._bg_alpha) * self._bg_level
                                  + self._bg_alpha * feature)

        threshold = self._bg_level * THRESHOLD_MULT
        with self._lock:
            self.thresh_history.append(threshold)

        # Step 5: refractory gate + threshold check
        refractory_chirps = int(REFRACTORY_S / CHIRP_DURATION)
        self._chirps_since_blink += 1
        self._chirp_idx += 1

        if feature > threshold and self._chirps_since_blink >= refractory_chirps:
            self._chirps_since_blink = 0
            return True
        return False

    def snapshot(self):
        with self._lock:
            return (
                list(self.amp_history),
                list(self.phase_history),
                list(self.deriv_history),
                list(self.thresh_history),
                list(self.blink_indices),
                self.blink_count,
                self.status,
            )


def main():
    print("=" * 55)
    print("  Component 4: Phase Extraction & Blink Detection")
    print("=" * 55)
    print(f"  Calibration    : {CALIBRATION_S} s — sit still!")
    print(f"  Threshold mult : {THRESHOLD_MULT}×")
    print(f"  Refractory     : {REFRACTORY_S} s")
    print()

    det = PhaseBlinkDetector()
    det.start()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Component 4 — Phase & Blink Detection", fontsize=13,
                 fontweight="bold")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Face Bin — Amplitude")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_title("Unwrapped Phase (grey=raw, orange=breathing removed)")
    axes[2].set_ylabel("|Δphase|")
    axes[2].set_xlabel("Chirp index")
    axes[2].set_title("Blink Feature + Adaptive Threshold")

    (line_amp,)    = axes[0].plot([], [], color="#4fc3f7",   lw=0.9)
    (line_phase,)  = axes[1].plot([], [], color="#888",      lw=0.6, alpha=0.5,
                                  label="Raw unwrapped")
    (line_clean,)  = axes[1].plot([], [], color="#ffb74d",   lw=0.9,
                                  label="Breathing removed")
    (line_deriv,)  = axes[2].plot([], [], color="#a5d6a7",   lw=0.9,
                                  label="|Δphase|")
    (line_thresh,) = axes[2].plot([], [], color="#ef5350",   lw=1.2,
                                  linestyle="--", label="Adaptive threshold")
    (dots_blinks,) = axes[2].plot([], [], "x", color="#ffff00", ms=10,
                                  mew=2, label="Blink")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[2].legend(fontsize=8, loc="upper right")

    title_text = axes[0].text(0.02, 0.97, "", transform=axes[0].transAxes,
                              va="top", ha="left", fontsize=9,
                              color="darkorange", fontweight="bold")
    blink_text = axes[2].text(0.98, 0.97, "", transform=axes[2].transAxes,
                              va="top", ha="right", fontsize=10,
                              color="#ffff00", fontweight="bold")

    def update(_frame):
        amps, phases, derivs, threshs, blink_idx, blink_count, status = det.snapshot()

        n = min(len(amps), len(phases), len(derivs), len(threshs))
        if n < 5:
            return

        xs = np.arange(n)

        line_amp.set_data(xs, amps[-n:])
        axes[0].set_xlim(0, n)
        axes[0].relim(); axes[0].autoscale_view(scalex=False)

        phase_arr = np.array(phases[-n:])
        if n >= 18:
            clean_ph = remove_breathing(phase_arr, CHIRP_DURATION, BREATH_CUTOFF_HZ)
        else:
            clean_ph = phase_arr - phase_arr.mean()
        line_phase.set_data(xs, phase_arr)
        line_clean.set_data(xs, clean_ph)
        axes[1].relim(); axes[1].autoscale_view(scalex=False)

        deriv_arr  = np.array(derivs[-n:])
        thresh_arr = np.array(threshs[-n:])
        line_deriv.set_data(xs, deriv_arr)
        line_thresh.set_data(xs, thresh_arr)
        axes[2].set_ylim(0, max(thresh_arr.max() * 1.4, 0.01))

        # Show blinks
        valid = [i for i in blink_idx if i < n]
        if valid:
            dots_blinks.set_data(valid, deriv_arr[valid])
        else:
            dots_blinks.set_data([], [])

        title_text.set_text(status)
        blink_text.set_text(f"Total blinks: {blink_count}")

    ani = animation.FuncAnimation(fig, update, interval=120, blit=False)

    print("📊 Live plot open — blink naturally, close window to stop.\n")
    plt.tight_layout()
    plt.show()

    det.stop()
    print(f"✅ Stopped. Total blinks detected: {det.blink_count}")


if __name__ == "__main__":
    main()
