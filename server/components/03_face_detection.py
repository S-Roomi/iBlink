"""
Component 3: Face Detection & Live Range Plot
=============================================
Records live audio, runs the FMCW pipeline, and shows a LIVE updating
plot of the range profile.  After a short calibration period it locks
onto your face bin and highlights it in green.

What you'll learn:
  - Static clutter removal: subtract the mean calibration spectrum
  - Face bin locking: pick the highest-energy bin in the distance window
  - Live matplotlib animation driving from a background thread

Calibration phase: sit still for the first few seconds.
After calibration, your face bin will be highlighted.

Run:  python 03_face_detection.py
Close the plot window to stop.
"""

import time
import queue
import threading
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100
F_START        = 18000
F_END          = 22000
CHIRP_DURATION = 0.04
FACE_DIST_MIN  = 0.3    # metres
FACE_DIST_MAX  = 0.9    # metres
CALIBRATION_S  = 3.0    # seconds of calibration (sit still!)
SPEED_OF_SOUND = 343.0


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


class FaceDetector:
    """Minimal FMCW pipeline: calibrate → find face bin → track amplitude."""

    def __init__(self):
        self.chirp   = make_chirp(F_START, F_END, CHIRP_DURATION, SAMPLE_RATE)
        self.chirp_n = len(self.chirp)
        self.bins    = compute_range_bins(F_START, F_END, CHIRP_DURATION, self.chirp_n)

        # Range window indices
        lo = int(np.searchsorted(self.bins, FACE_DIST_MIN))
        hi = int(np.searchsorted(self.bins, FACE_DIST_MAX))
        self.win_lo = max(lo, 1)
        self.win_hi = min(hi, len(self.bins) - 1)

        # Calibration
        calib_n = int(CALIBRATION_S / CHIRP_DURATION)
        self._calib_spectra = []
        self._calib_n       = calib_n
        self._static_clutter = None  # set once calibration done

        # Outputs (updated by processing thread, read by animation)
        self.face_bin   = None
        self.face_dist  = None
        self.status     = "⏳ Calibrating — sit still…"
        self.profile    = np.zeros(len(self.bins))   # latest magnitude profile
        self._lock      = threading.Lock()

        # Audio
        self.rx_queue  = queue.Queue()
        self._running  = False
        self._rx_accum = np.zeros(0, dtype=np.float32)
        tx_buffer      = np.tile(self.chirp * 0.8, 8)
        self._tx_buf   = tx_buffer
        self._tx_pos   = [0]

    def start(self):
        self._running = True
        pa = pyaudio.PyAudio()
        self._pa = pa

        def audio_callback(in_data, frame_count, time_info, status):
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
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            output=True,
            frames_per_buffer=self.chirp_n,
            stream_callback=audio_callback,
        )
        t = threading.Thread(target=self._process_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    def _process_loop(self):
        while self._running:
            try:
                chunk = self.rx_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._rx_accum = np.concatenate([self._rx_accum, chunk])
            while len(self._rx_accum) >= self.chirp_n:
                chirp_rx       = self._rx_accum[: self.chirp_n]
                self._rx_accum = self._rx_accum[self.chirp_n:]
                self._process_one_chirp(chirp_rx)

    def _process_one_chirp(self, rx_chunk):
        beat = rx_chunk[: self.chirp_n] * self.chirp
        spec = np.fft.rfft(beat, n=self.chirp_n)

        # ── Calibration phase ──────────────────────────────────────────────────
        if self._static_clutter is None:
            self._calib_spectra.append(spec.copy())
            if len(self._calib_spectra) >= self._calib_n:
                # Build static clutter template = mean of calibration spectra
                self._static_clutter = np.mean(self._calib_spectra, axis=0)

                # Lock face bin: highest energy in distance window after clutter removal
                clean  = spec - self._static_clutter
                energy = np.abs(clean[self.win_lo : self.win_hi])
                fb     = self.win_lo + int(np.argmax(energy))

                with self._lock:
                    self.face_bin  = fb
                    self.face_dist = self.bins[fb]
                    self.status    = (f"✅ Face locked at bin {fb} "
                                      f"({self.bins[fb]:.2f} m)")
            return

        # ── Live tracking phase ────────────────────────────────────────────────
        clean   = spec - self._static_clutter
        profile = np.abs(clean)

        with self._lock:
            self.profile = profile

    def snapshot(self):
        with self._lock:
            return (self.profile.copy(), self.face_bin,
                    self.face_dist, self.status)


def main():
    print("=" * 55)
    print("  Component 3: Face Detection & Live Range Plot")
    print("=" * 55)
    print(f"  Face window    : {FACE_DIST_MIN} m – {FACE_DIST_MAX} m")
    print(f"  Calibration    : {CALIBRATION_S} s — sit still!")
    print()

    detector = FaceDetector()
    detector.start()

    bins = detector.bins

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Component 3 — Live Face Detection", fontsize=13, fontweight="bold")

    # ── Left: full range profile ──────────────────────────────────────────────
    ax_full = axes[0]
    (line_full,) = ax_full.plot([], [], color="steelblue", lw=0.9)
    (dot_full,)  = ax_full.plot([], [], "ro", ms=8, label="Face bin")
    ax_full.axvspan(FACE_DIST_MIN, FACE_DIST_MAX, alpha=0.12,
                    color="green", label="Search window")
    ax_full.set_xlim(0, 3)
    ax_full.set_ylim(0, 1)
    ax_full.set_xlabel("Distance (m)")
    ax_full.set_ylabel("Magnitude (clutter-removed)")
    ax_full.set_title("Full Range Profile")
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)
    text_status = ax_full.text(0.02, 0.97, "", transform=ax_full.transAxes,
                               va="top", ha="left", fontsize=9,
                               color="darkorange", fontweight="bold")

    # ── Right: zoomed face window ─────────────────────────────────────────────
    ax_zoom = axes[1]
    (line_zoom,) = ax_zoom.plot([], [], color="darkorange", lw=1.2)
    (dot_zoom,)  = ax_zoom.plot([], [], "r^", ms=10, label="Face bin")
    ax_zoom.set_xlim(FACE_DIST_MIN - 0.05, FACE_DIST_MAX + 0.05)
    ax_zoom.set_ylim(0, 1)
    ax_zoom.set_xlabel("Distance (m)")
    ax_zoom.set_ylabel("Magnitude")
    ax_zoom.set_title("Zoomed — Face Window")
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.3)
    text_dist = ax_zoom.text(0.5, 0.95, "", transform=ax_zoom.transAxes,
                             va="top", ha="center", fontsize=11,
                             color="red", fontweight="bold")

    def update(_frame):
        profile, face_bin, face_dist, status = detector.snapshot()

        if profile is None or len(profile) == 0:
            return line_full, dot_full, line_zoom, dot_zoom

        # Normalise for display
        mx = profile.max() if profile.max() > 0 else 1.0
        norm = profile / mx

        # ── Full plot ─────────────────────────────────────────────────────────
        line_full.set_data(bins[: len(norm)], norm)
        ax_full.set_ylim(0, 1.05)

        text_status.set_text(status)

        if face_bin is not None and face_bin < len(norm):
            dot_full.set_data([bins[face_bin]], [norm[face_bin]])
        else:
            dot_full.set_data([], [])

        # ── Zoom plot ─────────────────────────────────────────────────────────
        lo = int(np.searchsorted(bins, FACE_DIST_MIN - 0.05))
        hi = int(np.searchsorted(bins, FACE_DIST_MAX + 0.05))
        lo = max(lo, 0)
        hi = min(hi, len(norm))
        line_zoom.set_data(bins[lo:hi], norm[lo:hi])
        ax_zoom.set_ylim(0, 1.05)

        if face_bin is not None and lo <= face_bin < hi:
            dot_zoom.set_data([bins[face_bin]], [norm[face_bin]])
            text_dist.set_text(f"Face at {face_dist:.2f} m")
        else:
            dot_zoom.set_data([], [])
            text_dist.set_text("Calibrating…")

        return line_full, dot_full, line_zoom, dot_zoom, text_status, text_dist

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)

    print("📊 Live plot open — close the window to stop.\n")
    plt.tight_layout()
    plt.show()

    detector.stop()
    print("✅ Stopped.")


if __name__ == "__main__":
    main()
