"""
Component 1: Chirp Emitter
==========================
Generates and plays a continuous ultrasonic FMCW (Frequency-Modulated
Continuous Wave) chirp through your speaker — the "transmit" side of
the BlinkListener radar.

What you'll learn:
  - What a chirp signal is (a tone that sweeps from f_start → f_end)
  - Why a Hanning window is applied (smooth edges → less spectral leakage)
  - How PyAudio streams audio from a callback function
  - Why we repeat the chirp in a tile buffer for seamless looping

Run:  python 01_chirp_emitter.py
Press Ctrl+C to stop.
"""

import time
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy import signal

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100   # Hz — standard CD quality
F_START       = 18000   # Hz — chirp sweeps from here …
F_END         = 22000   # Hz  … to here  (mostly above human hearing ~20 kHz)
CHIRP_DURATION = 0.04   # seconds (40 ms per chirp)
VOLUME        = 0.8     # 0.0 – 1.0


def make_chirp(f_start, f_end, duration, sample_rate):
    """
    Build one chirp waveform.

    A chirp is a sinusoid whose instantaneous frequency increases linearly
    from f_start to f_end over the duration.  The Hanning window tapers the
    edges to zero so repeated chirps join without clicks.
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # scipy.signal.chirp generates a linear frequency sweep
    c = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method="linear")

    # Hanning window: avoids amplitude discontinuity at chirp boundaries
    c *= np.hanning(n_samples)

    return c.astype(np.float32)


def plot_chirp(chirp, sample_rate):
    """Show the chirp waveform and its spectrogram so you can see the sweep."""
    t = np.linspace(0, len(chirp) / sample_rate, len(chirp))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("FMCW Chirp Signal", fontsize=14, fontweight="bold")

    # Waveform
    axes[0].plot(t * 1000, chirp, color="steelblue", lw=0.8)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Chirp Waveform (time domain)")
    axes[0].grid(True, alpha=0.3)

    # Spectrogram — shows the frequency sweep visually
    axes[1].specgram(chirp, Fs=sample_rate, NFFT=256, noverlap=200,
                     cmap="plasma")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Spectrogram — see the frequency sweep from 18 kHz → 22 kHz")
    axes[1].set_ylim(15000, 24000)

    plt.tight_layout()
    plt.savefig("chirp_plot.png", dpi=150)
    print("📊 Chirp plot saved to chirp_plot.png")
    plt.show()


def main():
    print("=" * 55)
    print("  Component 1: FMCW Chirp Emitter")
    print("=" * 55)

    chirp = make_chirp(F_START, F_END, CHIRP_DURATION, SAMPLE_RATE)
    chirp_n = len(chirp)

    print(f"  Sample rate    : {SAMPLE_RATE} Hz")
    print(f"  Chirp sweep    : {F_START} Hz → {F_END} Hz")
    print(f"  Chirp duration : {CHIRP_DURATION * 1000:.0f} ms")
    print(f"  Samples/chirp  : {chirp_n}")
    print(f"  Chirps/second  : {1 / CHIRP_DURATION:.1f}")
    print()

    # Plot before starting audio
    plot_chirp(chirp, SAMPLE_RATE)

    # Pre-tile the chirp into a large buffer so the callback never gaps
    tx_buffer = np.tile(chirp * VOLUME, 8)
    tx_pos    = [0]   # mutable container so the closure can update it

    def audio_callback(in_data, frame_count, time_info, status):
        """
        PyAudio calls this function every time it needs more audio samples.
        We slice the next `frame_count` samples from tx_buffer, wrapping
        around if we reach the end (seamless loop).
        """
        pos = tx_pos[0] % len(tx_buffer)
        end = pos + frame_count

        out = np.zeros(frame_count, dtype=np.float32)
        if end <= len(tx_buffer):
            out[:] = tx_buffer[pos:end]
        else:
            first = len(tx_buffer) - pos
            out[:first] = tx_buffer[pos:]
            out[first:] = tx_buffer[: frame_count - first]

        tx_pos[0] = (tx_pos[0] + frame_count) % len(tx_buffer)
        return (out * 32767).astype(np.int16).tobytes(), pyaudio.paContinue

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=chirp_n,
        stream_callback=audio_callback,
    )

    print("🔊 Emitting ultrasonic chirp — press Ctrl+C to stop …")
    try:
        while stream.is_active():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\n✅ Stopped.")


if __name__ == "__main__":
    main()
