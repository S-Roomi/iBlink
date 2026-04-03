"""
Component 2: FMCW Beat Signal & Range FFT
==========================================
Records a short burst of audio while simultaneously playing the chirp,
then shows you the "beat" signal and Range-FFT that converts it into
a distance profile.

What you'll learn:
  - FMCW de-chirp: multiply TX × RX → beat frequency ∝ distance
  - Range FFT: turns beat frequencies into a distance axis
  - How each FFT bin maps to a real-world distance in metres
  - Static clutter: why the wall behind you shows up as a big spike

The math in one line:
    beat_freq = (2 * distance * bandwidth) / (speed_of_sound * chirp_duration)

Run:  python 02_range_fft.py
"""

import time
import queue
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100
F_START        = 18000
F_END          = 22000
CHIRP_DURATION = 0.04          # seconds
N_CHIRPS       = 100           # how many chirps to collect before plotting
SPEED_OF_SOUND = 343.0         # m/s


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chirp(f_start, f_end, duration, sample_rate):
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    c = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method="linear")
    c *= np.hanning(n)
    return c.astype(np.float32)


def compute_range_bins(f_start, f_end, chirp_duration, chirp_n):
    """
    Maps FFT bin index → distance in metres.

    Beat frequency formula (FMCW):
        fb = 2 * d * B / (c * T)
    → d = fb * c * T / (2 * B)

    where B = bandwidth, T = chirp duration, c = speed of sound.
    """
    bw = f_end - f_start
    k  = np.arange(chirp_n // 2 + 1)       # FFT bin indices
    fb = k / chirp_duration                 # beat frequency for each bin (Hz)
    distances = fb * SPEED_OF_SOUND * chirp_duration / (2 * bw)
    return distances


def dechirp_and_fft(rx_chunk, tx_chirp):
    """
    De-chirp: multiply received signal by the known TX chirp.
    The product contains a low-frequency beat component whose frequency
    encodes the round-trip travel time (and thus distance).

    FFT of the beat signal → complex spectrum; magnitude = range profile.
    """
    rx   = rx_chunk[: len(tx_chirp)].astype(np.float32)
    beat = rx * tx_chirp                        # mix-down / de-chirp
    spec = np.fft.rfft(beat, n=len(tx_chirp))  # complex spectrum
    return spec


def main():
    print("=" * 55)
    print("  Component 2: FMCW Beat Signal & Range FFT")
    print("=" * 55)

    chirp   = make_chirp(F_START, F_END, CHIRP_DURATION, SAMPLE_RATE)
    chirp_n = len(chirp)
    bins    = compute_range_bins(F_START, F_END, CHIRP_DURATION, chirp_n)

    print(f"  Chirp samples  : {chirp_n}")
    print(f"  FFT bins       : {len(bins)}")
    print(f"  Range per bin  : {bins[1]:.4f} m")
    print(f"  Max range      : {bins[-1]:.2f} m")
    print()

    # Show what bins correspond to interesting distances
    for dist_target in [0.3, 0.5, 0.9, 1.5, 2.0]:
        idx = int(np.searchsorted(bins, dist_target))
        if idx < len(bins):
            print(f"  {dist_target:.1f} m  →  bin {idx:4d}  "
                  f"(actual {bins[idx]:.3f} m)")
    print()

    tx_buffer = np.tile(chirp * 0.8, 8)
    tx_pos    = [0]
    rx_queue  = queue.Queue()

    def audio_callback(in_data, frame_count, time_info, status):
        # RX: save incoming audio
        rx = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        rx_queue.put(rx)

        # TX: play next slice of chirp
        pos = tx_pos[0] % len(tx_buffer)
        end = pos + frame_count
        out = np.zeros(frame_count, dtype=np.float32)
        if end <= len(tx_buffer):
            out[:] = tx_buffer[pos:end]
        else:
            first       = len(tx_buffer) - pos
            out[:first] = tx_buffer[pos:]
            out[first:] = tx_buffer[: frame_count - first]
        tx_pos[0] = (tx_pos[0] + frame_count) % len(tx_buffer)
        return (out * 32767).astype(np.int16).tobytes(), pyaudio.paContinue

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        output=True,
        frames_per_buffer=chirp_n,
        stream_callback=audio_callback,
    )

    print(f"🎙️  Recording {N_CHIRPS} chirp cycles … sit in front of your laptop")
    
    spectra  = []
    rx_accum = np.zeros(0, dtype=np.float32)

    while len(spectra) < N_CHIRPS:
        try:
            chunk    = rx_queue.get(timeout=0.5)
            rx_accum = np.concatenate([rx_accum, chunk])
            while len(rx_accum) >= chirp_n:
                chirp_rx  = rx_accum[:chirp_n]
                rx_accum  = rx_accum[chirp_n:]
                spec      = dechirp_and_fft(chirp_rx, chirp)
                spectra.append(np.abs(spec))
                if len(spectra) % 10 == 0:
                    print(f"  Collected {len(spectra)}/{N_CHIRPS} chirps …")
        except queue.Empty:
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    print("✅ Done recording.\n")

    # ── Analysis & Plots ──────────────────────────────────────────────────────

    spectra_arr = np.array(spectra)             # shape: (N_CHIRPS, chirp_n//2+1)
    mean_spec   = np.mean(spectra_arr, axis=0)  # average range profile

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("FMCW Range FFT — Component 2", fontsize=14, fontweight="bold")

    # 1. Single beat signal (time domain)
    beat_example = spectra_arr[N_CHIRPS // 2]
    axes[0, 0].plot(bins[:200], beat_example[:200], color="steelblue", lw=0.8)
    axes[0, 0].set_xlabel("Distance (m)")
    axes[0, 0].set_ylabel("Magnitude")
    axes[0, 0].set_title("Single Range Profile (first 5 m)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Mean range profile
    axes[0, 1].plot(bins, mean_spec, color="darkorange", lw=1.2)
    axes[0, 1].axvspan(0.3, 0.9, alpha=0.15, color="green", label="Face window (0.3–0.9 m)")
    axes[0, 1].set_xlabel("Distance (m)")
    axes[0, 1].set_ylabel("Mean Magnitude")
    axes[0, 1].set_title(f"Mean Range Profile ({N_CHIRPS} chirps)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 3)

    # 3. Range-time heatmap (shows how reflections evolve)
    extent = [0, bins[min(300, len(bins)-1)],
              0, N_CHIRPS * CHIRP_DURATION]
    axes[1, 0].imshow(spectra_arr[:, :300], aspect="auto",
                      origin="lower", extent=extent, cmap="inferno")
    axes[1, 0].set_xlabel("Distance (m)")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].set_title("Range-Time Heatmap (closer = stronger static clutter)")

    # 4. Zoom into face-distance window
    lo = int(np.searchsorted(bins, 0.2))
    hi = int(np.searchsorted(bins, 1.1))
    axes[1, 1].plot(bins[lo:hi], mean_spec[lo:hi], color="mediumpurple", lw=1.2)
    peak_idx = lo + int(np.argmax(mean_spec[lo:hi]))
    axes[1, 1].axvline(bins[peak_idx], color="red", linestyle="--", lw=1.5,
                       label=f"Strongest reflector: {bins[peak_idx]:.2f} m")
    axes[1, 1].set_xlabel("Distance (m)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].set_title("Zoomed — Face Distance Window (0.2–1.1 m)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("range_fft_plot.png", dpi=150)
    print("📊 Range-FFT plot saved to range_fft_plot.png")
    plt.show()

    print(f"\n💡 Strongest reflector in face window: bin {peak_idx}  "
          f"→  {bins[peak_idx]:.2f} m from your laptop")
    print("   (This should roughly match how far you were sitting from the microphone)")


if __name__ == "__main__":
    main()
