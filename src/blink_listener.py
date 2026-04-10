"""
BlinkListener — Acoustic FMCW Eye-Blink Detector
=================================================
Based on: Liu et al., "BlinkListener: Listen to Your Eye Blink Using Your
Smartphone", IMWUT 2021 (ACM DOI 10.1145/3463521)

How it works (different from passive mic listening):
----------------------------------------------------
1. TRANSMIT  : Speaker plays a continuous linear frequency-sweep (chirp)
               aimed at the user's face.  The chirp sweeps from f_start to
               f_end over each period T_chirp.
2. RECEIVE   : Microphone (co-located with speaker) records the room,
               capturing both the direct signal and all reflections.
3. MIX       : Multiply TX chirp × RX signal  →  "beat" signal whose
               frequency is proportional to the round-trip distance of each
               reflector (FMCW ranging principle).
4. RANGE-FFT : FFT over one chirp period  →  one complex bin per reflector.
               The bin whose distance matches the user's face (~0.3-0.7 m)
               is the "eye bin".
5. PHASE/AMP : Track phase and amplitude of the eye bin over time.
               An eye-blink causes a sub-mm displacement of the eyelid,
               producing a transient change in both amplitude and phase.
6. DETECT    : Threshold the combined phase+amplitude derivative to fire a
               blink event, with a refractory gate.

Setup
-----
* Place laptop ~30-60 cm from your face, camera height.
* Run the script — your speakers will emit a faint high-frequency tone
  (18-22 kHz, mostly inaudible to adults).
* Sit still; blink naturally.

Dependencies
------------
pip install numpy scipy pyaudio matplotlib sounddevice
"""

import numpy as np
import pyaudio
import threading
import time
import queue
from typing import Optional, Tuple
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ── chirp parameters ──────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100          # Hz  — standard audio rate
F_START        = 18000          # Hz  — chirp start frequency (mostly inaudible)
F_END          = 22000          # Hz  — chirp end frequency
CHIRP_DURATION = 0.04          # s   — one chirp = 40 ms  →  25 chirps/s
# Range resolution = speed_of_sound / (2 * bandwidth)
# = 343 / (2 * 4000) ≈ 4.3 cm  — fine enough to resolve eye motion

SPEED_OF_SOUND = 343.0          # m/s at ~20 °C

# ── distance window to search for the user's face ────────────────────────────
FACE_DIST_MIN  = 0.3           # m  — closer default for laptop use
FACE_DIST_MAX  = 0.9           # m

# ── detection parameters ─────────────────────────────────────────────────────
THRESHOLD_MULT    = 2.5        # lowered from 3.5 — catches softer natural blinks
REFRACTORY_S      = 0.35        # seconds — minimum gap between two blinks
CALIBRATION_S     = 3.0         # seconds of quiet calibration before detection
SMOOTH_WINDOW     = 3         # chirps — smooth feature over this many chirps to reduce noise

# ── derived constants ─────────────────────────────────────────────────────────
CHIRP_SAMPLES  = int(CHIRP_DURATION * SAMPLE_RATE)
REFRACTORY_CHIRPS = int(REFRACTORY_S / CHIRP_DURATION)


def make_chirp(f_start: float, f_end: float, duration: float,
               sample_rate: int) -> np.ndarray:
    """Generate one linear FMCW chirp, normalised to [-1, 1]."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    chirp = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    # Apply Hanning window to reduce spectral leakage
    chirp *= np.hanning(len(chirp))
    return chirp.astype(np.float32)


class BlinkListener:
    """
    Active acoustic FMCW eye-blink detector.

    The speaker continuously transmits chirps; the microphone records the
    mixed direct+reflected signal.  Each received chirp is correlated against
    the transmitted chirp (= FMCW mixing), then FFT'd to produce a range
    profile.  The complex value of the range bin corresponding to the user's
    face is tracked; blinks appear as transient changes in phase and amplitude.
    """

    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,    
                 f_start: float = F_START,
                 f_end: float = F_END,
                 chirp_duration: float = CHIRP_DURATION,
                 threshold_mult: float = THRESHOLD_MULT):

        self.sr            = sample_rate
        self.f_start       = f_start
        self.f_end         = f_end
        self.chirp_dur     = chirp_duration
        self.chirp_n       = int(chirp_duration * sample_rate)
        self.threshold_mult = threshold_mult

        # Build the reference chirp (TX signal)
        self.tx_chirp = make_chirp(f_start, f_end, chirp_duration, sample_rate)

        # Pre-compute which FFT bin corresponds to each distance
        # beat_freq = (2 * dist * bandwidth) / (speed_of_sound * chirp_duration)
        self._range_bins = self._compute_range_bins()

        # Face bin — will be set during calibration
        self.face_bin: Optional[int] = None

        # Signal history (one complex value per chirp)
        self.phase_history  = deque(maxlen=500)
        self.amp_history    = deque(maxlen=500)
        self.deriv_history  = deque(maxlen=500)
        self.thresh_history = deque(maxlen=500)

        # Detection state
        self.blink_count             = 0
        self._background_level       = None
        self._chirps_since_last_blink = REFRACTORY_CHIRPS  # start ready
        self._calibration_chirps     = int(CALIBRATION_S / chirp_duration)
        self._chirp_index            = 0

        # Audio I/O
        self.audio     = pyaudio.PyAudio()
        self.rx_queue: queue.Queue = queue.Queue()
        self._running  = False

        # TX buffer: repeat the chirp endlessly
        self._tx_buffer  = np.tile(self.tx_chirp, 8)   # 8-chirp lookahead
        self._tx_pos     = 0
        self._tx_lock    = threading.Lock()

        # RX accumulator — collects samples until we have a full chirp
        self._rx_accum   = np.zeros(0, dtype=np.float32)

    # ── range bin mapping ─────────────────────────────────────────────────────

    def _compute_range_bins(self) -> np.ndarray:
        """Return the distance (m) corresponding to each FFT bin."""
        bandwidth   = self.f_end - self.f_start
        n_fft       = self.chirp_n
        # beat frequency for bin k  →  f_beat = k / (chirp_duration)
        # range from beat freq      →  d = f_beat * speed * chirp_duration / (2 * BW)
        bin_indices = np.arange(n_fft // 2 + 1)
        f_beat      = bin_indices / self.chirp_dur
        distances   = f_beat * SPEED_OF_SOUND * self.chirp_dur / (2 * bandwidth)
        return distances

    def _face_bin_range(self) -> tuple[int, int]:
        """Return (bin_lo, bin_hi) for the face distance window."""
        d     = self._range_bins
        lo    = int(np.searchsorted(d, FACE_DIST_MIN))
        hi    = int(np.searchsorted(d, FACE_DIST_MAX))
        return max(lo, 1), min(hi, len(d) - 1)

    # ── FMCW processing ───────────────────────────────────────────────────────

    def _process_chirp(self, rx_chunk: np.ndarray) -> Optional[complex]:
        """
        Process one received chirp:
          1. Mix (multiply) with TX reference
          2. FFT  →  range profile
          3. Return complex value of the eye bin
        """
        if len(rx_chunk) < self.chirp_n:
            return None

        rx = rx_chunk[:self.chirp_n].astype(np.float32)

        # ── FMCW mixing: multiply RX by conjugate of TX ───────────────────────
        # This produces a beat signal at frequency ∝ reflector distance.
        beat = rx * self.tx_chirp

        # ── Range FFT ─────────────────────────────────────────────────────────
        spectrum = np.fft.rfft(beat, n=self.chirp_n)

        # ── Calibration: find the range bin with highest energy in face window ─
        if self.face_bin is None and self._chirp_index >= self._calibration_chirps:
            lo, hi = self._face_bin_range()
            face_energy = np.abs(spectrum[lo:hi])
            self.face_bin = lo + int(np.argmax(face_energy))
            dist = self._range_bins[self.face_bin] if self.face_bin < len(self._range_bins) else 0
            print(f"\n  Face detected at range bin {self.face_bin} "
                  f"(~{dist:.2f} m).  Starting blink detection …")

        if self.face_bin is None:
            return None

        return complex(spectrum[self.face_bin])

    def _detect_blink(self, face_signal: complex) -> bool:
        """
        Decide whether a blink occurred based on amplitude+phase change.
        Returns True once per blink (refractory-gated).
        """
        amp   = abs(face_signal)
        phase = np.angle(face_signal)

        self.amp_history.append(amp)
        self.phase_history.append(phase)

        if len(self.amp_history) < 3:
            return False

        # ── combined derivative: |d(amp)/dt| + scale * |d(phase)/dt| ─────────
        d_amp   = abs(self.amp_history[-1]   - self.amp_history[-2])
        d_phase = abs(self.phase_history[-1] - self.phase_history[-2])
        # Unwrap phase jumps near ±π
        if d_phase > np.pi:
            d_phase = abs(d_phase - 2 * np.pi)

        # Phase is the more reliable blink signal — raised weight from 0.3 → 0.8
        amp_mean = float(np.mean(self.amp_history)) if self.amp_history else 1.0
        raw_feature = d_amp + amp_mean * 0.8 * d_phase

        # ── smooth feature over a short window to reduce single-chirp noise ───
        # A real blink spans ~4-10 chirps; noise spikes are usually 1 chirp wide.
        self.deriv_history.append(raw_feature)
        if len(self.deriv_history) >= SMOOTH_WINDOW:
            feature = float(np.mean(list(self.deriv_history)[-SMOOTH_WINDOW:]))
        else:
            feature = raw_feature

        # ── adaptive background ───────────────────────────────────────────────
        if self._background_level is None:
            self._background_level = feature
        elif feature < self._background_level * self.threshold_mult:
            # Only update background when we're NOT in a blink
            self._background_level = 0.99 * self._background_level + 0.01 * feature

        threshold = self._background_level * self.threshold_mult
        self.thresh_history.append(threshold)

        # ── refractory gate ───────────────────────────────────────────────────
        self._chirps_since_last_blink += 1

        if (feature > threshold
                and self._chirps_since_last_blink >= REFRACTORY_CHIRPS):
            self._chirps_since_last_blink = 0
            return True

        return False

    # ── audio callbacks ───────────────────────────────────────────────────────

    def _tx_callback(self, in_data, frame_count, time_info, status):
        """Duplex callback: output TX chirp, queue RX samples."""
        # ── RX: save incoming audio ───────────────────────────────────────────
        rx = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.rx_queue.put(rx)

        # ── TX: slice the next frame from the repeating chirp buffer ──────────
        with self._tx_lock:
            needed = frame_count
            out    = np.zeros(needed, dtype=np.float32)
            pos    = self._tx_pos % len(self._tx_buffer)
            end    = pos + needed
            if end <= len(self._tx_buffer):
                out[:] = self._tx_buffer[pos:end]
            else:
                first  = len(self._tx_buffer) - pos
                out[:first] = self._tx_buffer[pos:]
                out[first:] = self._tx_buffer[:needed - first]
            self._tx_pos = (self._tx_pos + needed) % len(self._tx_buffer)

        out_bytes = (out * 32767).astype(np.int16).tobytes()
        return (out_bytes, pyaudio.paContinue)

    def _processing_thread(self):
        """Background thread: drain RX queue and process chirps."""
        while self._running or not self.rx_queue.empty():
            try:
                chunk = self.rx_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._rx_accum = np.concatenate([self._rx_accum, chunk])

            # Process as many complete chirps as available
            while len(self._rx_accum) >= self.chirp_n:
                chirp_rx        = self._rx_accum[:self.chirp_n]
                self._rx_accum  = self._rx_accum[self.chirp_n:]
                self._chirp_index += 1

                face_sig = self._process_chirp(chirp_rx)
                if face_sig is None:
                    continue

                if self._detect_blink(face_sig):
                    self.blink_count += 1
                    print(f"\rBlinks detected: {self.blink_count}   ", end="", flush=True)

    # ── public interface ──────────────────────────────────────────────────────

    def start(self):
        """Open duplex audio stream and start detection."""
        # Find a device that supports both input and output
        dev_index = self._find_duplex_device()

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
                stream_callback=self._tx_callback
            )
        except OSError:
            # Fallback: try separate default devices
            print("  Duplex device not found — trying separate in/out devices …")
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sr,
                input=True,
                output=True,
                frames_per_buffer=self.chirp_n,
                stream_callback=self._tx_callback
            )

        self._running = True
        self._proc_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self._proc_thread.start()

        print("=" * 55)
        print("  BlinkListener — FMCW Acoustic Eye-Blink Detector")
        print("=" * 55)
        print(f"  Chirp: {F_START/1000:.0f}–{F_END/1000:.0f} kHz, "
              f"{CHIRP_DURATION*1000:.0f} ms each")
        print(f"  Face range: {FACE_DIST_MIN:.2f}–{FACE_DIST_MAX:.2f} m")
        print(f"\n  Sit ~30-60 cm from your laptop camera.")
        print(f"  Calibrating for {CALIBRATION_S:.0f} s — sit still …")
        time.sleep(CALIBRATION_S + 0.5)
        print("  Blink naturally!  Press Ctrl-C to stop.\n")

        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Tear down the stream."""
        self._running = False
        if hasattr(self, '_proc_thread'):
            self._proc_thread.join(timeout=2.0)
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print(f"\n\nStopped.  Total blinks detected: {self.blink_count}")
        self._plot_results()

    def _find_duplex_device(self) -> Optional[int]:
        """Try to find a device index that supports both input and output."""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and info['maxOutputChannels'] > 0:
                return i
        return None   # let PyAudio use defaults

    # ── visualisation ─────────────────────────────────────────────────────────

    def _plot_results(self):
        if len(self.deriv_history) < 10:
            print("Not enough data to plot.")
            return

        # Truncate all histories to the shortest one to guarantee equal lengths
        n      = min(len(self.deriv_history), len(self.thresh_history),
                     len(self.amp_history),   len(self.phase_history))
        deriv  = list(self.deriv_history)[-n:]
        thresh = list(self.thresh_history)[-n:]
        amps   = list(self.amp_history)[-n:]
        phases = list(self.phase_history)[-n:]
        t      = np.arange(n) * self.chirp_dur

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        axes[0].plot(t, amps, lw=0.8, label='Amplitude')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Face Bin — Amplitude Over Time')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, np.unwrap(phases), lw=0.8, color='orange', label='Phase (unwrapped)')
        axes[1].set_ylabel('Phase (rad)')
        axes[1].set_title('Face Bin — Phase Over Time')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, deriv,  lw=0.8, label='Combined derivative (feature)')
        axes[2].plot(t, thresh, lw=1.2, linestyle='--', color='red', label='Threshold')
        # Mark blink events (peaks above threshold) — both arrays now length n
        thresh_arr = np.array(thresh)
        peaks, _ = find_peaks(deriv, height=thresh_arr, distance=REFRACTORY_CHIRPS)
        axes[2].plot(t[peaks], np.array(deriv)[peaks], 'rx', ms=8, label='Detected blinks')
        axes[2].set_ylabel('Feature value')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Blink Detection Feature')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('blinklistener_results.png', dpi=120)
        print("Plot saved to blinklistener_results.png")
        plt.show()

    # ── range profile snapshot ────────────────────────────────────────────────

    def plot_range_profile(self):
        """
        Record 3 seconds of audio and plot the averaged range profile.
        More averaging = cleaner profile, easier to identify the face peak.
        """
        print("Recording 3 s for range profile — sit still and face the laptop …")
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sr,
            input=True,
            output=True,
            frames_per_buffer=self.chirp_n,
            stream_callback=self._tx_callback
        )
        time.sleep(3.0)          # increased from 1 s → 3 s for better averaging
        stream.stop_stream()
        stream.close()

        # Drain queue
        rx_data = np.zeros(0, dtype=np.float32)
        while not self.rx_queue.empty():
            rx_data = np.concatenate([rx_data, self.rx_queue.get()])

        if len(rx_data) < self.chirp_n:
            print("Not enough data for range profile.")
            return

        # Average range profile over all chirps — more chirps = less noise
        n_chirps = len(rx_data) // self.chirp_n
        profiles = []
        for i in range(n_chirps):
            chunk    = rx_data[i*self.chirp_n:(i+1)*self.chirp_n]
            beat     = chunk * self.tx_chirp
            spectrum = np.abs(np.fft.rfft(beat, n=self.chirp_n))
            profiles.append(spectrum)
        avg_profile = np.mean(profiles, axis=0)
        distances   = self._range_bins

        # ── find the strongest peak inside the face window ────────────────────
        lo, hi = self._face_bin_range()
        window_profile = avg_profile[lo:hi]
        best_local_bin = lo + int(np.argmax(window_profile))
        best_dist      = distances[best_local_bin]
        best_energy    = avg_profile[best_local_bin]

        # ── also find the global strongest peak for reference ─────────────────
        global_best_bin  = int(np.argmax(avg_profile[:len(distances)]))
        global_best_dist = distances[global_best_bin]

        print(f"\n  Strongest peak IN window  : {best_dist:.2f} m  "
              f"(energy {best_energy:.4f})")
        print(f"  Strongest peak OVERALL    : {global_best_dist:.2f} m")
        if global_best_dist < FACE_DIST_MIN or global_best_dist > FACE_DIST_MAX:
            print(f"\n  *** The overall strongest peak is OUTSIDE the window.")
            print(f"      Suggested fix — change constants at top of file:")
            print(f"        FACE_DIST_MIN = {max(0.1, global_best_dist - 0.15):.2f}")
            print(f"        FACE_DIST_MAX = {global_best_dist + 0.15:.2f}")
        else:
            print(f"\n  OK  Face peak is inside the window — no changes needed.")
            print(f"      For a tighter lock, you could set:")
            print(f"        FACE_DIST_MIN = {max(0.1, best_dist - 0.15):.2f}")
            print(f"        FACE_DIST_MAX = {best_dist + 0.15:.2f}")

        plt.figure(figsize=(10, 4))
        plt.plot(distances[:len(avg_profile)], avg_profile, lw=0.8)
        plt.axvspan(FACE_DIST_MIN, FACE_DIST_MAX, alpha=0.15, color='green',
                    label=f'Face search window ({FACE_DIST_MIN}–{FACE_DIST_MAX} m)')
        # Mark the best peak inside the window
        plt.axvline(best_dist, color='blue', linestyle='--', lw=1.2,
                    label=f'Best in-window peak ({best_dist:.2f} m)')
        # Mark global peak if different
        if abs(global_best_dist - best_dist) > 0.05:
            plt.axvline(global_best_dist, color='red', linestyle=':', lw=1.2,
                        label=f'Global peak ({global_best_dist:.2f} m) — outside window!')
        plt.xlabel('Distance (m)')
        plt.ylabel('Reflected energy')
        plt.title(f'FMCW Range Profile  ({n_chirps} chirps averaged)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2.0)
        plt.tight_layout()
        plt.savefig('range_profile.png', dpi=120)
        print("\n  Plot saved to range_profile.png")
        plt.show()
        plt.tight_layout()
        plt.savefig('range_profile.png', dpi=120)
        print("Range profile saved to range_profile.png")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════

def list_audio_devices():
    """Print all available audio devices."""
    p = pyaudio.PyAudio()
    print("\nAvailable audio devices:")
    print(f"  {'Idx':<5} {'In':<5} {'Out':<5} Name")
    print("  " + "-" * 50)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"  {i:<5} {int(info['maxInputChannels']):<5} "
              f"{int(info['maxOutputChannels']):<5} {info['name']}")
    p.terminate()


def main():
    print("=" * 55)
    print("  BlinkListener — Active Acoustic Eye-Blink Detector")
    print("=" * 55)
    print("\n  Based on: Liu et al., IMWUT 2021")
    print("  Method: FMCW chirp sonar — speaker transmits,")
    print("          mic detects changes in face reflection.\n")
    print("  Options:")
    print("  1. Start blink detection")
    print("  2. Show range profile (verify face detection)")
    print("  3. List audio devices")
    print("  4. Exit")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == '1':
        bl = BlinkListener()
        bl.start()

    elif choice == '2':
        bl = BlinkListener()
        bl.plot_range_profile()

    elif choice == '3':
        list_audio_devices()

    elif choice == '4':
        print("Bye!")

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()