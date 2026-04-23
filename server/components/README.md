# BlinkListener — Learning Component Scripts

These five scripts break the `bank_alert.py` system into individual,
runnable pieces so you can study and demonstrate each stage independently.

## Prerequisites

```bash
pip install numpy scipy pyaudio matplotlib
pip install readchar   # optional — makes Component 5 nicer
```

---

## Script Map

```
01_chirp_emitter.py       TX only  — generates & plays the FMCW chirp
02_range_fft.py           TX + RX  — de-chirp, Range FFT, distance profile
03_face_detection.py      + clutter removal + face bin locking (live plot)
04_phase_blink.py         + phase extraction + blink detection (live plot)
05_alert_logic.py         alert window logic — NO audio needed (keyboard sim)
```

---

## Component 1 — Chirp Emitter (`01_chirp_emitter.py`)

**Concepts:** chirp signal, Hanning window, PyAudio callback, seamless loop buffer.

Plays a continuous 18–22 kHz frequency sweep through your speaker.
Also saves `chirp_plot.png` — the waveform and spectrogram of one chirp.

```
Run:   python 01_chirp_emitter.py
Stop:  Ctrl+C
```

**Key function:** `make_chirp(f_start, f_end, duration, sample_rate)`

---

## Component 2 — Range FFT (`02_range_fft.py`)

**Concepts:** FMCW de-chirp (TX × RX), beat frequency, Range FFT, bin→distance mapping.

Records 100 chirp cycles, then shows four plots:
- Single range profile
- Mean range profile with the face-distance window shaded
- Range-time heatmap
- Zoomed face window with strongest reflector marked

```
Run:   python 02_range_fft.py   (sit in front of your laptop)
```

**Key formula:**
```
distance = beat_freq × c × T / (2 × BW)
```

---

## Component 3 — Face Detection (`03_face_detection.py`)

**Concepts:** static clutter removal, face bin locking, live matplotlib animation.

After `CALIBRATION_S` seconds of sitting still, the system builds a mean
clutter template, subtracts it, and locks onto the highest-energy bin
in the `0.3–0.9 m` distance window.

Live plot:
- Left panel: full range profile (green = search window, red dot = face bin)
- Right panel: zoomed view with distance label

```
Run:   python 03_face_detection.py   (sit still for first 3 s)
Stop:  close the plot window
```

---

## Component 4 — Phase & Blink Detection (`04_phase_blink.py`)

**Concepts:** complex phasor phase, phase unwrapping, Butterworth HP filter,
adaptive threshold EMA, refractory gate.

Three-panel live plot:
1. Face-bin amplitude
2. Raw + breathing-cancelled unwrapped phase
3. Phase derivative + adaptive threshold + detected blink markers (✕)

```
Run:   python 04_phase_blink.py   (blink naturally after calibration)
Stop:  close the plot window
```

**Why phase, not amplitude?**
An eyelid moves ~0.1–0.5 mm.  Phase is sensitive to sub-wavelength
displacements; amplitude is not.  Phase shift Δφ = 4π·Δd / λ_eff.

---

## Component 5 — Alert Logic (`05_alert_logic.py`)

**Concepts:** sliding window (deque of timestamps), countdown timer, dismiss flow.

No microphone or speaker needed — press SPACE to simulate a blink.

```
Run:   python 05_alert_logic.py
Keys:  SPACE = blink | D = dismiss | Q = quit
```

Shows a live progress bar and triggers the alert when the threshold is crossed.
Explains how to tune `window_s` and `threshold` at the end.

---

## Full System Pipeline (for reference)

```
Speaker ──► FMCW chirp (18–22 kHz sweep, 40 ms)
                │
Mic     ──► RX audio
                │
          TX × RX  ──► beat signal
                │
           Range FFT  ──► complex spectrum
                │
     subtract calibration mean  ──► clutter-free spectrum
                │
         pick face bin (highest energy in 0.3–0.9 m)
                │
         extract complex phase  ──► unwrap  ──► HP filter (remove breathing)
                │
         |Δphase| feature  ──► smooth  ──► adaptive threshold
                │
         refractory gate  ──► blink event
                │
         sliding time window  ──► blink count  ──► ALERT
```
