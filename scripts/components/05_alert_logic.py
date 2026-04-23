"""
Component 5: Sliding-Window Alert Logic
========================================
Demonstrates the alert trigger in isolation — no audio hardware needed.
You simulate blinks by pressing the SPACEBAR; the script maintains a
sliding time window and fires an alert when the count threshold is crossed.

What you'll learn:
  - Sliding window using a deque of timestamps
  - Rolling countdown until the oldest blink falls out of the window
  - How to tune window_seconds vs. threshold for different use-cases
  - The dismiss / reset flow

Run:  python 05_alert_logic.py
Press SPACE to simulate a blink, Q to quit.
"""

import time
import sys
import threading
from collections import deque

# ── Try to import keyboard-input library; fall back to input() ────────────────
try:
    import readchar
    USE_READCHAR = True
except ImportError:
    USE_READCHAR = False

# ── Parameters ────────────────────────────────────────────────────────────────
ALERT_WINDOW_S  = 5    # rolling window in seconds
ALERT_THRESHOLD = 5    # number of blinks to trigger alert


class AlertWindow:
    """
    Maintains a sliding window of blink timestamps.

    Call  record_blink()   to log a blink now.
    Read  blinks_in_window to see the current count.
    Read  alert_triggered  to see if the threshold was crossed.
    Call  dismiss()        to reset the alert.
    """

    def __init__(self, window_s: float, threshold: int):
        self.window_s  = window_s
        self.threshold = threshold

        self._blink_times: deque = deque()
        self.alert_triggered     = False
        self.alert_time: float   = None
        self.blinks_in_window    = 0
        self.total_blinks        = 0

    def record_blink(self) -> bool:
        """Log one blink. Returns True if this blink triggered the alert."""
        now = time.time()
        self._blink_times.append(now)
        self.total_blinks += 1
        self._prune()
        self.blinks_in_window = len(self._blink_times)

        if self.blinks_in_window >= self.threshold and not self.alert_triggered:
            self.alert_triggered = True
            self.alert_time      = now
            return True
        return False

    def _prune(self):
        """Remove blinks that have fallen outside the rolling window."""
        cutoff = time.time() - self.window_s
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

    def seconds_until_reset(self) -> float:
        """Seconds until the oldest blink in the window expires."""
        self._prune()
        self.blinks_in_window = len(self._blink_times)
        if not self._blink_times:
            return self.window_s
        oldest  = self._blink_times[0]
        expires = oldest + self.window_s
        return max(0.0, expires - time.time())

    def dismiss(self):
        self.alert_triggered  = False
        self.alert_time       = None
        self._blink_times.clear()
        self.blinks_in_window = 0


def render(aw: AlertWindow):
    """Print a live status line."""
    aw._prune()
    count     = aw.blinks_in_window
    remaining = aw.seconds_until_reset()
    pct       = min(count / aw.threshold, 1.0)
    bar_len   = 30
    filled    = int(pct * bar_len)
    bar       = "█" * filled + "░" * (bar_len - filled)

    if aw.alert_triggered:
        line = (f"\r🚨 ALERT TRIGGERED!  Total blinks: {aw.total_blinks}  "
                f"[press D to dismiss]          ")
    else:
        line = (f"\r  [{bar}] {count}/{aw.threshold} blinks  "
                f"| window resets in {remaining:.1f} s  "
                f"| total: {aw.total_blinks}  ")

    sys.stdout.write(line)
    sys.stdout.flush()


def display_loop(aw: AlertWindow, stop_event: threading.Event):
    """Background thread: refresh the status display every 100 ms."""
    while not stop_event.is_set():
        render(aw)
        time.sleep(0.1)


def main():
    print("=" * 60)
    print("  Component 5: Sliding-Window Alert Logic Demo")
    print("=" * 60)
    print(f"  Window  : {ALERT_WINDOW_S} s")
    print(f"  Threshold: {ALERT_THRESHOLD} blinks in that window → ALERT")
    print()

    aw = AlertWindow(ALERT_WINDOW_S, ALERT_THRESHOLD)

    stop_event = threading.Event()
    disp_thread = threading.Thread(target=display_loop, args=(aw, stop_event),
                                   daemon=True)
    disp_thread.start()

    if USE_READCHAR:
        print("  Controls: SPACE = blink  |  D = dismiss alert  |  Q = quit")
        print()
        try:
            while True:
                ch = readchar.readchar()
                if ch in (" ", "b"):
                    fired = aw.record_blink()
                    if fired:
                        sys.stdout.write("\n")
                        print("\n  *** 🚨 ALERT FIRED — threshold crossed! ***\n")
                elif ch in ("d", "D"):
                    if aw.alert_triggered:
                        aw.dismiss()
                        sys.stdout.write("\n")
                        print("  ✅ Alert dismissed.\n")
                elif ch in ("q", "Q", "\x03"):
                    break
        except KeyboardInterrupt:
            pass
    else:
        # Fallback: type commands and press Enter
        print("  (Install 'readchar' for a better experience: pip install readchar)")
        print("  Commands: b = blink | d = dismiss | q = quit, then press ENTER")
        print()
        try:
            while True:
                cmd = input().strip().lower()
                if cmd in ("b", ""):
                    fired = aw.record_blink()
                    if fired:
                        print("\n  *** 🚨 ALERT FIRED — threshold crossed! ***\n")
                elif cmd == "d":
                    if aw.alert_triggered:
                        aw.dismiss()
                        print("  ✅ Alert dismissed.\n")
                elif cmd == "q":
                    break
        except (KeyboardInterrupt, EOFError):
            pass

    stop_event.set()
    print(f"\n\n✅ Quit. Total blinks recorded: {aw.total_blinks}")
    print()

    # ── Summary: explain each parameter's effect ─────────────────────────────
    print("=" * 60)
    print("  HOW TO TUNE the alert logic")
    print("=" * 60)
    print(f"""
  window_s  = {ALERT_WINDOW_S} s  → blinks only count if they fall within this window.
              Increase it if you want to allow more time for the signal.
              Decrease it to require blinks to be rapid.

  threshold = {ALERT_THRESHOLD}     → alert fires when this many blinks occur in the window.
              Increase to reduce false positives.
              Decrease to make the system more sensitive.

  The sliding window is implemented with a deque of float timestamps.
  Every time we check, we pop timestamps older than (now - window_s).
  This is O(k) where k = blinks in window ≈ constant → very efficient.
""")


if __name__ == "__main__":
    main()
