"""
BlinkListener — Streamlit Frontend
====================================
A Streamlit UI that talks to the BlinkListener Python backend server.

Run this frontend:
    streamlit run blinklistener_ui.py

Dependencies:
    pip install streamlit requests matplotlib numpy scipy
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st
from scipy.signal import find_peaks

# ── config ────────────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:5000"

DEFAULTS = dict(
    f_start        = 18000,
    f_end          = 22000,
    chirp_duration = 0.04,
    face_dist_min  = 0.3,
    face_dist_max  = 0.9,
    threshold_mult = 2.5,
    refractory_s   = 0.35,
    calibration_s  = 3.0,
    smooth_window  = 3,
)

st.set_page_config(
    page_title="BlinkListener",
    page_icon="👁️",
    layout="wide",
)


# ── server helpers ────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    """Make a request to the backend server, returning (data, error)."""
    try:
        resp = requests.request(method, SERVER_URL + path, timeout=5, **kwargs)
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to server. Is `blinklistener_server.py` running?"
    except Exception as e:
        return None, str(e)


def get_status():
    data, err = api("GET", "/status")
    return data, err


def get_devices():
    data, err = api("GET", "/devices")
    return data or [], err


# ── chart builder ─────────────────────────────────────────────────────────────

def build_figure(history: dict, chirp_dur: float):
    amp    = history.get("amp",    [])
    phase  = history.get("phase",  [])
    deriv  = history.get("deriv",  [])
    thresh = history.get("thresh", [])
    peaks  = history.get("blink_peaks", [])

    n = min(len(amp), len(phase), len(deriv), len(thresh))
    if n < 5:
        return None

    t           = np.arange(n) * chirp_dur
    deriv_arr   = np.array(deriv[:n])
    thresh_arr  = np.array(thresh[:n])
    amp_arr     = np.array(amp[:n])
    phase_arr   = np.array(phase[:n])
    peaks_arr   = [p for p in peaks if p < n]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines[:].set_color("#444")

    axes[0].plot(t, amp_arr, lw=0.9, color="#4fc3f7", label="Amplitude")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Face Bin — Amplitude")
    axes[0].legend(facecolor="#333", labelcolor="white")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, np.unwrap(phase_arr), lw=0.9, color="#ffb74d",
                 label="Phase (unwrapped)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_title("Face Bin — Phase")
    axes[1].legend(facecolor="#333", labelcolor="white")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(t, deriv_arr,  lw=0.9, color="#a5d6a7", label="Feature")
    axes[2].plot(t, thresh_arr, lw=1.2, linestyle="--", color="#ef5350",
                 label="Threshold")
    if peaks_arr:
        axes[2].plot(t[peaks_arr], deriv_arr[peaks_arr], "x", color="yellow",
                     ms=8, mew=2, label=f"Blinks ({len(peaks_arr)})")
    axes[2].set_ylabel("Feature value")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Blink Detection Feature")
    axes[2].legend(facecolor="#333", labelcolor="white")
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


# ── session state ─────────────────────────────────────────────────────────────

if "running" not in st.session_state:
    st.session_state.running = False


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    st.caption("Adjust parameters before starting detection.")

    # Server connection indicator
    _, conn_err = get_status()
    if conn_err:
        st.error(f"🔴 Server offline\n\n{conn_err}")
    else:
        st.success("🟢 Server online")

    st.divider()
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
    devices     = get_devices()
    duplex_devs = [d for d in devices if d["inputs"] > 0 and d["outputs"] > 0]
    dev_labels  = [f"[{d['index']}] {d['name']}" for d in duplex_devs]
    dev_label   = st.selectbox("Duplex device", ["Auto"] + dev_labels,
                                disabled=st.session_state.running)
    dev_index   = (None if dev_label == "Auto"
                   else int(dev_label.split("]")[0].strip("[")))

    with st.expander("All audio devices"):
        for d in devices:
            st.text(f"[{d['index']}] in={d['inputs']} out={d['outputs']}  {d['name']}")

# ── build request payload ─────────────────────────────────────────────────────
payload = dict(
    f_start        = f_start,
    f_end          = f_end,
    chirp_duration = chirp_dur / 1000.0,
    face_dist_min  = face_min,
    face_dist_max  = face_max,
    threshold_mult = thresh_mult,
    refractory_s   = refractory,
    calibration_s  = float(calib_s),
    smooth_window  = smooth_win,
    device_index   = dev_index,
)

# ── main page ─────────────────────────────────────────────────────────────────
st.title("👁️ BlinkListener")
st.caption("Active acoustic FMCW eye-blink detector · Liu et al., IMWUT 2021")

# ── control buttons ───────────────────────────────────────────────────────────
col_start, col_stop, col_reset = st.columns(3)

with col_start:
    if st.button("▶ Start Detection", type="primary",
                 disabled=st.session_state.running, use_container_width=True):
        data, err = api("POST", "/start", json=payload)
        if err:
            st.error(err)
        elif "error" in (data or {}):
            st.error(data["error"])
        else:
            st.session_state.running = True
            st.rerun()

with col_stop:
    if st.button("⏹ Stop", disabled=not st.session_state.running,
                 use_container_width=True):
        data, err = api("POST", "/stop")
        if err:
            st.error(err)
        st.session_state.running = False
        st.rerun()

with col_reset:
    if st.button("🔄 Reset", disabled=st.session_state.running,
                 use_container_width=True):
        api("POST", "/reset")
        st.rerun()

st.divider()

# ── live display ──────────────────────────────────────────────────────────────
status_box    = st.empty()
metric_row    = st.empty()
chart_holder  = st.empty()

if st.session_state.running:
    # ── live polling loop ────────────────────────────────────────────────────
    while st.session_state.running:
        snap, err = get_status()

        if err:
            status_box.error(err)
            time.sleep(1)
            continue

        if not snap.get("running", False):
            # Server stopped on its own
            st.session_state.running = False
            st.rerun()

        # Status banner
        raw_msg = snap.get("status_msg", "")
        if raw_msg.startswith("calibrating:"):
            secs = raw_msg.split(":")[1]
            status_box.info(f"🔊 Calibrating for {secs} s — sit still…")
        elif raw_msg.startswith("face_locked:"):
            dist = raw_msg.split(":")[1]
            status_box.success(f"✅ Face locked at {dist} m — blink away!")
        elif raw_msg == "stopped":
            status_box.warning("⏹️ Stopped")
        else:
            status_box.info(raw_msg or "Running…")

        # Metrics
        face_dist = snap.get("face_dist_m")
        metric_row.columns(4)   # reset
        c1, c2, c3, c4 = metric_row.columns(4)
        c1.metric("Blinks Detected",  snap.get("blink_count", 0))
        c2.metric("Chirps Processed", snap.get("chirp_index", 0))
        c3.metric("Face Bin",         snap.get("face_bin") or "—")
        c4.metric("Estimated Distance",
                  f"{face_dist:.2f} m" if face_dist else "—")

        # Chart
        history = snap.get("history", {})
        cd      = snap.get("chirp_dur", DEFAULTS["chirp_duration"])
        fig     = build_figure(history, cd)
        if fig:
            chart_holder.pyplot(fig)
            plt.close(fig)
        else:
            chart_holder.info("📊 Collecting data…")

        time.sleep(0.5)

else:
    # ── idle or stopped ───────────────────────────────────────────────────────
    snap, err = get_status()

    if err:
        status_box.error(err)
    elif snap:
        raw_msg = snap.get("status_msg", "idle")
        if raw_msg == "stopped":
            status_box.warning(
                f"⏹️ Session stopped. Total blinks: {snap.get('blink_count', 0)}"
            )
        elif raw_msg == "idle":
            status_box.info(
                "👆 Configure parameters in the sidebar and press **▶ Start Detection**."
            )

        # Show final chart if data exists
        history = snap.get("history", {})
        cd      = snap.get("chirp_dur", DEFAULTS["chirp_duration"])
        fig     = build_figure(history, cd)
        if fig:
            chart_holder.pyplot(fig)
            plt.close(fig)

    if not (snap and snap.get("history")):
        # Show explainer when there's no session data yet
        st.subheader("How it works")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**1. Transmit** — Your speaker plays a continuous ultrasonic chirp (18–22 kHz).

**2. Receive** — The microphone records the room, capturing reflections.

**3. Mix (FMCW)** — TX × RX produces a beat frequency ∝ reflector distance.
            """)
        with col_b:
            st.markdown("""
**4. Range FFT** — FFT over each chirp period identifies the face bin.

**5. Phase & amplitude** — An eye-blink produces a transient change in both.

**6. Detect** — Threshold the combined derivative with a refractory gate.
            """)

        st.subheader("Setup tips")
        st.markdown("""
- Place your laptop **~30–60 cm** from your face at camera height.
- The ultrasonic tone is mostly inaudible to adults.
- Sit **still** during calibration, then blink naturally.
- Adjust the **Face Distance Window** in the sidebar if detection is unreliable.
        """)
