import type { DeviceInfo, StatusResponse } from "../api/client";

type Props = {
  devices: DeviceInfo[];
  deviceIndex: string;
  onDeviceChange: (value: string) => void;
  onRefreshDevices: () => void;
  onStart: () => void;
  onStop: () => void;
  onDismiss: () => void;
  status: StatusResponse | null;
  error: string;
  busy: boolean;
  connectionLabel: string;
};

function Metric({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "alert";
}) {
  return (
    <article className={`metric-card ${tone === "alert" ? "metric-card--alert" : ""}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      {hint ? <span className="metric-hint">{hint}</span> : null}
    </article>
  );
}

export default function BlinkDetector({
  devices,
  deviceIndex,
  onDeviceChange,
  onRefreshDevices,
  onStart,
  onStop,
  onDismiss,
  status,
  error,
  busy,
  connectionLabel,
}: Props) {
  const duplexDevices = devices.filter((device) => device.is_duplex);
  const statusTone = status?.alert_triggered ? "alert" : status?.running ? "running" : "idle";
  const primaryStatus = status?.status_msg ?? "No active session";
  const faceDistance =
    status?.face_distance_m != null ? `${status.face_distance_m.toFixed(2)} m` : "Searching";
  const secondsUntilReset =
    status?.running && status.blinks_in_window > 0
      ? `${Math.ceil(status.seconds_until_reset)}s`
      : "Ready";
  const calibrationState =
    status?.running && status.sensing_enabled === false
      ? `${Math.ceil(status.calibration_remaining_s ?? 0)}s remaining`
      : status?.running
        ? "Monitoring"
        : "Idle";
  const featureValue =
    status?.feature_value != null ? status.feature_value.toFixed(4) : "--";
  const thresholdValue =
    status?.threshold_value != null ? status.threshold_value.toFixed(4) : "--";
  const deviceValue = status?.selected_device_name ?? "System default";

  return (
    <section className="blinkdetector">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Session</p>
          <h2 className="section-title">Controls</h2>
        </div>
      </div>

      <div className="hero-card">
        <div className="hero-actions">
          <button type="button" className="button button--ghost" onClick={onRefreshDevices} disabled={busy}>
            Refresh devices
          </button>
          <button type="button" className="button button--primary" onClick={onStart} disabled={busy}>
            {busy ? "Working..." : "Start detection"}
          </button>
          <button
            type="button"
            className="button button--secondary"
            onClick={onStop}
            disabled={busy || !status?.running}
          >
            Stop session
          </button>
          <button
            type="button"
            className="button button--danger"
            onClick={onDismiss}
            disabled={busy || !status?.alert_triggered}
          >
            Dismiss alert
          </button>
        </div>
        <div className={`status-badge status-badge--${statusTone}`}>
          <span className="status-badge__dot" />
          <span>{connectionLabel}</span>
        </div>
      </div>

      <div className="control-grid">
        <label className="field-card" htmlFor="audio-device-select">
          <span className="field-label">Audio device</span>
          <select
            id="audio-device-select"
            title="Select audio device"
            value={deviceIndex}
            onChange={(e) => onDeviceChange(e.target.value)}
            className="field-select"
            disabled={busy}
          >
            <option value="">Auto-select operating system default</option>
            {devices.map((device) => (
              <option key={device.index} value={device.index}>
                {device.index} - {device.name}
                {device.is_duplex ? " (duplex)" : ""}
              </option>
            ))}
          </select>
        </label>

        <div className="field-card field-card--summary">
          <span className="field-label">Available devices</span>
          <span className="field-help">{devices.length} total, {duplexDevices.length} duplex</span>
          <div className="device-tags">
            {duplexDevices.length > 0 ? (
              duplexDevices.slice(0, 4).map((device) => (
                <span key={device.index} className="device-tag">
                  {device.name}
                </span>
              ))
            ) : (
              <span className="device-tag device-tag--muted">No duplex device found yet</span>
            )}
          </div>
        </div>
      </div>

      {error ? <p className="blinkdetector-error">{error}</p> : null}

      <div className="metrics-grid">
        <Metric label="Session" value={status?.running ? "Active" : "Idle"} hint={primaryStatus} />
        <Metric label="Blinks detected" value={String(status?.blink_count ?? 0)} />
        <Metric
          label="Window count"
          value={String(status?.blinks_in_window ?? 0)}
          hint={`${status?.alert_threshold ?? 0} / ${status?.alert_window_s ?? 0}s`}
        />
        <Metric label="Face distance" value={faceDistance} />
      </div>

      <div className="metrics-grid metrics-grid--secondary">
        <Metric label="Chirps processed" value={String(status?.chirps_processed ?? 0)} />
        <Metric label="Face bin" value={status?.face_bin != null ? String(status.face_bin) : "--"} />
        <Metric label="Alert mode" value={status?.alert_triggered ? "Triggered" : "Monitoring"} />
        <Metric
          label="Reset timer"
          value={secondsUntilReset}
          tone={status?.alert_triggered ? "alert" : "default"}
        />
      </div>

      <div className="metrics-grid metrics-grid--secondary">
        <Metric
          label="Selected device"
          value={deviceValue}
          hint={status?.selected_device_index != null ? `Index ${status.selected_device_index}` : "Auto"}
        />
        <Metric label="Detector state" value={calibrationState} />
        <Metric label="Feature value" value={featureValue} />
        <Metric label="Threshold value" value={thresholdValue} />
      </div>
    </section>
  );
}
