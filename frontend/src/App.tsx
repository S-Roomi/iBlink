import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, type DeviceInfo, type StatusResponse } from "./api/client";
import BlinkDetector from "./components/BlinkDetector";
import ChartPanel from "./components/ChartPanel";
import "./App.css";

function App() {
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [deviceIndex, setDeviceIndex] = useState<string>("");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState("");
  const [chartTick, setChartTick] = useState(() => Date.now());
  const [busy, setBusy] = useState(false);
  const [connectionState, setConnectionState] = useState<"idle" | "polling" | "live">("idle");
  const wsRef = useRef<WebSocket | null>(null);

  const chartSrc = useMemo(() => `${api.chartUrl()}?dpi=100&t=${chartTick}`, [chartTick]);
  const connectionLabel = useMemo(() => {
    if (connectionState === "live") return "Live WebSocket stream connected";
    if (connectionState === "polling") return "REST polling fallback active";
    return "Ready to connect";
  }, [connectionState]);

  const closeWs = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionState((current) => (current === "live" ? "polling" : current));
  };

  const loadDevices = async () => {
    try {
      setError("");
      setDevices(await api.getDevices());
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const loadStatus = async () => {
    try {
      const s = await api.getStatus();
      setStatus(s);
      setChartTick(Date.now());
      setError("");
      setConnectionState((current) => (current === "live" ? current : "polling"));
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setStatus(null);
        setConnectionState("idle");
        return;
      }
      setStatus(null);
      setError((e as Error).message);
    }
  };

  const connectWs = () => {
    if (wsRef.current && wsRef.current.readyState <= 1) return;

    const ws = new WebSocket(api.wsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionState("live");
      setError("");
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "status") {
          setStatus(msg.data as StatusResponse);
          setChartTick(Date.now());
          setConnectionState("live");
        } else if (msg.type === "no_session") {
          setStatus(null);
          setConnectionState("idle");
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      setError("WebSocket error");
      setConnectionState("polling");
    };
    ws.onclose = () => {
      wsRef.current = null;
      setConnectionState((current) => (current === "idle" ? current : "polling"));
    };
  };

  const start = async () => {
    try {
      setBusy(true);
      setError("");
      await api.startSession({
        device_index: deviceIndex === "" ? null : Number(deviceIndex),
      });
      await loadStatus();
      connectWs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    try {
      setBusy(true);
      setError("");
      await api.stopSession();
      closeWs();
      setStatus(null);
      setConnectionState("idle");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const dismiss = async () => {
    try {
      setBusy(true);
      setError("");
      await api.dismissAlert();
      await loadStatus();
      wsRef.current?.send(JSON.stringify({ action: "dismiss" }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    void (async () => {
      await loadDevices();
      await loadStatus();
    })();

    const id = setInterval(() => {
      void loadStatus();
    }, 1000);

    return () => {
      clearInterval(id);
      wsRef.current?.close();
    };
  }, []);

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1 className="app-title">iBlink</h1>
          <p className="app-subtitle">A clean React control surface for the FastAPI blink detection API</p>
        </div>

        <div className="status-pill">
          <span
            className={`status-dot ${
              status?.alert_triggered ? "status-dot--alert" : status?.running ? "status-dot--running" : ""
            }`}
          />
          <span>{connectionLabel}</span>
        </div>
      </header>

      <div className="app-grid">
        <section className="panel">
          <BlinkDetector
            devices={devices}
            deviceIndex={deviceIndex}
            onDeviceChange={setDeviceIndex}
            onRefreshDevices={loadDevices}
            onStart={start}
            onStop={stop}
            onDismiss={dismiss}
            status={status}
            error={error}
            busy={busy}
            connectionLabel={connectionLabel}
          />
        </section>

        <section className="panel">
          <ChartPanel
            chartSrc={chartSrc}
            onRefresh={() => setChartTick(Date.now())}
            hasSession={Boolean(status?.running)}
            alertTriggered={Boolean(status?.alert_triggered)}
          />
        </section>
      </div>
    </main>
  );
}

export default App;
