export type DeviceInfo = {
    index: number;
    name: string;
    input_channels: number;
    output_channels: number;
    is_duplex: boolean;
};

export type SessionConfig = {
    sample_rate?: number;
    f_start?: number;
    f_end?: number;
    chirp_duration?: number;
    face_dist_min?: number;
    face_dist_max?: number;
    threshold_mult?: number;
    refractory_s?: number;
    calibration_s?: number;
    smooth_window?: number;
    alert_window_s?: number;
    alert_threshold?: number;
    sensing_delay_s?: number;
    device_index?: number | null;
};

export type StatusResponse = {
    running: boolean;
    status_msg: string;
    blink_count: number;
    chirps_processed: number;
    face_bin: number | null;
    face_distance_m: number | null;
    blinks_in_window: number;
    alert_threshold: number;
    alert_window_s: number;
    seconds_until_reset: number;
    alert_triggered: boolean;
    alert_time: number | null;
};

export type StartResponse = { ok: boolean; message: string };

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
    status: number;

    constructor(status: number, message: string) {
        super(message);
        this.name = "ApiError";
        this.status = status;
    }
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...init,
    });

    if (!res.ok) {
        const text = await res.text();
        let message = `${res.status} ${res.statusText}`;

        if (text) {
            try {
                const parsed = JSON.parse(text) as { detail?: string };
                message = parsed.detail ? `${message} - ${parsed.detail}` : `${message} - ${text}`;
            } catch {
                message = `${message} - ${text}`;
            }
        }

        throw new ApiError(res.status, message);
    }

    if (res.status === 204) return null as T;
    return (await res.json()) as T;
}

export const api = {
    getDevices: () => http<DeviceInfo[]>("/api/devices"),
    startSession: (cfg: SessionConfig = {}) =>
        http<StartResponse>("/api/session/start", {
            method: "POST",
            body: JSON.stringify(cfg),
        }),
    stopSession: () =>
        http<StartResponse>("/api/session/stop", { method: "POST", body: "{}" }),
    dismissAlert: () =>
        http<StartResponse>("/api/session/dismiss", { method: "POST", body: "{}" }),
    getStatus: () => http<StatusResponse>("/api/session/status"),
    chartUrl: () => `${API_BASE}/api/session/chart`,
    wsUrl: () => API_BASE.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live",
};
