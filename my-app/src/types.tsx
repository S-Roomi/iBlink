export interface SessionStatus {
    is_active:              boolean,
    alert_triggered:        boolean,
    last_blink_time?:       string
}

export interface Device {
    id:             number,
    name:           string
}

export interface StatusResponse {
    running:                boolean,
    status_msg:             string,
    blink_count:            number,
    chirps_processed:       number,
    face_bin?:              number,
    face_distance_m:        number,
    blinks_in_window:       number,
    alert_threshold:        number,
    alert_window_s:         number,
    seconds_until_reset:    number,
    alert_triggered:        boolean,
    alert_time:             number

}