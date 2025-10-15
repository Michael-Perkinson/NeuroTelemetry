import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def find_shoulders(
    dvdt_segment: np.ndarray,
    start_index: int,
    peak_index: int,
    threshold_factor: float = 0.1,
) -> int:
    if dvdt_segment.size == 0:
        return start_index

    # Cap at the global minimum (most negative dv/dt) in this segment
    global_min_index = int(np.argmin(dvdt_segment))
    if global_min_index <= 0:
        return start_index  # nothing to the left to search

    dvdt_left_of_min = dvdt_segment[:global_min_index]

    # Last +→– zero crossing in the left-of-min region
    zero_crossing_indices: list[int] = []
    for i in range(dvdt_left_of_min.size - 1):
        if dvdt_left_of_min[i] >= 0 and dvdt_left_of_min[i + 1] < 0:
            zero_crossing_indices.append(i + 1)

    if zero_crossing_indices:
        search_start_index = zero_crossing_indices[-1]
    else:
        nearest_to_zero_index = int(np.argmin(np.abs(dvdt_left_of_min)))
        if dvdt_left_of_min[nearest_to_zero_index] > -0.5:
            search_start_index = nearest_to_zero_index
        else:
            search_start_index = 0
            print(
                f"Warning: No zero crossing found for peak at {peak_index}, "
                "using start of segment"
            )

    if search_start_index >= global_min_index:
        return int(start_index + global_min_index - 1)

    search_region = dvdt_segment[search_start_index:global_min_index]
    if search_region.size == 0:
        return int(start_index + search_start_index)

    # Threshold relative to the minimum within the search region
    local_min_index = int(np.argmin(search_region))
    local_min_value = float(search_region[local_min_index])
    threshold_value = local_min_value * threshold_factor

    below_threshold_indices = np.flatnonzero(search_region <= threshold_value)
    if below_threshold_indices.size > 0:
        shoulder_offset = int(below_threshold_indices[0])
    else:
        # Fallback: one sample before the global minimum
        shoulder_offset = (global_min_index - 1) - search_start_index

    return int(start_index + search_start_index + shoulder_offset)


def find_peaks_and_shoulders(time, pressure, dvdt):
    time_win = np.asarray(time.values)
    pressure_win = pressure
    dvdt_win = dvdt

    fs = 500

    peaks, props = find_peaks(-pressure_win, prominence=3, distance=int(0.05 * fs))

    # ---- Adaptive envelope filter (kills those +30 wobble blips) ----
    win_s = 20  # 20-s window
    win = max(5, int(win_s * fs))
    s = pd.Series(pressure_win)

    upper_env = s.rolling(win, center=True, min_periods=1).quantile(0.80).to_numpy()
    lower_env = s.rolling(win, center=True, min_periods=1).quantile(0.20).to_numpy()
    local_amp = np.maximum(upper_env - lower_env, 1e-6)  # avoid zero
    # trough depth vs local upper
    depth = upper_env[peaks] - pressure_win[peaks]

    depth_frac = 0.45  # keep if depth > 45% of local amplitude
    keep_depth = depth >= (depth_frac * local_amp[peaks])

    peaks = peaks[keep_depth]

    # Shoulders
    shoulders = []
    for i, pk in enumerate(peaks):
        start = peaks[i - 1] if i > 0 else 0
        dvdt_seg = dvdt_win[start:pk]
        shoulder = find_shoulders(dvdt_seg, start, pk)
        shoulders.append(shoulder)

    return time_win, pressure_win, dvdt_win, peaks, shoulders


def analyse_peaks(
    time_windows: list[tuple[float, float]],
    pressure_data: pd.DataFrame,
) -> list[tuple[float, float, np.ndarray, pd.Series, np.ndarray, pd.Series]]:
    results = []

    for start_time, end_time in time_windows:
        window_mask = (pressure_data["TimeSinceReference"] >= start_time) & (
            pressure_data["TimeSinceReference"] <= end_time
        )
        smoothed_window = pressure_data.loc[window_mask, "SmoothedPressure"]
        pressure_data_window = pressure_data[window_mask]
        dvdt_window = pressure_data.loc[window_mask, "dvdt"]

        smoothed_array = smoothed_window.to_numpy()
        dvdt_array = dvdt_window.to_numpy()
        time_array = pressure_data_window["TimeSinceReference"]

        _, _, _, peaks, shoulders = find_peaks_and_shoulders(
            time_array, smoothed_array, dvdt_array
        )

        # Ensure NumPy arrays
        peaks = np.asarray(peaks, dtype=int)
        shoulders = np.asarray(shoulders, dtype=int)

        # Truncate to match lengths if necessary
        if len(peaks) != len(shoulders):
            min_len = min(len(peaks), len(shoulders))
            peaks = peaks[:min_len]
            shoulders = shoulders[:min_len]

        # Extract corresponding timestamps
        peak_times = pressure_data_window["TimeSinceReference"].iloc[peaks]
        shoulder_times = pressure_data_window["TimeSinceReference"].iloc[shoulders]

        if len(peak_times) != len(shoulder_times):
            peak_times = peak_times[: len(shoulder_times)]

        results.append(
            (start_time, end_time, peaks, peak_times, shoulders, shoulder_times)
        )

    return results
