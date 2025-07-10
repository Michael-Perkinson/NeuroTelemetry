import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple


def find_shoulders(
    dvdt_seg: np.ndarray,
    start_idx: int,
    peak_idx: int,
    threshold_factor: float = 0.1
) -> int:
    """Method 1: First derivative threshold approach with zero-crossing constraint"""
    if len(dvdt_seg) == 0:
        return start_idx

    zero_crossings: List[int] = []
    for i in range(len(dvdt_seg) - 1):
        if dvdt_seg[i] >= 0 and dvdt_seg[i + 1] < 0:
            zero_crossings.append(i + 1)

    if not zero_crossings:
        closest_to_zero_idx = np.argmin(np.abs(dvdt_seg))
        if dvdt_seg[closest_to_zero_idx] > -0.5:
            zero_crossings = [closest_to_zero_idx]

    if zero_crossings:
        search_start = zero_crossings[-1]
        dvdt_search = dvdt_seg[search_start:]
        search_offset = search_start
    else:
        print(
            f"Warning: No zero crossing found for peak at {peak_idx}, using full segment")
        dvdt_search = dvdt_seg
        search_offset = 0

    if len(dvdt_search) == 0:
        return start_idx + search_offset

    min_slope = np.min(dvdt_search)
    min_slope_idx = np.argmin(dvdt_search)
    threshold = min_slope * threshold_factor
    steep_points = np.where(dvdt_search <= threshold)[0]

    if len(steep_points) > 0:
        shoulder_rel = steep_points[0]
    else:
        print("Used fallback to steepest point")
        shoulder_rel = min_slope_idx

    return start_idx + search_offset + shoulder_rel


def find_peaks_and_shoulders(
    time: pd.Series,
    pressure: np.ndarray,
    dvdt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    
    time_win = time.values
    pressure_win = pressure
    dvdt_win = dvdt

    peaks, _ = find_peaks(-pressure_win, prominence=3, distance=100)

    shoulders: List[int] = []
    for i, pk in enumerate(peaks):
        start = peaks[i - 1] if i > 0 else 0
        dvdt_seg = dvdt_win[start:pk]
        shoulder = find_shoulders(dvdt_seg, start, pk)
        shoulders.append(shoulder)

    return time_win, pressure_win, dvdt_win, peaks, shoulders


def analyse_shoulders(
    time_windows: List[Tuple[float, float]],
    pressure_data: pd.DataFrame,
    dvdt: pd.Series
) -> List[
    Tuple[
        float, float, np.ndarray, pd.Series, np.ndarray, pd.Series
    ]
]:
    results = []

    for start_time, end_time in time_windows:
        window_mask = (pressure_data['TimeSinceReference'] >= start_time) & (
            pressure_data['TimeSinceReference'] <= end_time)
        smoothed_window = pressure_data.loc[window_mask, 'SmoothedPressure']
        pressure_data_window = pressure_data[window_mask]
        dvdt_window = dvdt[window_mask]

        smoothed_array = smoothed_window.to_numpy()
        dvdt_array = dvdt_window.to_numpy()
        time_array = pressure_data_window['TimeSinceReference'].to_numpy()

        _, _, _, peaks, shoulders = find_peaks_and_shoulders(
            time_array, smoothed_array, dvdt_array
        )

        peaks = np.array(peaks)
        shoulders = np.array(shoulders)

        if len(peaks) != len(shoulders):
            min_len = min(len(peaks), len(shoulders))
            peaks = peaks[:min_len]
            shoulders = shoulders[:min_len]

        peak_times = pressure_data_window['TimeSinceReference'].iloc[peaks]
        shoulder_times = pressure_data_window['TimeSinceReference'].iloc[shoulders]

        if len(peak_times) != len(shoulder_times):
            peak_times = peak_times[:len(shoulder_times)]

        results.append((
            start_time,
            end_time,
            peaks,
            peak_times,
            shoulders,
            shoulder_times
        ))

    return results
