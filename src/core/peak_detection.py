from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

DEFAULT_PROMINENCE = 3.0
DEFAULT_MIN_PEAK_DISTANCE = 50
DEFAULT_SHOULDER_THRESHOLD_FACTOR = 0.1


# Small helpers
def extract_window(
    df: pd.DataFrame,
    start_time: float,
    end_time: float,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    Slice a telemetry window and return (time_series, smoothed_array, dvdt_array).
    Never raises; returns empty arrays if the window has no data.
    """
    mask = (df["TimeSinceReference"] >= start_time) & (
        df["TimeSinceReference"] <= end_time)
    win = df.loc[mask]

    time_series = win["TimeSinceReference"]
    smoothed = win["SmoothedPressure"].to_numpy(
        dtype=float, copy=False) if "SmoothedPressure" in win else np.array([], dtype=float)
    dvdt = win["dvdt"].to_numpy(
        dtype=float, copy=False) if "dvdt" in win else np.array([], dtype=float)

    return time_series, smoothed, dvdt


def detect_peaks(pressure: np.ndarray,
                 prominence: float = DEFAULT_PROMINENCE,
                 distance: int = DEFAULT_MIN_PEAK_DISTANCE) -> np.ndarray:
    """
    Detect peaks for *inverted* pressure (i.e., minima in original pressure).
    Returns integer indices (possibly empty).
    """
    if pressure.size == 0:
        return np.empty(0, dtype=int)

    # We find minima of pressure by finding peaks in (-pressure)
    peaks, _ = find_peaks(-pressure, prominence=prominence, distance=distance)
    return peaks.astype(int, copy=False)


def find_zero_crossings_left_to_min(dvdt_segment: np.ndarray) -> list[int]:
    """
    Return indices of +→– zero crossings in dv/dt for the portion left of its global minimum.
    If no valid left part exists, returns [].
    """
    if dvdt_segment.size == 0:
        return []

    global_min_idx = int(np.argmin(dvdt_segment))
    if global_min_idx <= 0:
        return []

    left = dvdt_segment[:global_min_idx]
    crossings: list[int] = []
    # +to- means prev >= 0 and next < 0
    for i in range(left.size - 1):
        if left[i] >= 0.0 and left[i + 1] < 0.0:
            crossings.append(i + 1)
    return crossings


def choose_shoulder_start_index(
    dvdt_segment: np.ndarray, zero_crossings: list[int]
) -> int:
    """
    Pick the search start index for shoulder detection:
    - Prefer last +→– zero crossing.
    - Otherwise choose point closest to zero if it's not too negative.
    - Otherwise fall back to 0.
    Returns an index within [0, len(dvdt_segment)).
    """
    if dvdt_segment.size == 0:
        return 0

    global_min_idx = int(np.argmin(dvdt_segment))
    left = dvdt_segment[:global_min_idx] if global_min_idx > 0 else dvdt_segment

    if zero_crossings:
        start_idx = zero_crossings[-1]
    else:
        # closest to zero
        if left.size > 0:
            nz = int(np.argmin(np.abs(left)))
            start_idx = nz if left[nz] > -0.5 else 0
        else:
            start_idx = 0

    # Clamp to left side of global minimum
    return int(min(start_idx, max(global_min_idx - 1, 0)))


def find_shoulders_for_peaks(
    dvdt: np.ndarray,
    peaks: np.ndarray,
    threshold_factor: float = DEFAULT_SHOULDER_THRESHOLD_FACTOR,
) -> np.ndarray:
    """
    For each peak index, scan dv/dt between previous peak (exclusive) and the peak,
    and find the first sample below (threshold_factor * local_min) after a chosen start point.
    Returns integer indices (aligned with 'peaks'), truncated/scaled as needed.
    """
    if dvdt.size == 0 or peaks.size == 0:
        return np.empty(0, dtype=int)

    shoulders: list[int] = []

    for i, pk in enumerate(peaks):
        start = peaks[i - 1] if i > 0 else 0
        if start >= pk:
            shoulders.append(max(pk - 1, 0))
            continue

        seg = dvdt[start:pk]
        if seg.size == 0:
            shoulders.append(max(pk - 1, 0))
            continue

        zeros = find_zero_crossings_left_to_min(seg)
        search_start = choose_shoulder_start_index(seg, zeros)

        global_min_idx = int(np.argmin(seg))
        # Guard for degenerate ranges
        if search_start >= global_min_idx:
            shoulders.append(start + max(global_min_idx - 1, 0))
            continue

        search_region = seg[search_start:global_min_idx]
        if search_region.size == 0:
            shoulders.append(start + search_start)
            continue

        local_min_idx = int(np.argmin(search_region))
        local_min_val = float(search_region[local_min_idx])
        thresh = local_min_val * threshold_factor  # negative number scaled

        below = np.flatnonzero(search_region <= thresh)
        if below.size > 0:
            shoulder_offset = int(below[0])
        else:
            # fallback: one sample before the global min (but within window)
            shoulder_offset = max(global_min_idx - 1 - search_start, 0)

        shoulders.append(start + search_start + shoulder_offset)

    return np.asarray(shoulders, dtype=int)


# Compatibility wrapper kept (name + return shape)
def find_peaks_and_shoulders(
    time: pd.Series,
    pressure: np.ndarray,
    dvdt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    KEPT FOR BACKWARD COMPATIBILITY.
    Returns: (time_array, pressure_array, dvdt_array, peaks_indices, shoulders_list)
    """
    time_arr = np.asarray(time.values)
    pressure_arr = np.asarray(pressure, dtype=float)
    dvdt_arr = np.asarray(dvdt, dtype=float)

    peaks = detect_peaks(
        pressure_arr, prominence=DEFAULT_PROMINENCE, distance=DEFAULT_MIN_PEAK_DISTANCE)
    shoulders = find_shoulders_for_peaks(
        dvdt_arr, peaks, threshold_factor=DEFAULT_SHOULDER_THRESHOLD_FACTOR).tolist()

    return time_arr, pressure_arr, dvdt_arr, peaks, shoulders


def analyse_peaks(
    time_windows: list[tuple[float, float]],
    pressure_data: pd.DataFrame,
) -> list[tuple[float, float, np.ndarray, pd.Series, np.ndarray, pd.Series]]:
    """
    For each (start, end) window:
      - extract window arrays
      - detect peaks and shoulders
      - align their counts
      - return (start, end, peaks_idx, peak_times_series, shoulders_idx, shoulder_times_series)
    """
    results: list[tuple[float, float, np.ndarray,
                        pd.Series, np.ndarray, pd.Series]] = []

    for start_time, end_time in time_windows:
        # 1) window extraction
        time_series, smoothed_array, dvdt_array = extract_window(
            pressure_data, start_time, end_time)

        # 2) detection
        _, _, _, peaks, shoulders_list = find_peaks_and_shoulders(
            time_series, smoothed_array, dvdt_array
        )
        peaks_idx = np.asarray(peaks, dtype=int)
        shoulders_idx = np.asarray(shoulders_list, dtype=int)

        # 3) align lengths defensively
        if peaks_idx.size != shoulders_idx.size:
            n = min(peaks_idx.size, shoulders_idx.size)
            peaks_idx = peaks_idx[:n]
            shoulders_idx = shoulders_idx[:n]

        # 4) convert to times
        #    (safe guard: indices could be empty)
        if time_series.empty:
            peak_times = pd.Series(dtype=float)
            shoulder_times = pd.Series(dtype=float)
        else:
            # use iloc with bounds checks
            max_iloc = len(time_series) - 1
            peaks_idx = peaks_idx[peaks_idx <= max_iloc]
            shoulders_idx = shoulders_idx[shoulders_idx <= max_iloc]

            series = time_series.reset_index(drop=True)
            peak_times = series.iloc[peaks_idx]
            shoulder_times = series.iloc[shoulders_idx]

            if len(peak_times) != len(shoulder_times):
                n = min(len(peak_times), len(shoulder_times))
                peak_times = peak_times.iloc[:n]
                shoulder_times = shoulder_times.iloc[:n]

        results.append(
            (start_time, end_time, peaks_idx,
             peak_times, shoulders_idx, shoulder_times)
        )

    return results
