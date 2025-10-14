import math

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter


def calculate_dynamic_bins(
    array_length: int,
    b_ref: int = 600,
    t_ref: int = 1200,
    k: int = 5,
) -> int:
    """Dynamically calculate histogram bin count using logarithmic scaling."""
    bins = int(b_ref * math.log(1 + (array_length * k / t_ref)))
    return min(array_length, bins)


def butter_lowpass_filter(data, cutoff_hz, fs, order=3):
    """Apply a low-pass Butterworth filter to 1D data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)  # type: ignore
    return filtfilt(b, a, data)


def compute_first_derivative(signal, fs):
    """Compute first derivative via Savitzky–Golay."""
    return savgol_filter(
        signal,
        window_length=35,
        polyorder=2,
        deriv=1,
        delta=1 / fs,
        mode="interp",
    )


def get_time_bounds(pressure_data: pd.DataFrame) -> tuple[float, float]:
    """Returns the min and max of TimeSinceReference from the pressure data."""
    min_time = pressure_data["TimeSinceReference"].min()
    max_time = pressure_data["TimeSinceReference"].max()
    return min_time, max_time


def compute_time_window(photo_min_time, photo_max_time, injection_sec, pre_min, post_min):
    start_time = injection_sec - pre_min * 60
    end_time = injection_sec + post_min * 60
    return max(photo_min_time, start_time), min(photo_max_time, end_time)

def get_nearest_points(
    target_times: list[float], df: pd.DataFrame, time_col: str, value_col: str
) -> tuple[list[float], list[float]]:
    """Find nearest (time, value) pairs for targets using searchsorted."""
    available_times = df[time_col].values
    values = df[value_col].values

    target_times = np.asarray(target_times)

    # Find indices where each target would be inserted to maintain order
    insert_indices = np.searchsorted(available_times, target_times)

    # Clip to valid range
    insert_indices = np.clip(insert_indices, 1, len(available_times) - 1)

    # Compare distances to neighbors to choose closer index
    left_indices = insert_indices - 1
    right_indices = insert_indices

    left_dists = np.abs(available_times[left_indices] - target_times)
    right_dists = np.abs(available_times[right_indices] - target_times)

    best_indices = np.where(left_dists <= right_dists, left_indices, right_indices)

    matched_times = available_times[best_indices]
    matched_values = values[best_indices]

    return matched_times.tolist(), matched_values.tolist()
