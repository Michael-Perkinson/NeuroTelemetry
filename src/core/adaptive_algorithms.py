from typing import Tuple
import pandas as pd
import math
from scipy.signal import butter, filtfilt, savgol_filter


def calculate_dynamic_bins(array_length: int, b_ref: int = 600, t_ref: int = 1200, k: int = 5) -> int:
    """Dynamically calculate the number of bins for the histogram using logarithmic scaling."""

    bins = int(b_ref * math.log(1 + (array_length * k / t_ref)))
    return min(array_length, bins)


def butter_lowpass_filter(data, cutoff_hz, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
    return filtfilt(b, a, data)


def compute_first_derivative(signal, fs):
    """Compute first derivative via Savitzky–Golay."""
    return savgol_filter(
        signal, window_length=75, polyorder=2,
        deriv=1, delta=1/fs, mode='interp'
    )


def get_time_bounds(pressure_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns the min and max of TimeSinceReference from the pressure data.
    """
    min_time = pressure_data['TimeSinceReference'].min()
    max_time = pressure_data['TimeSinceReference'].max()
    return min_time, max_time
