import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks, detrend

# Parameters
CUTOFF_HZ = 10
SG_WINDOW = 35
SG_POLY = 2
SAMPLING_RATE = 500
TIME_RANGE = (2200, 2600)
PROMINENCE = 3
MIN_PEAK_DISTANCE_SAMPLES = 20
THRESHOLD_FACTOR = 0.1  # Adjust this to tune shoulder detection sensitivity


def butter_lowpass_filter(data, cutoff_hz, fs, order=3):
    nyq = fs / 2
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def preprocess(raw_pressure):
    detrended = detrend(raw_pressure, type='linear')
    buttered = butter_lowpass_filter(
        detrended, cutoff_hz=CUTOFF_HZ, fs=SAMPLING_RATE)
    smoothed = savgol_filter(
        buttered, window_length=SG_WINDOW, polyorder=SG_POLY)
    return smoothed, detrended


def compute_first_derivative(signal, fs):
    """Compute first derivative via Savitzky–Golay."""
    return savgol_filter(
        signal, window_length=SG_WINDOW, polyorder=SG_POLY,
        deriv=1, delta=1/fs, mode='interp'
    )


def find_peak_shoulders(dvdt_seg, start_idx, peak_idx, threshold_factor=THRESHOLD_FACTOR):
    """First derivative threshold approach with zero-crossing constraint"""
    if len(dvdt_seg) == 0:
        return start_idx

    # Find the last zero crossing (or closest to zero) before the peak
    # This represents the end of the flat/baseline phase
    zero_crossings = []
    for i in range(len(dvdt_seg) - 1):
        if dvdt_seg[i] >= 0 and dvdt_seg[i + 1] < 0:  # Positive to negative crossing
            zero_crossings.append(i + 1)  # Take the first negative point

    # If no zero crossing found, find the point closest to zero
    if not zero_crossings:
        print(f'No zero crossing for peak at {peak_idx}')
        closest_to_zero_idx = np.argmin(np.abs(dvdt_seg))
        # Only use if it's reasonably close to zero (not deep negative)
        if dvdt_seg[closest_to_zero_idx] > -0.5:  # Adjust threshold as needed
            zero_crossings = [closest_to_zero_idx]

    # Determine search start point
    if zero_crossings:
        search_start = zero_crossings[-1]  # Use the last zero crossing
        dvdt_search = dvdt_seg[search_start:]
        search_offset = search_start
    else:
        # Fallback: search from beginning but with warning
        print(
            f"Warning: No zero crossing found for peak at {peak_idx}, using full segment")
        dvdt_search = dvdt_seg
        search_offset = 0

    if len(dvdt_search) == 0:
        return start_idx + search_offset

    # Find the steepest negative slope in the search region
    min_slope = np.min(dvdt_search)
    min_slope_idx = np.argmin(dvdt_search)

    # Set threshold as fraction of steepest slope
    threshold = min_slope * threshold_factor

    # Find first point where slope exceeds threshold (becomes steep enough)
    steep_points = np.where(dvdt_search <= threshold)[0]

    if len(steep_points) > 0:
        shoulder_rel = steep_points[0]
    else:
        print("Used fallback to steepest point")
        shoulder_rel = min_slope_idx

    return start_idx + search_offset + shoulder_rel


def find_peaks_and_shoulders(time, pressure, dvdt):
    # mask = (time >= TIME_RANGE[0]) & (time <= TIME_RANGE[1])
    # time_win = time[mask].values
    time_win = time.values
    pressure_win = pressure
    dvdt_win = dvdt

    # Find peaks (breathing in = downward deflections = negative peaks)
    peaks, _ = find_peaks(
        -pressure_win,
        prominence=PROMINENCE,
        distance=MIN_PEAK_DISTANCE_SAMPLES
    )

    shoulders = []

    for i, pk in enumerate(peaks):
        # Define search window (from previous peak or start to current peak)
        start = peaks[i-1] if i > 0 else 0

        # Get segment for analysis
        dvdt_seg = dvdt_win[start:pk]

        # Find shoulder using first derivative threshold method
        shoulder = find_peak_shoulders(dvdt_seg, start, pk)
        shoulders.append(shoulder)

    return time_win, pressure_win, dvdt_win, peaks, shoulders


def plot_results(time_win, pressure_win, dvdt_win, peaks, shoulders):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot 1: Pressure signal
    ax1.plot(time_win, pressure_win, color='steelblue',
             label='Pressure', linewidth=1.5)
    ax1.scatter(time_win[peaks], pressure_win[peaks],
                color='red', label='Peaks', zorder=3, s=50)
    ax1.scatter(time_win[shoulders], pressure_win[shoulders],
                color='orange', label='Shoulders', zorder=3, s=40)
    ax1.set_ylabel("Pressure")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: First derivative
    ax2.plot(time_win, dvdt_win, color='darkorange',
             label='dv/dt', linewidth=1.5)
    ax2.scatter(time_win[shoulders], dvdt_win[shoulders],
                color='orange', s=40, zorder=3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel("dv/dt")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Breathing Signal: First Derivative Threshold Shoulder Detection", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(
        Path("pressure_segment_exports/pressure_segment_window1.csv"))
    time = df['TimeSinceReference']
    raw_pressure = df['Pressure'].values

    # Preprocess
    smoothed, _ = preprocess(raw_pressure)

    # Compute first derivative
    dvdt = compute_first_derivative(smoothed, SAMPLING_RATE)

    # Find peaks and shoulders
    time_win, pressure_win, dvdt_win, peaks, shoulders = find_peaks_and_shoulders(
        time, smoothed, dvdt
    )

    # Plot results
    plot_results(time_win, pressure_win, dvdt_win, peaks, shoulders)
