from scipy.signal import detrend
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_segment_csv(csv_path: Path) -> pd.DataFrame:
    """Load a single exported window CSV."""
    return pd.read_csv(csv_path)


def butter_lowpass_filter(data, cutoff_hz=10, fs=500, order=3):
    """Apply a low-pass Butterworth filter to suppress high-frequency noise."""
    nyq = fs / 2
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def apply_combined_filters(raw_pressure: np.ndarray) -> dict:
    """Apply detrend, then combinations of filters for comparison."""
    filtered_versions = {}

    detrended = detrend(raw_pressure, type='linear')
    filtered_versions['Detrended Raw'] = detrended

    butter = butter_lowpass_filter(detrended)
    filtered_versions['Butter-only'] = butter

    filtered_versions['Butter + SG (35,2)'] = savgol_filter(butter, 35, 2)
    filtered_versions['Butter + SG (75,2)'] = savgol_filter(butter, 75, 2)
    filtered_versions['SG-only (35,2)'] = savgol_filter(detrended, 35, 2)
    filtered_versions['SG-only (75,2)'] = savgol_filter(detrended, 75, 2)

    return filtered_versions



def find_negative_peaks(
    time, trace, time_range=(2290, 2310), prominence=1.5, distance_samples=100
):
    """
    Find indices of negative peaks in a time window with filtering.
    - prominence: how deep the peak must be relative to surrounding data
    - distance_samples: minimum number of samples between peaks (e.g. 100 = 200 ms at 500 Hz)
    """
    mask = (time >= time_range[0]) & (time <= time_range[1])
    time_window = time[mask]
    trace_window = trace[mask]

    # Find peaks in inverted trace (because they're negative)
    peaks, _ = find_peaks(-trace_window,
                          prominence=prominence, distance=distance_samples)

    return time_window.values[peaks], trace_window[peaks]



def compute_avg_amplitudes_and_peaks(time, traces: dict, time_range=(2290, 2310)):
    """Compute average amplitude and return peaks for marking."""
    results = {}
    peaks_info = {}

    for label, trace in traces.items():
        peak_times, peak_vals = find_negative_peaks(time, trace, time_range)
        if len(peak_vals) > 0:
            avg_amp = np.mean(peak_vals)
            results[label] = avg_amp
            peaks_info[label] = (peak_times, peak_vals)
        else:
            results[label] = None
            peaks_info[label] = ([], [])

    return results, peaks_info


def plot_comparisons(time, traces, peaks_info, time_range=(2290, 2310)):
    """Plot filtered traces and mark detected peaks, zoomed into time range."""
    fig, axs = plt.subplots(len(traces), 1, figsize=(12, 12), sharex=True)

    for i, (label, trace) in enumerate(traces.items()):
        axs[i].plot(time, trace, label=label, color='steelblue')
        peak_times, peak_vals = peaks_info[label]
        axs[i].scatter(peak_times, peak_vals, color='red', label='Peaks', s=10)
        axs[i].set_title(label)
        axs[i].set_ylim(-15, 2)
        axs[i].legend()

    axs[-1].set_xlabel("Time (s)")
    axs[0].set_xlim(time_range)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = Path("pressure_segment_exports/pressure_segment_window1.csv")
    df = load_segment_csv(csv_path)

    time = df['TimeSinceReference']
    raw_pressure = df['Pressure'].values

    filtered_versions = apply_combined_filters(raw_pressure)

    # Find peaks and compute average amplitudes before plotting
    avg_amplitudes, peaks_info = compute_avg_amplitudes_and_peaks(
        time, filtered_versions, time_range=(2290, 2310))

    # Print results
    print("\nAverage amplitudes of negative peaks (2290–2310 s):")
    for label, amp in avg_amplitudes.items():
        if amp is not None:
            print(f"{label}: {amp:.3f}")
        else:
            print(f"{label}: No peaks found")

    # Plot and mark peaks
    plot_comparisons(time, filtered_versions,
                     peaks_info, time_range=(2290, 2310))
