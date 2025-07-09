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
TIME_RANGE = (2290, 2310)
PROMINENCE = 1.5
MIN_PEAK_DISTANCE_SAMPLES = 100


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


def compute_dvdt(signal, fs):
    dt = 1 / fs
    return np.gradient(signal, dt)


def find_peaks_and_prepeaks(time, pressure, dvdt):
    """Find peaks and pre-peaks using dv/dt logic.

    Pre-peak is defined as the last point where dv/dt was rising
    before it started dropping — i.e., local maximum before dv/dt trough.
    """
    mask = (time >= TIME_RANGE[0]) & (time <= TIME_RANGE[1])
    time_win = time[mask].values
    pressure_win = pressure[mask]
    dvdt_win = dvdt[mask]

    # Find negative peaks in pressure
    peaks, _ = find_peaks(-pressure_win, prominence=PROMINENCE,
                          distance=MIN_PEAK_DISTANCE_SAMPLES)
    pre_peaks = []

    for i, peak_idx in enumerate(peaks):
        # Define start for search:
        if i == 0:
            search_start = 0
        else:
            search_start = peaks[i - 1]

        dvdt_segment = dvdt_win[search_start:peak_idx]
        if len(dvdt_segment) < 3:
            pre_peaks.append(None)
            continue

        # Find the steepest point (trough in dv/dt)
        min_dvdt_idx = np.argmin(dvdt_segment)

        # Walk backward from that trough to find last rising slope (local max in dv/dt)
        for j in range(min_dvdt_idx, 0, -1):
            if dvdt_segment[j] > dvdt_segment[j - 1]:
                pre_peaks.append(search_start + j)
                break
        else:
            pre_peaks.append(search_start)  # fallback if nothing found

    return time_win, pressure_win, peaks, pre_peaks


def plot_pressure_and_dvdt(time_win, pressure_win, dvdt_win, peaks, pre_peaks):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot pressure trace
    ax1.plot(time_win, pressure_win, color='steelblue',
             label='Butter + SG (35,2)')
    ax1.scatter(time_win[peaks], pressure_win[peaks],
                color='red', label='Peaks', zorder=3)
    valid_pre = [p for p in pre_peaks if p is not None]
    ax1.scatter(time_win[valid_pre], pressure_win[valid_pre],
                color='orange', label='Pre-peaks', zorder=3)
    ax1.set_ylabel("Pressure (detrended)", color='steelblue')
    ax1.set_ylim(-15, 0)
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Twin y-axis for dv/dt
    ax2 = ax1.twinx()
    ax2.plot(time_win, dvdt_win, color='darkgreen', label='dv/dt')
    ax2.scatter(time_win[peaks], dvdt_win[peaks], color='red', s=10)
    ax2.scatter(time_win[valid_pre], dvdt_win[valid_pre], color='orange', s=10)
    ax2.set_ylabel("dv/dt", color='darkgreen')
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    fig.suptitle(
        "Butter + SG (35,2): Pressure and dv/dt with Peaks & Pre-peaks")
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(
        1, 1), bbox_transform=ax1.transAxes)
    plt.show()


if __name__ == "__main__":
    csv_path = Path("pressure_segment_exports/pressure_segment_window1.csv")
    df = pd.read_csv(csv_path)

    time = df['TimeSinceReference']
    raw_pressure = df['Pressure'].values

    smoothed, detrended = preprocess(raw_pressure)
    dvdt = compute_dvdt(smoothed, SAMPLING_RATE)

    time_win, pressure_win, peaks, pre_peaks = find_peaks_and_prepeaks(
        time, smoothed, dvdt)

    plot_pressure_and_dvdt(time_win, pressure_win, dvdt[(
        time >= TIME_RANGE[0]) & (time <= TIME_RANGE[1])], peaks, pre_peaks)
