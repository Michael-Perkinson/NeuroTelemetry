import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from typing import Dict, Tuple


def analyse_photometry_peaks(
    photometry_data: pd.DataFrame,
    main_signal_col: str = "dFoF_465",
    prominence: float = 0.05,
    amp_thresh: float = 0.01,
    waveform_window: float = 5.0,   # seconds before/after peak for waveform averaging
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Detect peaks in photometry signal, extract per-peak metrics (no troughs).
    Returns:
        per_peak_df: PeakTime, Amplitude, PeakInterval, PeaksPerMinute, Width, PeakCount
        summary: dict of summary statistics
    """

    if "TimeSinceReference" not in photometry_data.columns:
        raise ValueError(
            "photometry_data must contain a 'TimeSinceReference' column.")
    if main_signal_col.strip() not in photometry_data.columns.str.strip():
        raise ValueError(
            f"photometry_data missing required column: {main_signal_col}")

    t = photometry_data["TimeSinceReference"].to_numpy()
    y = photometry_data[main_signal_col.strip()].to_numpy()

    # --- Peak detection ---
    peaks, p_props = find_peaks(y, prominence=prominence)

    if len(peaks) == 0:
        return pd.DataFrame(), {"n_peaks": 0}

    # Extract peak metrics
    peak_times = t[peaks]
    peak_vals = y[peaks]

    # Widths (FWHM from scipy)
    widths_res = peak_widths(y, peaks, rel_height=0.5)
    widths = widths_res[0] * np.median(np.diff(t))  # convert to seconds

    # Apply amplitude threshold filter
    keep_mask = peak_vals > amp_thresh
    peak_times = peak_times[keep_mask]
    amplitudes = peak_vals[keep_mask]
    widths = widths[keep_mask]

    # Peak-to-peak intervals (ISI, in seconds)
    isi = np.diff(peak_times)
    isi = np.append(isi, np.nan)  # last peak gets NaN

    # Peaks per minute (cumulative rate at each peak time)
    elapsed_minutes = (peak_times - peak_times[0]) / 60.0
    peaks_per_minute = np.arange(1, len(peak_times) + 1) / elapsed_minutes
    peaks_per_minute[0] = np.nan  # undefined for the very first peak

    # Cumulative peak count
    peak_count = np.arange(1, len(peak_times) + 1)

    # --- Build dataframe ---
    per_peak_df = pd.DataFrame({
        "PeakTime": peak_times,
        "Amplitude": amplitudes,
        "PeakInterval": isi,
        "PeaksPerMinute": peaks_per_minute,
        "Width": widths,
        "PeakCount": peak_count
    })

    # --- Summary stats ---
    summary = {
        "n_peaks": len(per_peak_df),
        "mean_amp": float(np.mean(amplitudes)) if len(amplitudes) else np.nan,
        "mean_width": float(np.mean(widths)) if len(widths) else np.nan,
        "mean_peak_interval": float(np.nanmean(isi)) if len(isi) else np.nan,
        "overall_peaks_per_min": float(len(peak_times) / ((peak_times[-1] - peak_times[0]) / 60.0)) if len(peak_times) > 1 else np.nan,
    }

    return per_peak_df, summary


def bin_peaks(
    per_peak_df: pd.DataFrame,
    bin_edges: np.ndarray,
    injection_sec: float
) -> pd.DataFrame:
    """
    Bin detected peaks into fixed time windows relative to injection time.
    """
    if per_peak_df.empty:
        return pd.DataFrame(columns=[
            "BinStart", "BinEnd",
            "PeakCount",
            "MeanAmp", "SEMAmp",
            "MeanWidth", "SEMWidth",
            "MeanISI", "SEMISI"
        ])

    # Convert absolute → relative time
    t_rel = per_peak_df["PeakTime"].to_numpy() - injection_sec
    bin_indices = np.digitize(t_rel, bin_edges) - 1

    rows = []
    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        sub = per_peak_df.loc[mask]

        def mean_sem(x):
            if len(x) == 0:
                return np.nan, np.nan
            arr = x.to_numpy(dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return np.nan, np.nan
            return np.mean(arr), np.std(arr, ddof=1) / np.sqrt(len(arr))

        mean_amp, sem_amp = mean_sem(
            sub["Amplitude"]) if "Amplitude" in sub else (np.nan, np.nan)
        mean_width, sem_width = mean_sem(
            sub["Width"]) if "Width" in sub else (np.nan, np.nan)

        # ISIs: drop last so it doesn’t bleed into next bin
        isi_in_bin = sub["PeakInterval"].iloc[:-1] if len(sub) > 1 else []
        mean_isi, sem_isi = mean_sem(isi_in_bin) if len(
            isi_in_bin) else (np.nan, np.nan)

        rows.append([
            bin_edges[i], bin_edges[i+1],
            len(sub),
            mean_amp, sem_amp,
            mean_width, sem_width,
            mean_isi, sem_isi
        ])

    return pd.DataFrame(rows, columns=[
        "BinStart", "BinEnd",
        "PeakCount",
        "MeanAmp", "SEMAmp",
        "MeanWidth", "SEMWidth",
        "MeanISI", "SEMISI"
    ])
