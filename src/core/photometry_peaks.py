import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from typing import Dict, Tuple


def analyse_photometry_peaks(
    photometry_data: pd.DataFrame,
    main_signal_col: str = "dFoF_465",
    prominence: float = 0.08,
    amp_thresh: float = 0.01,
    waveform_window: float = 5.0,   # seconds before/after peak for waveform averaging
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Detect peaks in photometry signal, extract per-peak metrics (no troughs).
    Returns:
        per_peak_df: PeakTime, Amplitude, Width, PeakInterval
        summary: dict of summary statistics
    """

    if "TimeSinceReference" not in photometry_data.columns:
        raise ValueError(
            "photometry_data must contain a 'TimeSinceReference' column."
        )
    if main_signal_col.strip() not in photometry_data.columns.str.strip():
        raise ValueError(
            f"photometry_data missing required column: {main_signal_col}"
        )

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

    # Peak-to-peak intervals (ISI)
    isi = np.diff(peak_times)
    isi = np.append(isi, np.nan)  # last peak gets NaN

    # --- Build dataframe ---
    per_peak_df = pd.DataFrame({
        "PeakTime": peak_times,
        "Amplitude": amplitudes,
        "Width": widths,
        "PeakInterval": isi
    })

    # --- Summary stats ---
    summary = {
        "n_peaks": len(per_peak_df),
        "mean_amp": float(np.mean(amplitudes)) if len(amplitudes) else np.nan,
        "mean_width": float(np.mean(widths)) if len(widths) else np.nan,
        "mean_peak_interval": float(np.nanmean(isi)) if len(isi) else np.nan,
    }

    return per_peak_df, summary


def bin_peaks(per_peak_df: pd.DataFrame, bin_size_sec: float) -> pd.DataFrame:
    if per_peak_df.empty:
        return pd.DataFrame(columns=[
            "BinStart", "BinEnd",
            "MeanAmp", "SEMAmp",
            "MeanWidth", "SEMWidth",
            "MeanISI", "SEMISI"
        ])

    t_abs = per_peak_df["PeakTime"].to_numpy()
    t0 = t_abs.min()
    bins_abs = np.arange(t0, t_abs.max() + bin_size_sec, bin_size_sec)
    bin_indices = np.digitize(t_abs, bins_abs) - 1

    rows = []
    for i in range(len(bins_abs)-1):
        mask = bin_indices == i
        sub = per_peak_df.loc[mask]

        def mean_sem(x):
            if len(x) == 0:
                return np.nan, np.nan
            arr = x.to_numpy(dtype=float)
            if np.all(np.isnan(arr)):
                return np.nan, np.nan
            return np.nanmean(arr), np.nanstd(arr, ddof=1) / np.sqrt(np.sum(~np.isnan(arr)))

        mean_amp, sem_amp = mean_sem(
            sub["Amplitude"]) if "Amplitude" in sub else (np.nan, np.nan)
        mean_width, sem_width = mean_sem(
            sub["Width"]) if "Width" in sub else (np.nan, np.nan)

        # ISIs: drop the last in each bin so it doesn’t bleed to next bin
        isi_in_bin = sub["PeakInterval"].iloc[:-1] if len(sub) > 1 else []
        mean_isi, sem_isi = mean_sem(isi_in_bin) if len(
            isi_in_bin) else (np.nan, np.nan)

        rows.append([
            bins_abs[i] - t0, bins_abs[i+1] - t0,
            mean_amp, sem_amp,
            mean_width, sem_width,
            mean_isi, sem_isi
        ])

    return pd.DataFrame(rows, columns=[
        "BinStart", "BinEnd",
        "MeanAmp", "SEMAmp",
        "MeanWidth", "SEMWidth",
        "MeanISI", "SEMISI"
    ])
