import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from typing import Dict, Tuple


def analyse_photometry_peaks(
    photometry_data: pd.DataFrame,
    main_signal_col: str = "dFoF_465",
    prominence: float = 0.05,
    amp_thresh: float = 0.01,
) -> Tuple[pd.DataFrame, list, Dict[str, float]]:
    """
    Detect peaks in photometry signal, extract per-peak metrics (no troughs).
    Returns:
        per_peak_df: PeakTime, Amplitude, PeakInterval (ISI), PeakCount, Width
        summary: dict of summary statistics
    """
    if "TimeSinceReference" not in photometry_data.columns:
        raise ValueError("photometry_data must contain 'TimeSinceReference'")
    if main_signal_col.strip() not in photometry_data.columns.str.strip():
        raise ValueError(f"Missing required column: {main_signal_col}")

    t = photometry_data["TimeSinceReference"].to_numpy()
    y = photometry_data[main_signal_col.strip()].to_numpy()

    # --- Peak detection ---
    peaks, _ = find_peaks(y, prominence=prominence)
    if len(peaks) == 0:
        return pd.DataFrame(), [], {"n_peaks": 0}

    peak_times = t[peaks]
    peak_vals = y[peaks]

    # Widths
    widths_res = peak_widths(y, peaks, rel_height=0.5)
    widths = widths_res[0] * np.median(np.diff(t))

    # Amplitude threshold
    keep_mask = peak_vals > amp_thresh
    peak_times = peak_times[keep_mask]
    amplitudes = peak_vals[keep_mask]
    widths = widths[keep_mask]

    # ISI (sec between peaks)
    isi = np.diff(peak_times)
    isi = np.append(isi, np.nan)

    # Cumulative peak count
    peak_count = np.arange(1, len(peak_times) + 1)

    # --- Build dataframe ---
    per_peak_df = pd.DataFrame({
        "PeakTime": peak_times,
        "Amplitude": amplitudes,
        "PeakInterval": isi,
        "PeakCount": peak_count,
        "Width": widths,
    })

    # --- Summary (includes global peaks per min) ---
    window_minutes = (photometry_data["TimeSinceReference"].max() -
                      photometry_data["TimeSinceReference"].min()) / 60.0
    overall_peaks_per_min = len(
        peak_times) / window_minutes if window_minutes > 0 else np.nan

    summary = {
        "n_peaks": len(per_peak_df),
        "mean_amp": float(np.mean(amplitudes)) if len(amplitudes) else np.nan,
        "mean_width": float(np.mean(widths)) if len(widths) else np.nan,
        "mean_peak_interval": float(np.nanmean(isi)) if len(isi) else np.nan,
        "overall_peaks_per_min": overall_peaks_per_min,
    }
    
    peak_times = per_peak_df["PeakTime"].tolist(
        ) if not per_peak_df.empty else []

    return per_peak_df, peak_times, summary


def bin_peaks(
    per_peak_df: pd.DataFrame,
    bin_edges: np.ndarray,
    injection_sec: float
) -> pd.DataFrame:
    """
    Bin detected peaks into fixed time windows relative to injection time.
    Reports count, mean amp/width/ISI, and peaks per minute.
    """
    if per_peak_df.empty:
        return pd.DataFrame(columns=[
            "BinStart", "BinEnd",
            "PeakCount", "PeaksPerMinute",
            "MeanAmp", "SEMAmp",
            "MeanWidth", "SEMWidth",
            "MeanISI", "SEMISI"
        ])

    # Convert absolute → relative
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

        # ISIs within this bin
        isi_in_bin = sub["PeakInterval"].iloc[:-1] if len(sub) > 1 else []
        mean_isi, sem_isi = mean_sem(isi_in_bin) if len(
            isi_in_bin) else (np.nan, np.nan)

        # Peak count + normalized frequency
        peak_count = len(sub)
        bin_minutes = (bin_edges[i+1] - bin_edges[i]) / 60.0
        peaks_per_min = peak_count / bin_minutes if bin_minutes > 0 else np.nan

        rows.append([
            bin_edges[i], bin_edges[i+1],
            peak_count, peaks_per_min,
            mean_amp, sem_amp,
            mean_width, sem_width,
            mean_isi, sem_isi
        ])

    return pd.DataFrame(rows, columns=[
        "BinStart", "BinEnd",
        "PeakCount", "PeaksPerMinute",
        "MeanAmp", "SEMAmp",
        "MeanWidth", "SEMWidth",
        "MeanISI", "SEMISI"
    ])
