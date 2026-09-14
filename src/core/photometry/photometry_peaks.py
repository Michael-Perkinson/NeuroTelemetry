import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import find_peaks, peak_widths


def analyse_photometry_peaks(
    photometry_data: pd.DataFrame,
    main_signal_col: str = "dFoF_465",
    prominence: float = 0.05,
    amp_thresh: float = 0.01,
) -> tuple[pd.DataFrame, list[float], dict[str, float | int]]:
    """
    Detect peaks in photometry signal, extract per-peak metrics (no troughs).
    Returns the per-peak metrics, peak timestamps, and summary statistics.
    """
    if "TimeSinceReference" not in photometry_data.columns:
        raise ValueError("photometry_data must contain 'TimeSinceReference'")
    normalized_columns = photometry_data.columns.astype(str).str.strip()
    matching_columns = photometry_data.columns[
        normalized_columns == main_signal_col.strip()
    ]
    if matching_columns.empty:
        raise ValueError(f"Missing required column: {main_signal_col}")
    resolved_signal_col = matching_columns[0]

    t = photometry_data["TimeSinceReference"].to_numpy()
    y = photometry_data[resolved_signal_col].to_numpy()

    peaks, _ = find_peaks(y, prominence=prominence)
    if len(peaks) == 0:
        return pd.DataFrame(), [], {"n_peaks": 0}

    peak_times = t[peaks]
    peak_vals = y[peaks]

    widths_res = peak_widths(y, peaks, rel_height=0.5)
    widths = widths_res[0] * np.median(np.diff(t))

    keep_mask = peak_vals > amp_thresh
    peak_times = peak_times[keep_mask]
    amplitudes = peak_vals[keep_mask]
    widths = widths[keep_mask]

    isi = np.diff(peak_times)
    isi = np.append(isi, np.nan)

    peak_count = np.arange(1, len(peak_times) + 1)

    per_peak_df = pd.DataFrame(
        {
            "PeakTime": peak_times,
            "Amplitude": amplitudes,
            "PeakInterval": isi,
            "PeakCount": peak_count,
            "Width": widths,
        }
    )

    window_minutes = (
        photometry_data["TimeSinceReference"].max()
        - photometry_data["TimeSinceReference"].min()
    ) / 60.0
    overall_peaks_per_min = (
        len(peak_times) / window_minutes if window_minutes > 0 else np.nan
    )

    summary = {
        "n_peaks": len(per_peak_df),
        "mean_amp": float(np.mean(amplitudes)) if len(amplitudes) else np.nan,
        "mean_width": float(np.mean(widths)) if len(widths) else np.nan,
        "mean_peak_interval": float(np.nanmean(isi)) if len(isi) else np.nan,
        "overall_peaks_per_min": overall_peaks_per_min,
    }

    peak_times = per_peak_df["PeakTime"].tolist() if not per_peak_df.empty else []

    return per_peak_df, peak_times, summary


def bin_peaks(
    per_peak_df: pd.DataFrame, bin_edges: np.ndarray, injection_sec: float
) -> pd.DataFrame:
    """
    Bin detected peaks into fixed time windows relative to injection time.
    Reports count, mean amp/width/ISI, and peaks per minute.
    """
    if per_peak_df.empty:
        return pd.DataFrame(
            columns=[
                "BinStart",
                "BinEnd",
                "PeakCount",
                "PeaksPerMinute",
                "MeanAmp",
                "SEMAmp",
                "MeanWidth",
                "SEMWidth",
                "MeanISI",
                "SEMISI",
            ]
        )

    # Convert absolute → relative
    t_rel = per_peak_df["PeakTime"].to_numpy() - injection_sec
    bin_indices: npt.NDArray[np.intp] = np.asarray(
        np.digitize(t_rel, bin_edges) - 1, dtype=np.intp
    )

    n_bins = len(bin_edges) - 1
    in_range = (bin_indices >= 0) & (bin_indices < n_bins)
    peaks = per_peak_df.loc[in_range].copy()
    peaks["Bin"] = bin_indices[in_range]
    bin_index = pd.RangeIndex(n_bins, name="Bin")

    def grouped_stats(column: str, data: pd.DataFrame = peaks) -> pd.DataFrame:
        if column not in data:
            return pd.DataFrame(index=bin_index, columns=["mean", "sem"], dtype=float)
        grouped = data.groupby("Bin")[column].agg(["count", "mean", "std"])
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
        return grouped[["mean", "sem"]].reindex(bin_index)

    amplitude = grouped_stats("Amplitude")
    width = grouped_stats("Width")
    isi_source = peaks.loc[peaks["Bin"].duplicated(keep="last")]
    isi = grouped_stats("PeakInterval", isi_source)
    peak_count = peaks.groupby("Bin").size().reindex(bin_index, fill_value=0)
    bin_minutes = np.diff(bin_edges) / 60.0
    peaks_per_min = np.divide(
        peak_count.to_numpy(dtype=float),
        bin_minutes,
        out=np.full(n_bins, np.nan),
        where=bin_minutes > 0,
    )

    return pd.DataFrame(
        {
            "BinStart": bin_edges[:-1],
            "BinEnd": bin_edges[1:],
            "PeakCount": peak_count.to_numpy(),
            "PeaksPerMinute": peaks_per_min,
            "MeanAmp": amplitude["mean"].to_numpy(),
            "SEMAmp": amplitude["sem"].to_numpy(),
            "MeanWidth": width["mean"].to_numpy(),
            "SEMWidth": width["sem"].to_numpy(),
            "MeanISI": isi["mean"].to_numpy(),
            "SEMISI": isi["sem"].to_numpy(),
        }
    )
