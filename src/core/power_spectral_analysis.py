from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import welch

DEFAULT_PSD_RESAMPLE_HZ = 10.0
DEFAULT_PSD_SEGMENT_SECONDS = 60.0
DEFAULT_PSD_WELCH_WINDOWS = 3
DEFAULT_PSD_OVERLAP_FRACTION = 0.5
DEFAULT_PSD_NFFT = 2048


def _sanitize_peak_times(peak_times: pd.Series) -> np.ndarray:
    """Return finite, strictly increasing peak timestamps."""
    peaks = np.asarray(peak_times, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    if peaks.size < 2:
        return np.array([], dtype=float)

    keep = np.concatenate(([True], np.diff(peaks) > 0))
    return peaks[keep]


def _build_ttot_trace(period: dict[str, Any]) -> pd.DataFrame:
    """Create a clean breath-by-breath Ttot trace from period peak times."""
    peak_times = _sanitize_peak_times(period["peak_times"])
    if peak_times.size < 2:
        return pd.DataFrame(columns=["TimeOfBreath_s", "Ttot_s"])

    t_breath = peak_times[:-1]
    ttot = np.diff(peak_times)

    trace = pd.DataFrame(
        {
            "TimeOfBreath_s": t_breath,
            "Ttot_s": ttot,
        }
    )
    trace = trace.replace([np.inf, -np.inf], np.nan).dropna()
    trace = trace[trace["Ttot_s"] > 0].copy()
    trace = trace.drop_duplicates(subset="TimeOfBreath_s", keep="first")
    trace = trace.sort_values("TimeOfBreath_s").reset_index(drop=True)

    if trace.empty:
        return trace

    monotonic_mask = np.concatenate(
        ([True], np.diff(trace["TimeOfBreath_s"].to_numpy(dtype=float)) > 0)
    )
    return trace.loc[monotonic_mask].reset_index(drop=True)


def _iter_segment_bounds(
    start_time: float,
    end_time: float,
    segment_seconds: float,
) -> list[tuple[float, float]]:
    """Split a usable trace span into successive contiguous fixed-length segments."""
    total_duration = end_time - start_time
    if total_duration < segment_seconds:
        return []

    n_segments = int(total_duration // segment_seconds)
    return [
        (
            start_time + i * segment_seconds,
            start_time + (i + 1) * segment_seconds,
        )
        for i in range(n_segments)
    ]


def _resample_segment(
    trace_df: pd.DataFrame,
    segment_start: float,
    segment_seconds: float,
    resample_hz: float,
) -> pd.DataFrame | None:
    """
    Resample a Ttot trace segment with cubic spline interpolation.

    Includes one sample of padding on either side of the segment boundaries so the
    interpolation can evaluate the fixed grid without extrapolation.
    """
    times = trace_df["TimeOfBreath_s"].to_numpy(dtype=float)
    values = trace_df["Ttot_s"].to_numpy(dtype=float)
    if times.size < 4:
        return None

    segment_end = segment_start + segment_seconds
    sample_spacing = 1.0 / resample_hz
    resampled_time = segment_start + np.arange(
        0.0, segment_seconds, sample_spacing, dtype=float
    )

    idx_start = int(np.searchsorted(times, segment_start, side="left"))
    idx_end = int(np.searchsorted(times, segment_end, side="right"))

    lo = max(0, idx_start - 1)
    hi = min(times.size, idx_end + 1)

    segment_times = times[lo:hi]
    segment_values = values[lo:hi]

    if segment_times.size < 4:
        return None
    if segment_times[0] > resampled_time[0] or segment_times[-1] < resampled_time[-1]:
        return None

    spline = CubicSpline(segment_times, segment_values, extrapolate=False)
    resampled_ttot = spline(resampled_time)
    if np.isnan(resampled_ttot).any():
        return None

    centered_ttot = resampled_ttot - float(np.mean(resampled_ttot))
    return pd.DataFrame(
        {
            "SegmentTime_s": resampled_time - segment_start,
            "AbsoluteTime_s": resampled_time,
            "Ttot_s": resampled_ttot,
            "TtotCentered_s": centered_ttot,
        }
    )


def _compute_welch_psd(
    centered_signal: np.ndarray,
    resample_hz: float,
    nfft: int,
    welch_windows: int,
    overlap_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD using the collaborator-specified windowing."""
    if centered_signal.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    nperseg = centered_signal.size // welch_windows
    if nperseg < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    noverlap = int(round(nperseg * overlap_fraction))
    freq_hz, psd = welch(
        centered_signal,
        fs=resample_hz,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
    )
    return freq_hz, psd


def _mean_psd_table(
    key_name: str,
    key_value: str,
    freq_hz: np.ndarray,
    psd_arrays: list[np.ndarray],
) -> pd.DataFrame:
    """Build a long-format mean PSD dataframe for one summary group."""
    if not psd_arrays:
        columns = [key_name] if key_name else []
        return pd.DataFrame(columns=columns + ["Frequency_Hz", "PSD"])

    mean_psd = np.mean(np.vstack(psd_arrays), axis=0)
    data: dict[str, Any] = {
        "Frequency_Hz": freq_hz,
        "PSD": mean_psd,
    }
    if key_name:
        data = {key_name: [key_value] * len(freq_hz), **data}
    return pd.DataFrame(data)


def export_ttot_traces(
    window_periods: dict[str, list[dict]],
    output_folder: Path,
    log=print,
) -> None:
    """
    Export Ttot breath-by-breath traces for each behavioural window.

    Creates one CSV per window:
        TimeOfBreath_s, Ttot_s, PeriodIndex
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for window_key, periods in window_periods.items():
        rows = []

        for period_idx, period in enumerate(periods, start=1):
            trace_df = _build_ttot_trace(period)
            if trace_df.empty:
                continue

            for row in trace_df.itertuples(index=False):
                rows.append(
                    {
                        "TimeOfBreath_s": float(row.TimeOfBreath_s),
                        "Ttot_s": float(row.Ttot_s),
                        "PeriodIndex": period_idx,
                    }
                )

        if not rows:
            log(f"No valid breath data in window {window_key}, skipping CSV.")
            continue

        df = pd.DataFrame(rows)
        safe_key = window_key.replace(":", "_").replace("-", "_").replace(" ", "_")
        out_path = output_folder / f"Ttot_{safe_key}.csv"
        df.to_csv(out_path, index=False)
        log(f"Exported Ttot CSV for window {window_key} -> {out_path}")


def analyze_ttot_psd(
    window_periods: dict[str, list[dict[str, Any]]],
    output_folder: Path,
    log=print,
    resample_hz: float = DEFAULT_PSD_RESAMPLE_HZ,
    segment_seconds: float = DEFAULT_PSD_SEGMENT_SECONDS,
    welch_windows: int = DEFAULT_PSD_WELCH_WINDOWS,
    overlap_fraction: float = DEFAULT_PSD_OVERLAP_FRACTION,
    nfft: int = DEFAULT_PSD_NFFT,
) -> dict[str, Any]:
    """
    Compute collaborator-specified Ttot variability PSDs from valid periods.

    Returns dataframes for workbook export and writes detailed CSV outputs into the
    PSD subfolder.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, Any]] = []
    segment_psd_frames: list[pd.DataFrame] = []
    resampled_trace_frames: list[pd.DataFrame] = []
    per_window_psd_frames: list[pd.DataFrame] = []

    pooled_psd_arrays: list[np.ndarray] = []
    pooled_freq_hz: np.ndarray | None = None
    window_psd_lookup: dict[str, list[np.ndarray]] = {}

    total_candidate_segments = 0

    for window_key, periods in window_periods.items():
        window_psd_lookup.setdefault(window_key, [])

        for period_idx, period in enumerate(periods, start=1):
            period_name = period.get("name", f"Period_{period_idx}")
            trace_df = _build_ttot_trace(period)
            if trace_df.empty:
                continue

            trace_start = float(trace_df["TimeOfBreath_s"].iloc[0])
            trace_end = float(trace_df["TimeOfBreath_s"].iloc[-1])
            segment_bounds = _iter_segment_bounds(
                trace_start,
                trace_end,
                segment_seconds,
            )
            total_candidate_segments += len(segment_bounds)

            for segment_idx, (segment_start, segment_end) in enumerate(
                segment_bounds, start=1
            ):
                resampled_df = _resample_segment(
                    trace_df,
                    segment_start=segment_start,
                    segment_seconds=segment_seconds,
                    resample_hz=resample_hz,
                )
                if resampled_df is None:
                    continue

                freq_hz, psd = _compute_welch_psd(
                    centered_signal=resampled_df["TtotCentered_s"].to_numpy(
                        dtype=float
                    ),
                    resample_hz=resample_hz,
                    nfft=nfft,
                    welch_windows=welch_windows,
                    overlap_fraction=overlap_fraction,
                )
                if freq_hz.size == 0 or psd.size == 0:
                    continue

                if pooled_freq_hz is None:
                    pooled_freq_hz = freq_hz

                window_psd_lookup[window_key].append(psd)
                pooled_psd_arrays.append(psd)

                trace_export = resampled_df.copy()
                trace_export.insert(0, "SegmentIndex", segment_idx)
                trace_export.insert(0, "Period", period_name)
                trace_export.insert(0, "Window", window_key)
                resampled_trace_frames.append(trace_export)

                segment_psd_frames.append(
                    pd.DataFrame(
                        {
                            "Window": window_key,
                            "Period": period_name,
                            "SegmentIndex": segment_idx,
                            "Frequency_Hz": freq_hz,
                            "PSD": psd,
                        }
                    )
                )

                segment_rows.append(
                    {
                        "Window": window_key,
                        "Period": period_name,
                        "SegmentIndex": segment_idx,
                        "SegmentStart_s": segment_start,
                        "SegmentEnd_s": segment_end,
                        "SegmentDuration_s": segment_seconds,
                        "RawBreaths": int(
                            (
                                (trace_df["TimeOfBreath_s"] >= segment_start)
                                & (trace_df["TimeOfBreath_s"] <= segment_end)
                            ).sum()
                        ),
                        "ResampledPoints": int(len(resampled_df)),
                        "ResampleHz": resample_hz,
                        "MeanBeforeCentering_s": float(
                            resampled_df["Ttot_s"].mean()
                        ),
                        "MeanAfterCentering_s": float(
                            resampled_df["TtotCentered_s"].mean()
                        ),
                    }
                )

        if pooled_freq_hz is not None and window_psd_lookup[window_key]:
            per_window_psd_frames.append(
                _mean_psd_table(
                    key_name="Window",
                    key_value=window_key,
                    freq_hz=pooled_freq_hz,
                    psd_arrays=window_psd_lookup[window_key],
                )
            )

    segment_summary_df = pd.DataFrame(segment_rows)
    per_segment_psd_df = (
        pd.concat(segment_psd_frames, ignore_index=True)
        if segment_psd_frames
        else pd.DataFrame(
            columns=["Window", "Period", "SegmentIndex", "Frequency_Hz", "PSD"]
        )
    )
    resampled_traces_df = (
        pd.concat(resampled_trace_frames, ignore_index=True)
        if resampled_trace_frames
        else pd.DataFrame(
            columns=[
                "Window",
                "Period",
                "SegmentIndex",
                "SegmentTime_s",
                "AbsoluteTime_s",
                "Ttot_s",
                "TtotCentered_s",
            ]
        )
    )
    per_window_psd_df = (
        pd.concat(per_window_psd_frames, ignore_index=True)
        if per_window_psd_frames
        else pd.DataFrame(columns=["Window", "Frequency_Hz", "PSD"])
    )
    pooled_psd_df = (
        _mean_psd_table(
            key_name="",
            key_value="",
            freq_hz=pooled_freq_hz,
            psd_arrays=pooled_psd_arrays,
        )
        if pooled_freq_hz is not None and pooled_psd_arrays
        else pd.DataFrame(columns=["Frequency_Hz", "PSD"])
    )

    if not segment_summary_df.empty:
        segment_summary_df.to_csv(output_folder / "ttot_psd_segment_summary.csv", index=False)
    if not per_segment_psd_df.empty:
        per_segment_psd_df.to_csv(output_folder / "ttot_psd_per_segment.csv", index=False)
    if not per_window_psd_df.empty:
        per_window_psd_df.to_csv(output_folder / "ttot_psd_per_window.csv", index=False)
    if not pooled_psd_df.empty:
        pooled_psd_df.to_csv(output_folder / "ttot_psd_pooled.csv", index=False)
    if not resampled_traces_df.empty:
        resampled_traces_df.to_csv(
            output_folder / "ttot_resampled_segments.csv",
            index=False,
        )

    kept_segments = len(segment_summary_df)
    log(
        "Computed Ttot PSD "
        f"for {kept_segments} / {total_candidate_segments} candidate 60 s segments."
    )

    return {
        "config": {
            "Metric": "Ttot",
            "ResampleHz": resample_hz,
            "SegmentDuration_s": segment_seconds,
            "WelchWindows": welch_windows,
            "WelchOverlapFraction": overlap_fraction,
            "NFFT": nfft,
        },
        "segment_summary": segment_summary_df,
        "per_segment_psd": per_segment_psd_df,
        "per_window_psd": per_window_psd_df,
        "pooled_psd": pooled_psd_df,
        "resampled_traces": resampled_traces_df,
    }
