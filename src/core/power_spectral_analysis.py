from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.signal import welch

DEFAULT_PSD_RESAMPLE_HZ: float = 10.0        # Hz
DEFAULT_PSD_SEGMENT_SECONDS: float = 50.0    # s — Nico: 50 s keeps segments stationary
DEFAULT_PSD_END_TRIM_SAMPLES: int = 10       # Nico: drop first/last 10 samples per period
DEFAULT_PSD_WELCH_WINDOWS: int = 3           # 3 equal subwindows, 50% overlap (Bourdillon)
DEFAULT_PSD_OVERLAP_FRACTION: float = 0.5
DEFAULT_PSD_NFFT: int = 2048                 # Nico: zero-pad to 2048 for smooth curve
DEFAULT_PSD_SMOOTH_NPERSEG: int = 100        # Nico: fixed 100-sample window for saved/averaged PSD
DEFAULT_PSD_AUC_BAND_HZ: float = 0.25       # ±Hz around fmax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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

    trace = pd.DataFrame(
        {
            "TimeOfBreath_s": peak_times[:-1],
            "Ttot_s": np.diff(peak_times),
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


def _trim_period_ends(trace_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Drop the first and last n_samples breaths from a period trace.

    Nico trims 10 samples each end to remove edge instability before
    segmenting, avoiding the need for any HP filter or detrend.
    """
    if len(trace_df) <= 2 * n_samples:
        return pd.DataFrame(columns=trace_df.columns)
    return trace_df.iloc[n_samples: len(trace_df) - n_samples].reset_index(drop=True)


def _iter_segment_bounds(
    start_time: float,
    end_time: float,
    segment_seconds: float,
) -> list[tuple[float, float]]:
    """Split a trace span into successive contiguous fixed-length segments."""
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
    """Resample a Ttot trace segment with cubic spline interpolation.

    Uses one breath of padding on either side of the segment boundary so the
    spline never extrapolates. Returns None if coverage is insufficient or if
    the spline produces any NaN.

    No HP filter or detrend is applied here — Nico's approach is to trim
    period ends (removing edge instability) and then mean-subtract inline
    in the Welch call. This preserves the low-frequency content including
    the ~0.02 Hz peak that a 0.02 Hz HP filter would destroy.
    """
    times = trace_df["TimeOfBreath_s"].to_numpy(dtype=float)
    values = trace_df["Ttot_s"].to_numpy(dtype=float)
    if times.size < 4:
        return None

    segment_end = segment_start + segment_seconds
    resampled_time = segment_start + np.arange(
        0.0, segment_seconds, 1.0 / resample_hz, dtype=float
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

    return pd.DataFrame(
        {
            "SegmentTime_s": resampled_time - segment_start,
            "AbsoluteTime_s": resampled_time,
            "Ttot_s": resampled_ttot,
        }
    )


def _compute_welch_psd(
    signal: np.ndarray,
    resample_hz: float,
    smooth_nperseg: int,
    overlap_fraction: float,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD using a fixed nperseg window with zero-padding to nfft.

    Nico uses a fixed 100-sample window (not signal_length/3) with 50%
    overlap and nfft=2048 for the averaged PSD. This gives a smooth curve
    with fine frequency resolution via zero-padding. Mean is subtracted
    inline before the Welch call rather than via a separate centering step.
    """
    if signal.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    nperseg = min(smooth_nperseg, signal.size)
    if nperseg < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    noverlap = int(round(nperseg * overlap_fraction))
    centered = signal - float(np.mean(signal))
    freq_hz, psd = welch(
        centered,
        fs=resample_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
    )
    return freq_hz, psd


def _extract_psd_metrics(
    freq_hz: np.ndarray,
    mean_psd: np.ndarray,
    auc_band: float,
) -> dict[str, float]:
    """Extract fmax, PSDmax, and AUC (±auc_band Hz) from a mean PSD curve."""
    idx = int(np.argmax(mean_psd))
    fmax = float(freq_hz[idx])
    PSDmax = float(mean_psd[idx])
    mask = (freq_hz >= fmax - auc_band) & (freq_hz <= fmax + auc_band)
    AUC = (
        float(simpson(mean_psd[mask], x=freq_hz[mask])) if np.sum(mask) > 1 else np.nan
    )
    return {"fmax": fmax, "PSDmax": PSDmax, "AUC": AUC}


def _mean_psd_table(
    key_name: str,
    key_value: str,
    freq_hz: np.ndarray,
    psd_arrays: list[np.ndarray],
) -> pd.DataFrame:
    """Build a long-format mean PSD DataFrame for one summary group."""
    if not psd_arrays:
        cols = ([key_name] if key_name else []) + ["Frequency_Hz", "PSD"]
        return pd.DataFrame(columns=cols)
    mean_psd = np.mean(np.vstack(psd_arrays), axis=0)
    data: dict[str, Any] = {"Frequency_Hz": freq_hz, "PSD": mean_psd}
    if key_name:
        data = {key_name: [key_value] * len(freq_hz), **data}
    return pd.DataFrame(data)


def _metrics_table(
    per_window_metrics: dict[str, dict[str, float]],
    resample_hz: float,
) -> pd.DataFrame:
    """Build the compact per-window PSD table intended for collation/QC."""
    rows: list[dict[str, Any]] = []
    fmax_high_warn = resample_hz / 2 * 0.8

    for window_key, metrics in per_window_metrics.items():
        fmax = float(metrics.get("fmax", np.nan))
        warning = ""
        if np.isfinite(fmax):
            if fmax == 0.0:
                warning = "fmax at 0 Hz; inspect PSD curve"
            elif fmax > fmax_high_warn:
                warning = "fmax near Nyquist; inspect PSD curve"

        rows.append(
            {
                "Window": window_key,
                "fmax_Hz": fmax,
                "PSDmax_s2_per_Hz": float(metrics.get("PSDmax", np.nan)),
                "AUC_s2": float(metrics.get("AUC", np.nan)),
                "n_segments": int(metrics.get("n_segments", 0)),
                "QC_Warning": warning,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Window",
            "fmax_Hz",
            "PSDmax_s2_per_Hz",
            "AUC_s2",
            "n_segments",
            "QC_Warning",
        ],
    )


def _export_psd_workbook(
    output_folder: Path,
    config: dict[str, Any],
    metrics_df: pd.DataFrame,
    rejection_df: pd.DataFrame,
    segment_summary_df: pd.DataFrame,
    per_window_psd_df: pd.DataFrame,
    log=print,
) -> None:
    """Write the PSD workbook students can send back for easy collation."""
    workbook_path = output_folder / "ttot_psd_summary.xlsx"

    accepted_segments = int(len(segment_summary_df))
    rejected_rows = int(len(rejection_df))
    windows_with_psd = int(metrics_df["Window"].nunique()) if not metrics_df.empty else 0
    windows_with_warnings = (
        int((metrics_df["QC_Warning"].fillna("") != "").sum())
        if not metrics_df.empty
        else 0
    )

    summary_rows = [
        {"Field": "Metric", "Value": config.get("Metric", "Ttot")},
        {"Field": "AcceptedSegments", "Value": accepted_segments},
        {"Field": "RejectedPeriodsOrSegments", "Value": rejected_rows},
        {"Field": "WindowsWithPSD", "Value": windows_with_psd},
        {"Field": "WindowsWithQCWarnings", "Value": windows_with_warnings},
        {"Field": "SummaryPlotPNG", "Value": "ttot_psd_summary.png"},
        {"Field": "SummaryPlotSVG", "Value": "ttot_psd_summary.svg"},
    ]
    summary_rows.extend(
        {"Field": key, "Value": value}
        for key, value in config.items()
        if key != "Metric"
    )
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Run Summary", index=False)
        metrics_df.to_excel(writer, sheet_name="Window Metrics", index=False)
        rejection_df.to_excel(writer, sheet_name="Rejected", index=False)
        segment_summary_df.to_excel(writer, sheet_name="Segment Summary", index=False)
        per_window_psd_df.to_excel(writer, sheet_name="Mean PSD Curves", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                adjusted_width = min(max(max_length + 2, 12), 48)
                sheet.column_dimensions[column_cells[0].column_letter].width = (
                    adjusted_width
                )

    log(f"Ttot PSD workbook saved: {workbook_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_ttot_traces(
    window_periods: dict[str, list[dict[str, Any]]],
    output_folder: Path,
    log=print,
) -> None:
    """Export raw breath-by-breath Ttot traces to CSV for each behavioural window.

    Columns: TimeOfBreath_s, Ttot_s, PeriodIndex
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
            continue
        df = pd.DataFrame(rows)
        safe_key = window_key.replace(":", "_").replace("-", "_").replace(" ", "_")
        out_path = output_folder / f"Ttot_{safe_key}.csv"
        df.to_csv(out_path, index=False)

    n = len(window_periods)
    log(f"Ttot traces exported ({n} windows): {output_folder}")


def analyze_ttot_psd(
    window_periods: dict[str, list[dict[str, Any]]],
    output_folder: Path,
    log=print,
    resample_hz: float = DEFAULT_PSD_RESAMPLE_HZ,
    segment_seconds: float = DEFAULT_PSD_SEGMENT_SECONDS,
    end_trim_samples: int = DEFAULT_PSD_END_TRIM_SAMPLES,
    welch_windows: int = DEFAULT_PSD_WELCH_WINDOWS,
    overlap_fraction: float = DEFAULT_PSD_OVERLAP_FRACTION,
    nfft: int = DEFAULT_PSD_NFFT,
    smooth_nperseg: int = DEFAULT_PSD_SMOOTH_NPERSEG,
    auc_band_hz: float = DEFAULT_PSD_AUC_BAND_HZ,
) -> dict[str, Any]:
    """Compute Ttot variability PSD per behavioural window (Bourdillon / Nico).

    Pipeline per valid period:
      1. Build breath-by-breath Ttot series from peak_times.
      2. Trim first and last `end_trim_samples` breaths from each period to
         remove edge instability (Nico: 10 samples each end).
      3. Split trimmed trace into contiguous `segment_seconds` segments;
         discard any period shorter than one full segment.
      4. Resample each segment at `resample_hz` Hz via cubic spline.
      5. Welch PSD: fixed `smooth_nperseg`-sample window, 50% overlap,
         nfft=2048 zero-padding, mean subtracted inline (Nico approach).
         No HP filter or detrend — edge trimming handles stationarity.
      6. Average PSDs across all segments per window → representative PSD.
      7. Extract fmax, PSDmax, AUC (±`auc_band_hz` Hz) from the mean PSD.

    Writes CSV outputs to `output_folder` and returns DataFrames for export.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, Any]] = []
    segment_psd_frames: list[pd.DataFrame] = []
    resampled_trace_frames: list[pd.DataFrame] = []
    per_window_psd_frames: list[pd.DataFrame] = []
    rejection_rows: list[dict[str, Any]] = []

    pooled_psd_arrays: list[np.ndarray] = []
    pooled_freq_hz: np.ndarray | None = None
    window_psd_lookup: dict[str, list[np.ndarray]] = {}
    per_window_metrics: dict[str, dict[str, float]] = {}

    total_candidate_segments = 0

    for window_key, periods in window_periods.items():
        window_psd_lookup.setdefault(window_key, [])

        for period_idx, period in enumerate(periods, start=1):
            period_name = period.get("name", f"Period_{period_idx}")
            trace_df = _build_ttot_trace(period)
            if trace_df.empty:
                rejection_rows.append({
                    "Window": window_key,
                    "Period": period_name,
                    "RejectionReason": "no valid breaths after sanitization",
                    "PeriodDuration_s": np.nan,
                    "RawBreaths": 0,
                    "SegmentIndex": np.nan,
                })
                continue

            # Trim period ends before segmenting (Nico: 10 samples each end)
            trace_df = _trim_period_ends(trace_df, end_trim_samples)
            if trace_df.empty:
                rejection_rows.append({
                    "Window": window_key,
                    "Period": period_name,
                    "RejectionReason": f"too few breaths after trimming {end_trim_samples} samples each end",
                    "PeriodDuration_s": np.nan,
                    "RawBreaths": 0,
                    "SegmentIndex": np.nan,
                })
                continue

            trace_start = float(trace_df["TimeOfBreath_s"].iloc[0])
            trace_end = float(trace_df["TimeOfBreath_s"].iloc[-1])
            period_duration = trace_end - trace_start
            n_raw_breaths = len(trace_df)

            segment_bounds = _iter_segment_bounds(
                trace_start, trace_end, segment_seconds
            )

            if not segment_bounds:
                rejection_rows.append({
                    "Window": window_key,
                    "Period": period_name,
                    "RejectionReason": f"period too short for one {segment_seconds:.0f}s segment ({period_duration:.1f}s after trimming)",
                    "PeriodDuration_s": round(period_duration, 2),
                    "RawBreaths": n_raw_breaths,
                    "SegmentIndex": np.nan,
                })
                continue

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
                    rejection_rows.append({
                        "Window": window_key,
                        "Period": period_name,
                        "RejectionReason": "resampling failed (insufficient coverage or NaN in spline)",
                        "PeriodDuration_s": round(period_duration, 2),
                        "RawBreaths": n_raw_breaths,
                        "SegmentIndex": segment_idx,
                    })
                    continue

                freq_hz, psd = _compute_welch_psd(
                    signal=resampled_df["Ttot_s"].to_numpy(dtype=float),
                    resample_hz=resample_hz,
                    smooth_nperseg=smooth_nperseg,
                    overlap_fraction=overlap_fraction,
                    nfft=nfft,
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
                        "MeanTtot_s": float(resampled_df["Ttot_s"].mean()),
                    }
                )

        if pooled_freq_hz is not None and window_psd_lookup[window_key]:
            window_psds = window_psd_lookup[window_key]
            window_mean_psd = np.mean(np.vstack(window_psds), axis=0)
            per_window_psd_frames.append(
                _mean_psd_table(
                    key_name="Window",
                    key_value=window_key,
                    freq_hz=pooled_freq_hz,
                    psd_arrays=window_psds,
                )
            )
            metrics = _extract_psd_metrics(pooled_freq_hz, window_mean_psd, auc_band_hz)
            metrics["n_segments"] = len(window_psds)
            per_window_metrics[window_key] = metrics

    # --- Warn if fmax looks implausible ---
    # Nico found peaks at ~0.02 Hz and ~1 Hz in this mouse data.
    # Flag anything at DC (0 Hz exactly) or approaching Nyquist.
    _FMAX_HIGH_WARN = resample_hz / 2 * 0.8
    suspicious = {
        k: m["fmax"]
        for k, m in per_window_metrics.items()
        if m["fmax"] == 0.0 or m["fmax"] > _FMAX_HIGH_WARN
    }
    if suspicious:
        lines = ", ".join(f"{k}: {v:.4f} Hz" for k, v in suspicious.items())
        log(
            f"WARNING — fmax at DC or near Nyquist for: {lines}. "
            "Check PSD curve manually."
        )

    # --- Assemble DataFrames ---
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
            columns=["Window", "Period", "SegmentIndex", "SegmentTime_s",
                     "AbsoluteTime_s", "Ttot_s"]
        )
    )
    per_window_psd_df = (
        pd.concat(per_window_psd_frames, ignore_index=True)
        if per_window_psd_frames
        else pd.DataFrame(columns=["Window", "Frequency_Hz", "PSD"])
    )
    pooled_psd_df = (
        _mean_psd_table("", "", pooled_freq_hz, pooled_psd_arrays)
        if pooled_freq_hz is not None and pooled_psd_arrays
        else pd.DataFrame(columns=["Frequency_Hz", "PSD"])
    )

    # --- Export CSVs ---
    rejection_df = pd.DataFrame(rejection_rows) if rejection_rows else pd.DataFrame(
        columns=["Window", "Period", "RejectionReason", "PeriodDuration_s", "RawBreaths", "SegmentIndex"]
    )
    rejection_df.to_csv(output_folder / "ttot_psd_rejected.csv", index=False)
    log(f"Ttot PSD: {len(rejection_df)} periods/segments rejected — see ttot_psd_rejected.csv")

    if not segment_summary_df.empty:
        segment_summary_df.to_csv(
            output_folder / "ttot_psd_segment_summary.csv", index=False
        )
    if not per_segment_psd_df.empty:
        per_segment_psd_df.to_csv(
            output_folder / "ttot_psd_per_segment.csv", index=False
        )
    if not per_window_psd_df.empty:
        per_window_psd_df.to_csv(
            output_folder / "ttot_psd_per_window.csv", index=False
        )
    if not pooled_psd_df.empty:
        pooled_psd_df.to_csv(output_folder / "ttot_psd_pooled.csv", index=False)
    if not resampled_traces_df.empty:
        resampled_traces_df.to_csv(
            output_folder / "ttot_resampled_segments.csv", index=False
        )

    kept_segments = len(segment_summary_df)
    log(
        f"Ttot PSD: {kept_segments} / {total_candidate_segments} "
        f"candidate {segment_seconds:.0f}s segments used."
    )

    config = {
        "Metric": "Ttot",
        "AnalysisID": output_folder.parent.name,
        "ResampleHz": resample_hz,
        "SegmentDuration_s": segment_seconds,
        "EndTrimSamples": end_trim_samples,
        "WelchSmoothNperseg": smooth_nperseg,
        "WelchOverlapFraction": overlap_fraction,
        "NFFT": nfft,
        "AUCBandHz": auc_band_hz,
    }
    window_metrics_df = _metrics_table(per_window_metrics, resample_hz)
    window_metrics_df.insert(0, "AnalysisID", output_folder.parent.name)
    if not rejection_df.empty:
        rejection_df.insert(0, "AnalysisID", output_folder.parent.name)
    if not segment_summary_df.empty:
        segment_summary_df.insert(0, "AnalysisID", output_folder.parent.name)
    window_metrics_df.to_csv(output_folder / "ttot_psd_window_metrics.csv", index=False)
    _export_psd_workbook(
        output_folder=output_folder,
        config=config,
        metrics_df=window_metrics_df,
        rejection_df=rejection_df,
        segment_summary_df=segment_summary_df,
        per_window_psd_df=per_window_psd_df,
        log=log,
    )

    return {
        "config": config,
        "segment_summary": segment_summary_df,
        "per_segment_psd": per_segment_psd_df,
        "per_window_psd": per_window_psd_df,
        "pooled_psd": pooled_psd_df,
        "resampled_traces": resampled_traces_df,
        "per_window_metrics": per_window_metrics,
        "window_metrics": window_metrics_df,
        "rejection_log": rejection_df,
    }


def _plot_psd_results_interactive_legacy(psd_results: dict[str, Any]) -> None:
    """4-panel figure from analyze_ttot_psd output:
      1. Mean PSD curves per window (linear scale — Nico uses linear)
      2. fmax per window
      3. PSDmax per window
      4. AUC per window
    """
    per_window_psd_df: pd.DataFrame = psd_results.get("per_window_psd", pd.DataFrame())
    per_window_metrics: dict[str, dict[str, float]] = psd_results.get(
        "per_window_metrics", {}
    )
    auc_band = psd_results.get("config", {}).get("AUCBandHz", DEFAULT_PSD_AUC_BAND_HZ)
    segment_seconds = psd_results.get("config", {}).get("SegmentDuration_s", DEFAULT_PSD_SEGMENT_SECONDS)

    if per_window_psd_df.empty or not per_window_metrics:
        return

    window_keys = list(per_window_metrics.keys())
    fmax_vals = [per_window_metrics[k]["fmax"] for k in window_keys]
    psdmax_vals = [per_window_metrics[k]["PSDmax"] for k in window_keys]
    auc_vals = [per_window_metrics[k]["AUC"] for k in window_keys]
    x = np.arange(len(window_keys))

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2)
    ax_psd = fig.add_subplot(grid[0, 0])
    ax_fmax = fig.add_subplot(grid[0, 1])
    ax_psdmax = fig.add_subplot(grid[1, 0])
    ax_auc = fig.add_subplot(grid[1, 1])

    colors = plt.cm.tab10(np.linspace(0, 1, len(window_keys)))

    for color, window_key in zip(colors, window_keys, strict=False):
        subset = per_window_psd_df[per_window_psd_df["Window"] == window_key]
        if subset.empty:
            continue
        n_seg = int(per_window_metrics[window_key].get("n_segments", 0))
        ax_psd.plot(
            subset["Frequency_Hz"],
            subset["PSD"],
            color=color,
            alpha=0.8,
            linewidth=1.0,
            label=f"{window_key} (n={n_seg})",
        )

    ax_psd.set_title(f"Mean PSD per window — Ttot variability ({segment_seconds:.0f}s segments)")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (s²/Hz)")
    ax_psd.legend(fontsize=6)

    ax_fmax.plot(x, fmax_vals, marker="o")
    ax_fmax.set_title("fmax per window")
    ax_fmax.set_xlabel("Window index")
    ax_fmax.set_ylabel("Frequency (Hz)")

    ax_psdmax.plot(x, psdmax_vals, marker="o")
    ax_psdmax.set_title("PSDmax per window")
    ax_psdmax.set_xlabel("Window index")
    ax_psdmax.set_ylabel("Power (s²/Hz)")

    ax_auc.plot(x, auc_vals, marker="o")
    ax_auc.set_title(f"AUC \u00b1{auc_band} Hz around fmax")
    ax_auc.set_xlabel("Window index")
    ax_auc.set_ylabel("Area (s²)")

    plt.tight_layout()
    plot_psd_results(psd_results, show=True)


def plot_psd_results(
    psd_results: dict[str, Any],
    output_folder: Path | None = None,
    *,
    show: bool = False,
    log=print,
) -> None:
    """Save a 4-panel PSD QC figure; optionally show it interactively."""
    per_window_psd_df: pd.DataFrame = psd_results.get("per_window_psd", pd.DataFrame())
    per_window_metrics: dict[str, dict[str, float]] = psd_results.get(
        "per_window_metrics", {}
    )
    auc_band = psd_results.get("config", {}).get("AUCBandHz", DEFAULT_PSD_AUC_BAND_HZ)
    segment_seconds = psd_results.get("config", {}).get(
        "SegmentDuration_s", DEFAULT_PSD_SEGMENT_SECONDS
    )

    if per_window_psd_df.empty or not per_window_metrics:
        log("Ttot PSD plot skipped: no per-window PSD results.")
        return

    window_keys = list(per_window_metrics.keys())
    fmax_vals = [per_window_metrics[k]["fmax"] for k in window_keys]
    psdmax_vals = [per_window_metrics[k]["PSDmax"] for k in window_keys]
    auc_vals = [per_window_metrics[k]["AUC"] for k in window_keys]
    x = np.arange(len(window_keys))

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2)
    ax_psd = fig.add_subplot(grid[0, 0])
    ax_fmax = fig.add_subplot(grid[0, 1])
    ax_psdmax = fig.add_subplot(grid[1, 0])
    ax_auc = fig.add_subplot(grid[1, 1])

    colors = plt.cm.tab10(np.linspace(0, 1, len(window_keys)))

    for color, window_key in zip(colors, window_keys, strict=False):
        subset = per_window_psd_df[per_window_psd_df["Window"] == window_key]
        if subset.empty:
            continue
        n_seg = int(per_window_metrics[window_key].get("n_segments", 0))
        fmax = float(per_window_metrics[window_key].get("fmax", np.nan))
        psdmax = float(per_window_metrics[window_key].get("PSDmax", np.nan))
        ax_psd.plot(
            subset["Frequency_Hz"],
            subset["PSD"],
            color=color,
            alpha=0.8,
            linewidth=1.0,
            label=f"{window_key} (n={n_seg})",
        )
        if np.isfinite(fmax) and np.isfinite(psdmax):
            ax_psd.scatter([fmax], [psdmax], color=color, s=24, zorder=3)

    ax_psd.set_title(
        f"Mean PSD per window - Ttot variability ({segment_seconds:.0f}s segments)"
    )
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (s^2/Hz)")
    ax_psd.legend(fontsize=6)

    ax_fmax.bar(x, fmax_vals)
    ax_fmax.set_title("fmax per window")
    ax_fmax.set_ylabel("Frequency (Hz)")
    ax_fmax.set_xticks(x, window_keys, rotation=45, ha="right")

    ax_psdmax.bar(x, psdmax_vals)
    ax_psdmax.set_title("PSDmax per window")
    ax_psdmax.set_ylabel("Power (s^2/Hz)")
    ax_psdmax.set_xticks(x, window_keys, rotation=45, ha="right")

    ax_auc.bar(x, auc_vals)
    ax_auc.set_title(f"AUC +/-{auc_band} Hz around fmax")
    ax_auc.set_ylabel("Area (s^2)")
    ax_auc.set_xticks(x, window_keys, rotation=45, ha="right")

    plt.tight_layout()
    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)
        png_path = output_folder / "ttot_psd_summary.png"
        svg_path = output_folder / "ttot_psd_summary.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        log(f"Ttot PSD summary plot saved: {png_path}")
        log(f"Ttot PSD summary plot saved: {svg_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
