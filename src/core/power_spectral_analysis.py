from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.signal import welch

DEFAULT_PSD_RESAMPLE_HZ: float = 10.0        # Hz — try 7, 10, or 12.5 (Bourdillon)
DEFAULT_PSD_SEGMENT_SECONDS: float = 60.0    # fixed window length (Bourdillon pt. 7)
DEFAULT_PSD_WELCH_WINDOWS: int = 3           # equal subwindows (Bourdillon pt. 8)
DEFAULT_PSD_OVERLAP_FRACTION: float = 0.5   # 50% overlap (Bourdillon pt. 8)
DEFAULT_PSD_NFFT: int = 2048
DEFAULT_PSD_AUC_BAND_HZ: float = 0.25       # ±Hz around fmax (scaled for mice ~2-4 Hz)


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
    """
    Resample a Ttot trace segment with cubic spline interpolation.

    Uses one breath of padding on either side of the 60 s boundary so the
    interpolation never extrapolates. Returns None if coverage is insufficient
    or if the spline produces any NaN.
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

    # Mean-center only — linear detrend not needed since HP filter already applied
    # (Bourdillon, pers. comm. Jan 2026)
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
    """Welch PSD — Bourdillon et al. pt. 8: 3 subwindows, 50% overlap, Hamming."""
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_ttot_traces(
    window_periods: dict[str, list[dict[str, Any]]],
    output_folder: Path,
    log=print,
) -> None:
    """
    Export raw breath-by-breath Ttot traces to CSV for each behavioural window.

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
            log(f"No valid breath data in window {window_key}, skipping CSV.")
            continue
        df = pd.DataFrame(rows)
        safe_key = window_key.replace(":", "_").replace("-", "_").replace(" ", "_")
        out_path = output_folder / f"Ttot_{safe_key}.csv"
        df.to_csv(out_path, index=False)
        log(f"Exported Ttot CSV: {out_path}")


def analyze_ttot_psd(
    window_periods: dict[str, list[dict[str, Any]]],
    output_folder: Path,
    log=print,
    resample_hz: float = DEFAULT_PSD_RESAMPLE_HZ,
    segment_seconds: float = DEFAULT_PSD_SEGMENT_SECONDS,
    welch_windows: int = DEFAULT_PSD_WELCH_WINDOWS,
    overlap_fraction: float = DEFAULT_PSD_OVERLAP_FRACTION,
    nfft: int = DEFAULT_PSD_NFFT,
    auc_band_hz: float = DEFAULT_PSD_AUC_BAND_HZ,
) -> dict[str, Any]:
    """
    Compute Ttot variability PSD per behavioural window (Bourdillon et al.).

    Pipeline per valid period:
      1. Build breath-by-breath Ttot series from peak_times.
      2. Split trace into contiguous 60 s segments by absolute time; discard < 60 s.
      3. Resample each 60 s segment at `resample_hz` Hz (cubic spline, no extrap.).
      4. Mean-center the resampled segment (HP filter already removes drift).
      5. Welch PSD: 3 equal subwindows, 50% overlap, Hamming, nfft=2048.
      6. Average PSDs across all segments per window → representative PSD.
      7. Extract fmax, PSDmax, AUC (±`auc_band_hz` Hz) from the mean PSD.

    Writes CSV outputs to `output_folder` and returns DataFrames for Excel export.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, Any]] = []
    segment_psd_frames: list[pd.DataFrame] = []
    resampled_trace_frames: list[pd.DataFrame] = []
    per_window_psd_frames: list[pd.DataFrame] = []

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
                continue

            trace_start = float(trace_df["TimeOfBreath_s"].iloc[0])
            trace_end = float(trace_df["TimeOfBreath_s"].iloc[-1])
            segment_bounds = _iter_segment_bounds(
                trace_start, trace_end, segment_seconds
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
                    centered_signal=resampled_df["TtotCentered_s"].to_numpy(dtype=float),
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
                        "MeanBeforeCentering_s": float(resampled_df["Ttot_s"].mean()),
                        "MeanAfterCentering_s": float(
                            resampled_df["TtotCentered_s"].mean()
                        ),
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

    # --- Warn if any fmax looks biologically implausible (mouse Ttot variability) ---
    # fmax should sit in the low-frequency range of the variability spectrum, well below
    # the actual breathing rate (~2–4 Hz in mice). Values near DC or approaching Nyquist
    # (resample_hz / 2) are almost always noise artefacts.
    _FMAX_LOW_WARN = 0.05   # Hz — near-DC, variability signal unlikely
    _FMAX_HIGH_WARN = resample_hz / 2 * 0.8  # 80% of Nyquist — suspiciously high
    suspicious = {
        k: m["fmax"]
        for k, m in per_window_metrics.items()
        if m["fmax"] < _FMAX_LOW_WARN or m["fmax"] > _FMAX_HIGH_WARN
    }
    if suspicious:
        lines = ", ".join(f"{k}: {v:.3f} Hz" for k, v in suspicious.items())
        log(
            f"WARNING — fmax outside expected range ({_FMAX_LOW_WARN}–"
            f"{_FMAX_HIGH_WARN:.2f} Hz) for: {lines}. "
            "Check the PSD curve manually — may be noise."
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
                     "AbsoluteTime_s", "Ttot_s", "TtotCentered_s"]
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
        "candidate 60 s segments used."
    )

    return {
        "config": {
            "Metric": "Ttot",
            "ResampleHz": resample_hz,
            "SegmentDuration_s": segment_seconds,
            "WelchWindows": welch_windows,
            "WelchOverlapFraction": overlap_fraction,
            "NFFT": nfft,
            "AUCBandHz": auc_band_hz,
        },
        "segment_summary": segment_summary_df,
        "per_segment_psd": per_segment_psd_df,
        "per_window_psd": per_window_psd_df,
        "pooled_psd": pooled_psd_df,
        "resampled_traces": resampled_traces_df,
        "per_window_metrics": per_window_metrics,
    }


def plot_psd_results(psd_results: dict[str, Any]) -> None:
    """
    4-panel figure from analyze_ttot_psd output:
      1. Mean PSD curves per window
      2. fmax per window
      3. PSDmax per window
      4. AUC per window
    """
    per_window_psd_df: pd.DataFrame = psd_results.get("per_window_psd", pd.DataFrame())
    per_window_metrics: dict[str, dict[str, float]] = psd_results.get(
        "per_window_metrics", {}
    )
    auc_band = psd_results.get("config", {}).get("AUCBandHz", DEFAULT_PSD_AUC_BAND_HZ)

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
        ax_psd.semilogy(
            subset["Frequency_Hz"],
            subset["PSD"],
            color=color,
            alpha=0.7,
            linewidth=0.8,
            label=f"{window_key} (n={n_seg})",
        )

    ax_psd.set_title("Mean PSD per window — Ttot variability (60 s segments)")
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
    plt.show()
