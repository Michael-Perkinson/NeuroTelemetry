from typing import Any

import numpy as np
import pandas as pd

from src.core.logger import log_info
from src.core.respiratory_metrics import (
    calculate_binned_period_metrics,
    calculate_valid_period_metrics,
    summarize_respiratory_cycles,
)


def find_valid_periods(
    results: dict[
        str,
        list[
            tuple[
                float,
                float,
                np.ndarray,
                pd.Series,
                np.ndarray,
                pd.Series,
            ]
        ],
    ],
    time_windows: list[tuple[float, float]],
) -> tuple[
    list[float],
    list[float],
    list[tuple[float, float]],
    dict[str, list[dict[str, Any]]],
]:
    """
    Identify valid respiratory periods from peak/trough data and return
    period definitions without computing any metrics.
    """

    valid_peak_times_all: list[float] = []
    valid_pre_peak_times_all: list[float] = []
    updated_valid_periods: list[tuple[float, float]] = []

    # Per-window period containers
    window_periods: dict[str, list[dict[str, Any]]] = {}

    for window_start_time, window_end_time in time_windows:
        window_key = f"{window_start_time}-{window_end_time}"
        window_results = results.get(window_key, [])
        if not window_results:
            log_info(f"No results found for time window {window_key}. Skipping.")
            continue

        window_period_list: list[dict[str, Any]] = []

        for result in window_results:
            (
                start_time,
                end_time,
                raw_peaks,
                peak_times,
                raw_pre_peak_indices,
                pre_peak_times,
            ) = result

            # Convert arrays to Series
            peaks = pd.Series(raw_peaks, index=peak_times.index)
            pre_peak_indices = pd.Series(
                raw_pre_peak_indices, index=pre_peak_times.index
            )

            valid_periods = identify_new_periods(start_time, end_time, peak_times)

            if not valid_periods:
                log_info(f"No valid periods found for {window_key}. Skipping.")
                continue

            for i, (period_start_time, period_end_time) in enumerate(valid_periods):
                # Ensure period fully within this window
                if not (
                    window_start_time <= period_start_time <= window_end_time
                    and window_start_time <= period_end_time <= window_end_time
                ):
                    continue

                # Masks
                peak_mask = (peak_times >= period_start_time) & (
                    peak_times <= period_end_time
                )
                trough_mask = (pre_peak_times >= period_start_time) & (
                    pre_peak_times <= period_end_time
                )

                # Extract series
                period_peak_times = peak_times[peak_mask]
                period_trough_times = pre_peak_times[trough_mask]
                period_peaks = peaks[peak_mask]
                period_peak_indices = peaks[peak_mask]
                period_trough_indices = pre_peak_indices[trough_mask]

                # Align lengths (same logic as original)
                if len(period_peak_times) > len(period_trough_times):
                    period_peak_times = period_peak_times[1:]
                    period_peaks = period_peaks[1:]
                    period_peak_indices = period_peak_indices[1:]
                elif len(period_trough_times) > len(period_peak_times):
                    period_trough_times = period_trough_times[:-1]
                    period_trough_indices = period_trough_indices[:-1]

                if len(period_peak_times) < 2 or len(period_trough_times) < 2:
                    continue

                period_name = f"Period_{i + 1}_{period_start_time}-{period_end_time}"

                # Store per-period info
                window_period_list.append(
                    {
                        "name": period_name,
                        "start_time": period_start_time,
                        "end_time": period_end_time,
                        "peak_times": period_peak_times,
                        "trough_times": period_trough_times,
                        "peak_indices": period_peak_indices,
                        "trough_indices": period_trough_indices,
                    }
                )

                # Global flattened time lists for summary_data etc.
                valid_peak_times_all.extend(period_peak_times.tolist())
                valid_pre_peak_times_all.extend(period_trough_times.tolist())
                updated_valid_periods.append((period_start_time, period_end_time))

        if window_period_list:
            window_periods[window_key] = window_period_list

    return (
        valid_peak_times_all,
        valid_pre_peak_times_all,
        updated_valid_periods,
        window_periods,
    )


def compute_respiratory_metrics_for_periods(
    window_periods: dict[str, list[dict[str, Any]]],
    pressure_data: pd.DataFrame,
    temp_data: pd.DataFrame | None,
    activity_data: pd.DataFrame | None,
    time_windows: list[tuple[float, float]],
    bin_size_sec: int,
) -> dict[str, dict[str, Any]]:
    """
    Compute per-period, per-window, and global respiratory metrics
    from period definitions.
    """

    all_metrics: dict[str, dict[str, Any]] = {}

    # For global summary
    all_peak_series: list[pd.Series] = []
    all_trough_series: list[pd.Series] = []

    # Per-window loop in canonical order of time_windows
    for window_start_time, window_end_time in time_windows:
        window_key = f"{window_start_time}-{window_end_time}"
        periods = window_periods.get(window_key, [])
        if not periods:
            log_info(f"No valid periods stored for time window {window_key}. Skipping.")
            continue

        window_periods_metrics: dict[str, dict[str, Any]] = {}
        window_all_peak_series: list[pd.Series] = []
        window_all_trough_series: list[pd.Series] = []

        for period_info in periods:
            period_name = period_info["name"]
            period_start_time = period_info["start_time"]
            period_end_time = period_info["end_time"]
            period_peak_times: pd.Series = period_info["peak_times"]
            period_trough_times: pd.Series = period_info["trough_times"]

            # --- binned metrics per period ---
            binned_df = calculate_binned_period_metrics(
                period_start_time,
                period_end_time,
                bin_size_sec,
                period_peak_times,
                period_trough_times,
                pressure_data,
                temp_data if temp_data is not None else pd.DataFrame(),
                activity_data if activity_data is not None else pd.DataFrame(),
            )

            # --- summary metrics per period ---
            summary_metrics = calculate_valid_period_metrics(
                period_peak_times, period_trough_times, pressure_data
            )

            window_periods_metrics[period_name] = {
                "Binned": binned_df,
                "Summary": summary_metrics,
            }

            window_all_peak_series.append(period_peak_times)
            window_all_trough_series.append(period_trough_times)

            all_peak_series.append(period_peak_times)
            all_trough_series.append(period_trough_times)

        # Per-window summary across all periods in that window
        window_summary = summarize_respiratory_cycles(
            window_all_peak_series, window_all_trough_series, pressure_data
        )

        all_metrics[window_key] = {
            "WindowSummary": window_summary,
            "Periods": window_periods_metrics,
        }

    # Global summary across all windows/periods
    global_summary = summarize_respiratory_cycles(
        all_peak_series, all_trough_series, pressure_data
    )
    all_metrics["GlobalSummary"] = global_summary

    return all_metrics


def identify_new_periods(
    start_time: float,
    end_time: float,
    peak_times: pd.Series,
    min_peaks: int = 50,
    interval_window: int = 10,
    break_multiplier: float = 2.0,
) -> list[tuple[float, float]]:
    """Detect valid respiratory periods based on spacing between peaks."""
    peak_times_filtered = (
        peak_times[(peak_times >= start_time) & (peak_times <= end_time)]
        .sort_values()
        .to_numpy()
    )

    if len(peak_times_filtered) < min_peaks:
        log_info("Not enough peaks in the window to form a valid period.")
        return []

    valid_periods = []
    current_period_start = peak_times_filtered[0]
    current_period_peaks = [peak_times_filtered[0]]
    recent_intervals: list[float] = []

    for i in range(1, len(peak_times_filtered)):
        current_peak = peak_times_filtered[i]
        previous_peak = peak_times_filtered[i - 1]
        interval = current_peak - previous_peak

        recent_intervals.append(interval)
        if len(recent_intervals) > interval_window:
            recent_intervals.pop(0)

        median_interval = np.median(recent_intervals)

        if interval > break_multiplier * median_interval:
            if len(current_period_peaks) >= min_peaks:
                period_end = previous_peak
                valid_periods.append((current_period_start, period_end))
                log_info(
                    f"Valid period: {current_period_start:.3f}s to {period_end:.3f}s "
                    f"with {len(current_period_peaks)} peaks"
                )
            else:
                log_info(
                    f"Discarded: {current_period_start:.3f}s to "
                    f"{previous_peak:.3f}s with only {len(current_period_peaks)} peaks"
                )

            current_period_start = current_peak
            current_period_peaks = [current_peak]
            recent_intervals = []
            continue

        current_period_peaks.append(current_peak)

    if len(current_period_peaks) >= min_peaks:
        period_end = peak_times_filtered[-1]
        valid_periods.append((current_period_start, period_end))
        log_info(
            f"Valid period: {current_period_start:.3f}s to "
            f"{period_end:.3f}s with {len(current_period_peaks)} peaks"
        )
    else:
        log_info(
            f"Discarded last: {current_period_start:.3f}s to "
            f"{peak_times_filtered[-1]:.3f}s with only "
            f"{len(current_period_peaks)} peaks"
        )

    return valid_periods
