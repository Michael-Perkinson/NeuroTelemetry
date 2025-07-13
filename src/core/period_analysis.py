import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from src.core.respiratory_metrics import calculate_binned_period_metrics, calculate_valid_period_metrics, summarize_respiratory_cycles


def find_valid_periods(
    results: Dict[str, List[Tuple[float, float, np.ndarray, pd.Series, np.ndarray, pd.Series]]],
    pressure_data: pd.DataFrame,
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame,
    time_windows: List[Tuple[float, float]],
    bin_size_sec: int
) -> Tuple[
    List[float], List[float], List[Tuple[float, float]],
    Dict[str, Dict[str, Any]]
]:
    """
    Identify valid respiratory periods from peak/trough data and compute metrics.

    Returns:
        valid_peak_times_all: Flattened list of all peak times in valid periods.
        valid_pre_peak_times_all: Flattened list of all trough times in valid periods.
        updated_valid_periods: List of all valid (start, end) period tuples.
        all_metrics: Structured dictionary for export.
    """
    valid_peak_times_all = []
    valid_pre_peak_times_all = []
    updated_valid_periods = []
    all_metrics: Dict[str, Dict[str, Any]] = {}
    all_peak_series: List[pd.Series] = []
    all_trough_series: List[pd.Series] = []

    for window_start_time, window_end_time in time_windows:
        window_key = f"{window_start_time}-{window_end_time}"
        window_results = results.get(window_key, [])
        if not window_results:
            print(f"No results found for time window {window_key}. Skipping.")
            continue

        window_periods = {}
        window_all_peak_series: List[pd.Series] = []
        window_all_trough_series: List[pd.Series] = []

        for result in window_results:
            start_time, end_time, raw_peaks, peak_times, raw_pre_peak_indices, pre_peak_times = result

            # Convert arrays to Series
            peaks = pd.Series(raw_peaks, index=peak_times.index)
            pre_peak_indices = pd.Series(
                raw_pre_peak_indices, index=pre_peak_times.index)

            valid_periods = identify_new_periods(
                start_time, end_time, peak_times, pressure_data
            )

            if not valid_periods:
                print(f"No valid periods found for {window_key}. Skipping.")
                continue

            for i, (period_start_time, period_end_time) in enumerate(valid_periods):
                if not (window_start_time <= period_start_time <= window_end_time and
                        window_start_time <= period_end_time <= window_end_time):
                    continue

                # Masks
                peak_mask = (peak_times >= period_start_time) & (
                    peak_times <= period_end_time)
                trough_mask = (pre_peak_times >= period_start_time) & (
                    pre_peak_times <= period_end_time)

                # Extract series
                period_peak_times = peak_times[peak_mask]
                period_trough_times = pre_peak_times[trough_mask]
                period_peaks = peaks[peak_mask]
                period_peak_indices = peaks[peak_mask]
                period_trough_indices = pre_peak_indices[trough_mask]

                # Align lengths
                if len(period_peak_times) > len(period_trough_times):
                    period_peak_times = period_peak_times[1:]
                    period_peaks = period_peaks[1:]
                    period_peak_indices = period_peak_indices[1:]
                elif len(period_trough_times) > len(period_peak_times):
                    period_trough_times = period_trough_times[:-1]
                    period_trough_indices = period_trough_indices[:-1]

                # --- Metrics ---
                period_name = f"Period_{i+1}_{period_start_time}-{period_end_time}"

                binned_df = calculate_binned_period_metrics(
                    period_start_time, period_end_time, bin_size_sec,
                    period_peak_times, period_trough_times,
                    pressure_data, temp_data, activity_data
                )

                summary_metrics = calculate_valid_period_metrics(
                    period_peak_times,
                    period_trough_times,
                    pressure_data
                )

                # Store
                window_periods[period_name] = {
                    "Binned": binned_df,
                    "Summary": summary_metrics
                }
                window_all_peak_series.append(period_peak_times)
                window_all_trough_series.append(period_trough_times)

                all_peak_series.append(period_peak_times)
                all_trough_series.append(period_trough_times)

                valid_peak_times_all.extend(period_peak_times)
                valid_pre_peak_times_all.extend(period_trough_times)
                updated_valid_periods.append(
                    (period_start_time, period_end_time))

        window_summary = summarize_respiratory_cycles(
            window_all_peak_series,
            window_all_trough_series,
            pressure_data
        )

        all_metrics[window_key] = {
            "WindowSummary": window_summary,
            "Periods": window_periods
        }
        
    global_summary = summarize_respiratory_cycles(
        all_peak_series,
        all_trough_series,
        pressure_data
    )
    all_metrics["GlobalSummary"] = global_summary
    
    return valid_peak_times_all, valid_pre_peak_times_all, updated_valid_periods, all_metrics


def identify_new_periods(
    start_time: float,
    end_time: float,
    peak_times: pd.Series,
    pressure_data: pd.DataFrame,
    min_peaks: int = 50,
    interval_window: int = 10,
    break_multiplier: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Detect valid respiratory periods based on spacing between peaks.
    """
    peak_times_filtered = peak_times[(peak_times >= start_time) & (
        peak_times <= end_time)].sort_values().to_numpy()

    if len(peak_times_filtered) < min_peaks:
        print("Not enough peaks in the window to form a valid period.")
        return []

    valid_periods = []
    current_period_start = peak_times_filtered[0]
    current_period_peaks = [peak_times_filtered[0]]
    recent_intervals: List[float] = []

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
                print(
                    f"Valid period: {current_period_start:.3f}s to {period_end:.3f}s with {len(current_period_peaks)} peaks")
            else:
                print(
                    f"Discarded: {current_period_start:.3f}s to {previous_peak:.3f}s with only {len(current_period_peaks)} peaks")

            current_period_start = current_peak
            current_period_peaks = [current_peak]
            recent_intervals = []
            continue

        current_period_peaks.append(current_peak)

    if len(current_period_peaks) >= min_peaks:
        period_end = peak_times_filtered[-1]
        valid_periods.append((current_period_start, period_end))
        print(
            f"Valid period: {current_period_start:.3f}s to {period_end:.3f}s with {len(current_period_peaks)} peaks")
    else:
        print(
            f"Discarded last: {current_period_start:.3f}s to {peak_times_filtered[-1]:.3f}s with only {len(current_period_peaks)} peaks")

    return valid_periods
