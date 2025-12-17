from typing import Any

import numpy as np
import pandas as pd


def calculate_binned_period_metrics(
    period_start_time: float,
    period_end_time: float,
    bin_size_sec: int,
    peak_times: pd.Series,
    trough_times: pd.Series,
    pressure_data: pd.DataFrame,
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bins data and computes respiratory, temperature, and activity metrics.
    Assumes peak_times and trough_times are pd.Series with indices that map
    to pressure_data rows.
    """
    bin_edges = np.arange(
        period_start_time, period_end_time + bin_size_sec, bin_size_sec
    )

    if (period_end_time - period_start_time) % bin_size_sec != 0:
        bin_edges = bin_edges[:-1]

    binned_metrics: list[dict[str, Any]] = []

    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Masks
        bin_peak_mask = (peak_times >= bin_start) & (peak_times < bin_end)
        bin_trough_mask = (trough_times >= bin_start) & (trough_times < bin_end)

        bin_peaks = peak_times[bin_peak_mask]
        bin_troughs = trough_times[bin_trough_mask]

        bin_temp = temp_data[
            (temp_data["TimeSinceReference"] >= bin_start) &
            (temp_data["TimeSinceReference"] < bin_end)
        ]

        bin_activity = activity_data[
            (activity_data["TimeSinceReference"] >= bin_start) &
            (activity_data["TimeSinceReference"] < bin_end)
        ]

        if len(bin_peaks) < 2 or len(bin_troughs) < 2:
            print(f"Skipping bin {i + 1} due to insufficient peaks or troughs.")
            continue

        # Calculate respiratory metrics
        bin_metrics = calculate_respiratory_metrics(
            bin_peaks, bin_peaks, bin_troughs, pressure_data
        )
        bin_metrics["Bin_Start"] = float(bin_start)
        bin_metrics["Bin_End"] = float(bin_end)

        # Temperature metrics
        if not bin_temp.empty and "Temp" in bin_temp.columns:
            bin_metrics["Mean_Temperature"] = bin_temp["Temp"].mean()
            bin_metrics["SEM_Temperature"] = bin_temp["Temp"].std() / np.sqrt(
                bin_temp["Temp"].count()
            )
        else:
            bin_metrics["Mean_Temperature"] = np.nan
            bin_metrics["SEM_Temperature"] = np.nan

        # Activity metrics
        if not bin_activity.empty and "Activity" in bin_activity.columns:
            bin_metrics["Mean_Activity"] = bin_activity["Activity"].mean()
            bin_metrics["SEM_Activity"] = bin_activity["Activity"].std() / np.sqrt(
                bin_activity["Activity"].count()
            )
        else:
            bin_metrics["Mean_Activity"] = np.nan
            bin_metrics["SEM_Activity"] = np.nan

        binned_metrics.append(bin_metrics)

    binned_metrics_df = pd.DataFrame(binned_metrics)

    # Organize columns
    respiratory_cols = [
        col
        for col in binned_metrics_df.columns
        if col.startswith(("T_", "Duty_", "Resp", "Peak_", "Freq", "Pressure"))
    ]

    binned_metrics_df[""] = np.nan

    desired_columns = (
        ["Bin_Start", "Bin_End"]
        + respiratory_cols
        + [
            "",
            "Mean_Temperature",
            "SEM_Temperature",
            "",
            "Mean_Activity",
            "SEM_Activity",
        ]
    )

    for col in desired_columns:
        if col not in binned_metrics_df.columns:
            binned_metrics_df[col] = np.nan

    return binned_metrics_df[desired_columns]


def calculate_valid_period_metrics(
    peak_times: pd.Series, trough_times: pd.Series, pressure_data: pd.DataFrame
) -> dict[str, float]:
    """
    Computes mean and std respiratory metrics for a valid period,
    using time-masked peak/trough series where `.index` is aligned to pressure_data.
    """

    return calculate_respiratory_metrics(
        peak_times, peak_times, trough_times, pressure_data, include_std=True
    )


def calculate_respiratory_metrics(
    period_peaks: pd.Series,
    peak_indices: pd.Series,
    trough_indices: pd.Series,
    pressure_data: pd.DataFrame,
    include_std: bool = False,
) -> dict[str, float]:
    """
    Calculates summary respiratory metrics (mean ± std optionally).
    """

    if len(peak_indices) < 2 or len(trough_indices) < 2:
        return {
            "T_I_mean": np.nan,
            "T_E_mean": np.nan,
            "T_TOT_mean": np.nan,
            "Duty_Cycle_mean": np.nan,
            "Respiratory_Drive_mean": np.nan,
            "Peak_to_Peak_mean": np.nan,
            "Frequency_Hz": np.nan,
            "Pressure Difference": np.nan,
            **(
                {
                    "T_I_std": np.nan,
                    "T_E_std": np.nan,
                    "T_TOT_std": np.nan,
                    "Duty_Cycle_std": np.nan,
                    "Respiratory_Drive_std": np.nan,
                    "Peak_to_Peak_std": np.nan,
                    "Pressure Difference_std": np.nan,
                }
                if include_std
                else {}
            ),
        }

    # Align peak/trough order
    if trough_indices.iloc[0] > peak_indices.iloc[0]:
        peak_indices = peak_indices.iloc[1:]
        period_peaks = period_peaks.iloc[1:]
    if peak_indices.iloc[-1] < trough_indices.iloc[-1]:
        trough_indices = trough_indices.iloc[:-1]

    min_len = min(len(peak_indices), len(trough_indices))
    peak_indices = peak_indices.iloc[:min_len]
    trough_indices = trough_indices.iloc[:min_len]
    period_peaks = period_peaks.iloc[:min_len]

    # Time metrics
    peak_vals = peak_indices.values.astype(float)
    trough_vals = trough_indices.values.astype(float)

    T_I = peak_vals - trough_vals
    T_E = trough_vals[1:] - peak_vals[:-1]
    T_TOT = T_I[:-1] + T_E
    duty_cycle = T_I[:-1] / T_TOT

    peak_pressures = pressure_data.loc[
        peak_indices.index, "SmoothedPressure"
    ].values.astype(float)
    trough_pressures = pressure_data.loc[
        trough_indices.index, "SmoothedPressure"
    ].values.astype(float)
    PI = np.abs(peak_pressures[: len(T_I)] - trough_pressures[: len(T_I)])
    respiratory_drive = PI / T_I

    peak_to_peak_times = np.diff(peak_vals)
    frequencies = (
        1 / peak_to_peak_times if len(peak_to_peak_times) > 0 else np.array([])
    )

    metrics = {}

    # Mean + std pairs
    metrics["T_I_mean"] = T_I.mean()
    if include_std:
        metrics["T_I_std"] = T_I.std()

    metrics["T_E_mean"] = T_E.mean() if len(T_E) else np.nan
    if include_std:
        metrics["T_E_std"] = T_E.std() if len(T_E) else np.nan

    metrics["T_TOT_mean"] = T_TOT.mean() if len(T_TOT) else np.nan
    if include_std:
        metrics["T_TOT_std"] = T_TOT.std() if len(T_TOT) else np.nan

    metrics["Duty_Cycle_mean"] = duty_cycle.mean() if len(duty_cycle) else np.nan
    if include_std:
        metrics["Duty_Cycle_std"] = duty_cycle.std() if len(duty_cycle) else np.nan

    metrics["Respiratory_Drive_mean"] = (
        respiratory_drive.mean() if len(respiratory_drive) else np.nan
    )
    if include_std:
        metrics["Respiratory_Drive_std"] = (
            respiratory_drive.std() if len(respiratory_drive) else np.nan
        )

    metrics["Peak_to_Peak_mean"] = (
        peak_to_peak_times.mean() if len(peak_to_peak_times) else np.nan
    )
    if include_std:
        metrics["Peak_to_Peak_std"] = (
            peak_to_peak_times.std() if len(peak_to_peak_times) else np.nan
        )

    metrics["Frequency_Hz"] = frequencies.mean() if len(frequencies) else np.nan
    if include_std:
        metrics["Frequency_Hz_std"] = frequencies.std() if len(frequencies) else np.nan

    metrics["Pressure Difference"] = PI.mean() if len(PI) else np.nan
    if include_std:
        metrics["Pressure Difference_std"] = PI.std() if len(PI) else np.nan

    return metrics


def calculate_respiratory_metrics_raw(
    period_peaks: pd.Series,
    peak_indices: pd.Series,
    trough_indices: pd.Series,
    pressure_data: pd.DataFrame,
) -> dict[str, np.ndarray | float]:
    """
    Returns raw respiratory cycle metrics per breath (for post-hoc aggregation).
    """

    if len(peak_indices) < 2 or len(trough_indices) < 2:
        return {
            "T_I": np.array([]),
            "T_E": np.array([]),
            "T_TOT": np.array([]),
            "Duty_Cycle": np.array([]),
            "Drive": np.array([]),
            "Pressure_Diff": np.array([]),
            "Peak_to_Peak": np.array([]),
            "Freq": np.nan,
        }

    if trough_indices.iloc[0] > peak_indices.iloc[0]:
        peak_indices = peak_indices.iloc[1:]
        period_peaks = period_peaks.iloc[1:]
    if peak_indices.iloc[-1] < trough_indices.iloc[-1]:
        trough_indices = trough_indices.iloc[:-1]

    min_len = min(len(peak_indices), len(trough_indices))
    peak_indices = peak_indices.iloc[:min_len]
    trough_indices = trough_indices.iloc[:min_len]
    period_peaks = period_peaks.iloc[:min_len]

    peak_vals = peak_indices.values.astype(float)
    trough_vals = trough_indices.values.astype(float)

    T_I = peak_vals - trough_vals
    T_E = trough_vals[1:] - peak_vals[:-1]
    T_TOT = T_I[:-1] + T_E
    duty_cycle = T_I[:-1] / T_TOT

    peak_pressures = pressure_data.loc[
        peak_indices.index, "SmoothedPressure"
    ].values.astype(float)
    trough_pressures = pressure_data.loc[
        trough_indices.index, "SmoothedPressure"
    ].values.astype(float)
    PI = np.abs(peak_pressures[: len(T_I)] - trough_pressures[: len(T_I)])
    respiratory_drive = PI / T_I

    peak_to_peak_times = np.diff(peak_vals)
    frequency_hz = 1 / peak_to_peak_times.mean() if len(peak_to_peak_times) else np.nan

    return {
        "T_I": T_I,
        "T_E": T_E,
        "T_TOT": T_TOT,
        "Duty_Cycle": duty_cycle,
        "Drive": respiratory_drive,
        "Pressure_Diff": PI,
        "Peak_to_Peak": peak_to_peak_times,
        "Freq": frequency_hz,
    }


def summarize_respiratory_cycles(
    all_peaks: list[pd.Series],
    all_troughs: list[pd.Series],
    pressure_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute summary statistics across all respiratory cycles.

    Works at any scope (period, window, or full dataset) using raw per-cycle
    data. Produces overall mean and standard deviation without subgroup
    averaging.
    """

    metrics: dict[str, list[float]] = {
        "T_I": [],
        "T_E": [],
        "T_TOT": [],
        "Duty_Cycle": [],
        "Drive": [],
        "Pressure_Diff": [],
        "Peak_to_Peak": [],
        "Freq": [],
    }

    for peaks, troughs in zip(all_peaks, all_troughs, strict=False):
        if len(peaks) < 2 or len(troughs) < 2:
            continue

        raw = calculate_respiratory_metrics_raw(peaks, peaks, troughs, pressure_data)

        for key in metrics:
            val = raw.get(key)
            for key in metrics:
                val = raw.get(key)

                if key == "Freq":
                    # Freq is expected to be a scalar
                    if isinstance(val, (float | np.floating)) and not np.isnan(val):
                        metrics[key].append(float(val))
                else:
                    # Other metrics are expected to be iterable (Series, ndarray, list)
                    if val is not None:
                        if isinstance(val, (pd.Series | np.ndarray | list | tuple)):
                            if len(val) > 0:
                                metrics[key].extend(map(float, val))

    def agg(key: str) -> dict[str, float]:
        data = metrics[key]
        return {
            f"{key}_mean": float(np.mean(data)) if data else np.nan,
            f"{key}_std": float(np.std(data)) if data else np.nan,
        }

    result = {}
    for key in [
        "T_I",
        "T_E",
        "T_TOT",
        "Duty_Cycle",
        "Drive",
        "Pressure_Diff",
        "Peak_to_Peak",
    ]:
        result.update(agg(key))

    result["Frequency_Hz"] = (
        float(np.mean(metrics["Freq"])) if metrics["Freq"] else np.nan
    )
    result["Frequency_Hz_std"] = (
        float(np.std(metrics["Freq"])) if metrics["Freq"] else np.nan
    )

    return result
