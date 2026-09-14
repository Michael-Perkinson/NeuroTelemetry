from typing import Any

import numpy as np
import pandas as pd

from src.core.logger import log_warning


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
            (temp_data["TimeSinceReference"] >= bin_start)
            & (temp_data["TimeSinceReference"] < bin_end)
        ]

        bin_activity = activity_data[
            (activity_data["TimeSinceReference"] >= bin_start)
            & (activity_data["TimeSinceReference"] < bin_end)
        ]

        if len(bin_peaks) < 2 or len(bin_troughs) < 2:
            log_warning(f"Skipping bin {i + 1} due to insufficient peaks or troughs.")
            continue

        # Calculate respiratory metrics
        bin_metrics = calculate_period_respiratory_metrics(
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

    return calculate_period_respiratory_metrics(
        peak_times, peak_times, trough_times, pressure_data, include_std=True
    )


def calculate_period_respiratory_metrics(
    period_peaks: pd.Series,
    peak_indices: pd.Series,
    trough_indices: pd.Series,
    pressure_data: pd.DataFrame,
    include_std: bool = False,
) -> dict[str, float]:
    """
    Calculates summary respiratory metrics (mean ± std optionally).
    """
    raw_metrics = calculate_period_respiratory_metrics_raw(
        period_peaks, peak_indices, trough_indices, pressure_data
    )
    output_names = {
        "T_I": "T_I",
        "T_E": "T_E",
        "T_TOT": "T_TOT",
        "Duty_Cycle": "Duty_Cycle",
        "Drive": "Respiratory_Drive",
        "Peak_to_Peak": "Peak_to_Peak",
        "Freq": "Frequency_Hz",
        "Pressure_Diff": "Pressure Difference",
    }

    metrics: dict[str, float] = {}
    for raw_name, output_name in output_names.items():
        values = np.asarray(raw_metrics[raw_name], dtype=float)
        mean_name = (
            output_name
            if raw_name in {"Freq", "Pressure_Diff"}
            else f"{output_name}_mean"
        )
        metrics[mean_name] = float(values.mean()) if values.size else np.nan
        if include_std:
            metrics[f"{output_name}_std"] = (
                float(values.std()) if values.size else np.nan
            )

    return metrics


def calculate_period_respiratory_metrics_raw(
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
            "Freq": np.array([]),
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
    frequencies = (
        1 / peak_to_peak_times if len(peak_to_peak_times) > 0 else np.array([])
    )

    return {
        "T_I": T_I,
        "T_E": T_E,
        "T_TOT": T_TOT,
        "Duty_Cycle": duty_cycle,
        "Drive": respiratory_drive,
        "Pressure_Diff": PI,
        "Peak_to_Peak": peak_to_peak_times,
        "Freq": frequencies,
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

        raw = calculate_period_respiratory_metrics_raw(
            peaks, peaks, troughs, pressure_data
        )

        for key in metrics:
            val = raw.get(key)

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
        "Freq",
    ]:
        result.update(agg(key))

    result["Frequency_Hz"] = result.pop("Freq_mean", np.nan)
    result["Frequency_Hz_std"] = result.pop("Freq_std", np.nan)

    return result


def compute_atm_pressure_session_summary(
    atm_pressure_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return overall atmospheric-pressure stats for the full recording."""
    if atm_pressure_data.empty or "AtmPressure" not in atm_pressure_data.columns:
        return pd.DataFrame(columns=["Mean", "SD", "Min", "Max"])
    vals = atm_pressure_data["AtmPressure"].dropna()
    row: dict[str, Any] = {
        "Mean": float(vals.mean()),
        "SD": float(vals.std()),
        "Min": float(vals.min()),
        "Max": float(vals.max()),
    }
    return pd.DataFrame([row])


def compute_atm_pressure_time_bins(
    pressure_data: pd.DataFrame,
    atm_pressure_data: pd.DataFrame,
    bin_size_sec: int,
) -> pd.DataFrame:
    """
    Bin atmospheric pressure from start to end of recording, producing
    mean/SD/min/max/N per bin. Uses pressure_data only for the time axis.
    """
    min_time = pressure_data["TimeSinceReference"].min()
    max_time = pressure_data["TimeSinceReference"].max()

    bin_edges = np.arange(min_time, max_time + bin_size_sec, bin_size_sec)
    rows: list[dict[str, Any]] = []

    for i in range(len(bin_edges) - 1):
        bin_start = float(bin_edges[i])
        bin_end = float(bin_edges[i + 1])

        if atm_pressure_data.empty or "AtmPressure" not in atm_pressure_data.columns:
            rows.append(
                {
                    "Bin_Start_s": bin_start,
                    "Bin_End_s": bin_end,
                    "Mean": np.nan,
                    "SD": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                    "N": 0,
                }
            )
            continue

        upper_bound = (
            atm_pressure_data["TimeSinceReference"] <= bin_end
            if i == len(bin_edges) - 2
            else atm_pressure_data["TimeSinceReference"] < bin_end
        )
        mask = (atm_pressure_data["TimeSinceReference"] >= bin_start) & upper_bound
        vals = atm_pressure_data.loc[mask, "AtmPressure"].dropna()
        if vals.empty:
            rows.append(
                {
                    "Bin_Start_s": bin_start,
                    "Bin_End_s": bin_end,
                    "Mean": np.nan,
                    "SD": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                    "N": 0,
                }
            )
        else:
            rows.append(
                {
                    "Bin_Start_s": bin_start,
                    "Bin_End_s": bin_end,
                    "Mean": float(vals.mean()),
                    "SD": float(vals.std()),
                    "Min": float(vals.min()),
                    "Max": float(vals.max()),
                    "N": int(vals.count()),
                }
            )

    return pd.DataFrame(rows)
