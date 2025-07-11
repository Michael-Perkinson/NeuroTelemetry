from typing import Dict
import numpy as np
import pandas as pd
from typing import List, Any


def calculate_binned_period_metrics(
    period_start_time: float,
    period_end_time: float,
    bin_size_sec: int,
    peak_times: pd.Series,
    trough_times: pd.Series,
    pressure_data: pd.DataFrame,
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Bins data and computes respiratory, temperature, and activity metrics.
    Assumes peak_times and trough_times are pd.Series with indices that map to pressure_data rows.
    """
    bin_edges = np.arange(
        period_start_time, period_end_time + bin_size_sec, bin_size_sec)

    if (period_end_time - period_start_time) % bin_size_sec != 0:
        bin_edges = bin_edges[:-1]

    binned_metrics: List[Dict[str, Any]] = []

    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Masks
        bin_peak_mask = (peak_times >= bin_start) & (peak_times < bin_end)
        bin_trough_mask = (trough_times >= bin_start) & (
            trough_times < bin_end)

        bin_peaks = peak_times[bin_peak_mask]
        bin_troughs = trough_times[bin_trough_mask]

        bin_temp = temp_data[(temp_data.index >= bin_start)
                             & (temp_data.index < bin_end)]
        bin_activity = activity_data[(activity_data.index >= bin_start) & (
            activity_data.index < bin_end)]

        if len(bin_peaks) < 2 or len(bin_troughs) < 2:
            print(
                f"Skipping bin {i + 1} due to insufficient peaks or troughs.")
            continue

        # Calculate respiratory metrics
        bin_metrics = calculate_respiratory_metrics(
            bin_peaks, bin_peaks, bin_troughs, pressure_data
        )
        bin_metrics['Bin_Start'] = bin_start
        bin_metrics['Bin_End'] = bin_end

        # Temperature metrics
        if not bin_temp.empty and 'Temp' in bin_temp.columns:
            bin_metrics['Mean_Temperature'] = bin_temp['Temp'].mean()
            bin_metrics['SEM_Temperature'] = bin_temp['Temp'].std(
            ) / np.sqrt(bin_temp['Temp'].count())
        else:
            bin_metrics['Mean_Temperature'] = np.nan
            bin_metrics['SEM_Temperature'] = np.nan

        # Activity metrics
        if not bin_activity.empty and 'Activity' in bin_activity.columns:
            bin_metrics['Mean_Activity'] = bin_activity['Activity'].mean()
            bin_metrics['SEM_Activity'] = bin_activity['Activity'].std(
            ) / np.sqrt(bin_activity['Activity'].count())
        else:
            bin_metrics['Mean_Activity'] = np.nan
            bin_metrics['SEM_Activity'] = np.nan

        binned_metrics.append(bin_metrics)

    binned_metrics_df = pd.DataFrame(binned_metrics)

    # Organize columns
    respiratory_cols = [col for col in binned_metrics_df.columns if col.startswith(
        ("T_", "Duty_", "Resp", "Peak_", 'Freq', "Pressure"))]

    binned_metrics_df[''] = np.nan

    desired_columns = (
        ['Bin_Start', 'Bin_End'] +
        respiratory_cols +
        ['', 'Mean_Temperature', 'SEM_Temperature',
         '', 'Mean_Activity', 'SEM_Activity']
    )

    for col in desired_columns:
        if col not in binned_metrics_df.columns:
            binned_metrics_df[col] = np.nan

    return binned_metrics_df[desired_columns]


def calculate_valid_period_metrics(
    peak_times: pd.Series,
    trough_times: pd.Series,
    pressure_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes mean and std respiratory metrics for a valid period,
    using time-masked peak/trough series where `.index` is aligned to pressure_data.
    """

    print("\n[VALID PERIOD METRICS] Input check:")
    print(f"  ➤ Peaks: {len(peak_times)}, Troughs: {len(trough_times)}")
    print(f"  ➤ peak_times index: {peak_times.index[:3].tolist()}")
    print(f"  ➤ peak_times values: {peak_times.values[:3].tolist()}")
    print(f"  ➤ trough_times index: {trough_times.index[:3].tolist()}")
    print(f"  ➤ trough_times values: {trough_times.values[:3].tolist()}")

    # Use the index as the actual sample position
    corrected_peak_indices = pd.Series(
        peak_times.index.to_numpy(), index=peak_times.index)
    corrected_trough_indices = pd.Series(
        trough_times.index.to_numpy(), index=trough_times.index)

    return calculate_respiratory_metrics(
        peak_times,
        corrected_peak_indices,
        corrected_trough_indices,
        pressure_data,
        include_std=True
    )



def calculate_respiratory_metrics(
    period_peaks: pd.Series,
    peak_indices: pd.Series,
    trough_indices: pd.Series,
    pressure_data: pd.DataFrame,
    include_std: bool = False
) -> Dict[str, float]:
    """
    Calculates respiratory metrics (with optional stds) based on peak/trough timing and pressure.
    All inputs must be pd.Series with pressure_data-aligned indices.
    """

    # Step 1: Align peak and trough starts
    if trough_indices.iloc[0] > peak_indices.iloc[0]:
        peak_indices = peak_indices.iloc[1:]
        period_peaks = period_peaks.iloc[1:]

    if peak_indices.iloc[-1] < trough_indices.iloc[-1]:
        trough_indices = trough_indices.iloc[:-1]

    # Step 2: Ensure matching lengths
    min_len = min(len(peak_indices), len(trough_indices))
    peak_indices = peak_indices.iloc[:min_len]
    trough_indices = trough_indices.iloc[:min_len]
    period_peaks = period_peaks.iloc[:min_len]  # align for consistency

    # Step 3: Time intervals
    peak_vals = peak_indices.values.astype(float)
    trough_vals = trough_indices.values.astype(float)

    T_I = peak_vals - trough_vals  # inspiration time
    T_E = trough_vals[1:] - peak_vals[:-1]  # expiration time
    T_TOT = T_I[:-1] + T_E
    duty_cycle = T_I[:-1] / T_TOT

    # Step 4: Pressure difference & respiratory drive
    peak_pressures = np.asarray(
        pressure_data.loc[peak_indices.index, 'SmoothedPressure'], dtype=float)
    trough_pressures = np.asarray(
        pressure_data.loc[trough_indices.index, 'SmoothedPressure'], dtype=float)

    peak_pressures = peak_pressures[:len(T_I)]
    trough_pressures = trough_pressures[:len(T_I)]
    PI = np.abs(peak_pressures - trough_pressures)
    respiratory_drive = PI / T_I

    # Step 5: Peak-to-peak intervals
    peak_to_peak_times = np.diff(peak_vals)
    frequency_hz = 1 / peak_to_peak_times.mean() if len(peak_to_peak_times) > 0 else np.nan

    metrics: Dict[str, float] = {
        'T_I_mean': T_I.mean(),
        'T_E_mean': T_E.mean() if len(T_E) > 0 else np.nan,
        'T_TOT_mean': T_TOT.mean() if len(T_TOT) > 0 else np.nan,
        'Duty_Cycle_mean': duty_cycle.mean() if len(duty_cycle) > 0 else np.nan,
        'Respiratory_Drive_mean': respiratory_drive.mean() if len(respiratory_drive) > 0 else np.nan,
        'Peak_to_Peak_mean': peak_to_peak_times.mean() if len(peak_to_peak_times) > 0 else np.nan,
        'Frequency_Hz': frequency_hz,
        'Pressure Difference': PI.mean() if len(PI) > 0 else np.nan
    }

    if include_std:
        metrics.update({
            'T_I_std': T_I.std(),
            'T_E_std': T_E.std() if len(T_E) > 0 else np.nan,
            'T_TOT_std': T_TOT.std() if len(T_TOT) > 0 else np.nan,
            'Duty_Cycle_std': duty_cycle.std() if len(duty_cycle) > 0 else np.nan,
            'Respiratory_Drive_std': respiratory_drive.std() if len(respiratory_drive) > 0 else np.nan,
            'Peak_to_Peak_std': peak_to_peak_times.std() if len(peak_to_peak_times) > 0 else np.nan,
            'Pressure Difference_std': PI.std() if len(PI) > 0 else np.nan
        })

    return metrics


def calculate_combined_period_metrics(
    list_of_metric_dicts: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregates metrics across all valid periods (mean of means, etc).
    """
    if not list_of_metric_dicts:
        return {}

    df = pd.DataFrame(list_of_metric_dicts)
    combined = df.mean(numeric_only=True).to_dict()
    return combined
