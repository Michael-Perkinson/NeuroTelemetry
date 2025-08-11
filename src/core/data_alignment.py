from typing import Tuple
from typing import Optional, Tuple, Dict, List
from copy import deepcopy
import pandas as pd
import numpy as np
import re
from datetime import datetime
from scipy.signal import detrend, savgol_filter

from src.core.adaptive_algorithms import butter_lowpass_filter, compute_first_derivative
from src.core.logger import log_info, log_exception
from src.core.data_file_parser import detect_skip_rows


def _to_datetime_with_ms(series: pd.Series, dayfirst: bool, sample_rate_hz: float) -> pd.Series:
    """
    Vectorized timestamp parsing:
    - If first row already has .%f → parse all with %f.
    - Else reconstruct ms from sample rate starting at first timestamp.
    """
    s = series.astype(str).str.strip().str.upper()
    has_ms = bool(re.search(r'\.\d{1,6}\s*(AM|PM)\b', s.iloc[0]))

    if has_ms:
        fmt = "%d/%m/%Y %I:%M:%S.%f %p" if dayfirst else "%m/%d/%Y %I:%M:%S.%f %p"
        return pd.to_datetime(s, format=fmt, errors="coerce")

    # No milliseconds → build from first timestamp + sample rate
    fmt0 = "%d/%m/%Y %I:%M:%S %p" if dayfirst else "%m/%d/%Y %I:%M:%S %p"
    first_dt = pd.to_datetime(s.iloc[0], format=fmt0, errors="coerce")
    if pd.isna(first_dt):
        raise ValueError(f"Could not parse first datetime: {s.iloc[0]}")

    step_us = int(round(1_000_000 / sample_rate_hz))
    n = len(s)
    idx = pd.date_range(start=first_dt, periods=n, freq=f"{step_us}us")
    return pd.Series(idx, index=series.index)


def _decide_dayfirst_from_probe(first_dt_str: str, probe_mdy: datetime) -> bool:
    """
    Decide if dates are day-first or month-first from the first timestamp and a probe reference.
    Returns True for day/month/year, False for month/day/year.
    """
    # Normalize for matching; works with or without milliseconds
    s = str(first_dt_str).strip().upper()

    # Extract the two leading numbers (day/month in some order)
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/", s)
    if not m:
        return False  # fallback to month-first

    a, b = int(m.group(1)), int(m.group(2))
    pm, pd_ = probe_mdy.month, probe_mdy.day

    # Direct match to probe date
    if (a, b) == (pm, pd_):
        return False  # month/day
    if (a, b) == (pd_, pm):
        return True   # day/month

    # >12 rule
    if a > 12:
        return True
    if b > 12:
        return False

    # Final fallback
    return False


def split_data(data: pd.DataFrame, skip_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into meta and numerical data
    """
    data = data[0].str.split(',', expand=True)
    meta_data = data.iloc[:skip_rows]
    all_numerical_data = data.iloc[skip_rows:].copy().reset_index(drop=True)

    return meta_data, all_numerical_data


def extract_and_process_data(
    data: pd.DataFrame,
    behaviour_data: Optional[Dict[str, List[Tuple[int, float, float, float]]]],
    probe_date_time: str,
    video_date_time: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, datetime]:

    probe_reference_timestamp: datetime = datetime.strptime(
        probe_date_time, '%d/%m/%Y %I:%M:%S %p')
    video_reference_timestamp: datetime = datetime.strptime(
        video_date_time, '%d/%m/%Y %I:%M:%S %p')

    log_info(f'''Probe timestamp: {probe_reference_timestamp}''')
    log_info(f'''Video timestamp: {video_reference_timestamp}''')

    skip_rows = detect_skip_rows(data)

    meta_data, all_numerical_data = split_data(data, skip_rows)

    pressure_sample_rate, temp_sample_rate, activity_sample_rate = extract_sample_rates(
        meta_data)

    numerical_data = parse_numerical_data(
        all_numerical_data, probe_reference_timestamp, pressure_sample_rate)

    numerical_data, new_reference_timestamp, removed_nan_time_diff = align_and_clean_datetime(
        numerical_data, probe_reference_timestamp
    )

    video_time_diff: float = (
        video_reference_timestamp - probe_reference_timestamp).total_seconds()
    log_info(f'''Video start time difference: {video_time_diff} seconds''')

    total_time_diff: float = video_time_diff + removed_nan_time_diff
    log_info(
        f'''Total time difference (video + NaN): {total_time_diff} seconds''')

    numerical_data['TimeSinceReference'] = (
        numerical_data['DateTime'] - new_reference_timestamp
    ).dt.total_seconds()
    numerical_data['TimeSinceReference'] = pd.to_numeric(
        numerical_data['TimeSinceReference'], errors='coerce')

    temp_interval: int = int(pressure_sample_rate / temp_sample_rate)
    activity_interval: int = int(pressure_sample_rate / activity_sample_rate)

    downsampled_temp: pd.DataFrame = numerical_data.iloc[::temp_interval, [
        0, 2, 4]].reset_index(drop=True)
    downsampled_temp.columns = ['DateTime', 'Temp', 'TimeSinceReference']

    downsampled_activity: pd.DataFrame = numerical_data.iloc[::activity_interval, [
        0, 3, 4]].reset_index(drop=True)
    downsampled_activity.columns = [
        'DateTime', 'Activity', 'TimeSinceReference']

    pressure_data: pd.DataFrame = numerical_data[[
        'TimeSinceReference', 'Pressure']].copy()
    temp_data: pd.DataFrame = downsampled_temp[[
        'TimeSinceReference', 'Temp']].copy()
    activity_data: pd.DataFrame = downsampled_activity[[
        'TimeSinceReference', 'Activity']].copy()

    pressure_data = preprocess_pressure_data(
        pressure_data, pressure_sample_rate)

    if behaviour_data is not None:
        behaviour_data = adjust_behaviours(behaviour_data, total_time_diff)

    return pressure_data, temp_data, activity_data, numerical_data, new_reference_timestamp


def parse_numerical_data(
    all_numerical_data: pd.DataFrame,
    probe_reference_timestamp: datetime,  # always m/d/Y %I:%M:%S %p
    pressure_sample_rate: float
) -> pd.DataFrame:
    # Ensure 4 cols
    if all_numerical_data.shape[1] != 4:
        if all_numerical_data.shape[1] > 4:
            all_numerical_data = all_numerical_data.iloc[1:, :4]
        else:
            for i in range(4 - all_numerical_data.shape[1]):
                all_numerical_data[f'pad{i}'] = 0

    numerical_data = all_numerical_data.copy()

    numerical_data.columns = ['DateTime', 'Pressure', 'Temp', 'Activity']

    # Convert numeric columns
    for c in ['Pressure', 'Temp', 'Activity']:
        numerical_data[c] = pd.to_numeric(numerical_data[c], errors='coerce')

    # Decide order from probe vs first row
    first_str = str(numerical_data['DateTime'].iloc[0])
    dayfirst = _decide_dayfirst_from_probe(
        first_str, probe_reference_timestamp)

    # Parse or reconstruct ms
    numerical_data['DateTime'] = _to_datetime_with_ms(
        numerical_data['DateTime'], dayfirst=dayfirst, sample_rate_hz=pressure_sample_rate
    )

    invalid = numerical_data['DateTime'].isna().sum()
    if invalid:
        print(f"{invalid} DateTime values could not be parsed. First few bad rows:")
        print(numerical_data[numerical_data['DateTime'].isna()].head())

    return numerical_data


def align_and_clean_datetime(
    numerical_data: pd.DataFrame,
    probe_reference_timestamp: datetime
) -> Tuple[pd.DataFrame, datetime, float]:
    """Align and clean numerical data by removing NaNs and adjusting the timestamp reference."""

    removed_nan_time_diff: float = 0.0
    first_valid_index = numerical_data['Pressure'].first_valid_index()

    if first_valid_index is not None:
        # Safely get scalar pd.Timestamp
        datetime_value: pd.Timestamp = numerical_data.at[first_valid_index, 'DateTime']

        # Compute time difference
        time_diff = datetime_value - probe_reference_timestamp
        new_reference_timestamp: datetime = probe_reference_timestamp + time_diff

        log_info(
            f'Time difference between first valid pressure and reference timestamp: {time_diff}')

        # Trim data to start at first valid pressure
        numerical_data = numerical_data.loc[first_valid_index:].reset_index(
            drop=True)

        # Track offset caused by dropping NaNs
        removed_nan_time_diff = time_diff.total_seconds()
        log_info(
            f'Time difference due to NaN removal: {removed_nan_time_diff} seconds')
    else:
        raise ValueError(
            "No valid pressure data found. Cannot proceed with analysis.")

    return numerical_data, new_reference_timestamp, removed_nan_time_diff


def extract_sample_rates(
    meta_data: pd.DataFrame
) -> Tuple[float, float, float]:

    col2_row = meta_data[meta_data[0].str.contains('Col 2:')].iloc[0]
    pressure_sample_rate = float(col2_row[4].split(':')[1].strip())

    col3_row = meta_data[meta_data[0].str.contains('Col 3:')].iloc[0]
    temp_sample_rate = float(col3_row[4].split(':')[1].strip())

    col4_row = meta_data[meta_data[0].str.contains('Col 4:')].iloc[0]
    activity_sample_rate = float(col4_row[4].split(':')[1].strip())

    return pressure_sample_rate, temp_sample_rate, activity_sample_rate


def preprocess_pressure_data(pressure_data: pd.DataFrame, pressure_sample_rate: float) -> pd.DataFrame:
    """Preprocess pressure signal by detrending, low-pass filtering, and smoothing."""

    # Always copy to avoid SettingWithCopyWarning
    pressure_data = pressure_data.copy()

    # Interpolate missing values
    pressure_data['Pressure'] = pressure_data['Pressure'].interpolate(
        method='linear')

    # Drop trailing NaNs if any
    last_valid_index = pressure_data['Pressure'].last_valid_index()
    log_info(f"Last valid index (non-NaN in Pressure): {last_valid_index}")
    pressure_data = pressure_data.loc[:last_valid_index]

    # Detrend
    detrended = detrend(pressure_data['Pressure'].to_numpy(
        dtype='float64'), type='linear')

    # Butterworth low-pass filter
    filtered = butter_lowpass_filter(detrended, cutoff_hz=10, fs=500)

    # Ensure window is odd and fits signal length
    window = min(35, len(filtered) - 1)
    if window % 2 == 0:
        window -= 1

    # Savitzky-Golay smoothing
    smoothed = savgol_filter(filtered, window_length=window, polyorder=2)
    dvdt = compute_first_derivative(filtered, pressure_sample_rate)

    # Update DataFrame
    pressure_data['SmoothedPressure'] = smoothed
    pressure_data['dvdt'] = dvdt

    pressure_data.reset_index(drop=True, inplace=True)

    return pressure_data


def adjust_behaviours(
    behaviour_data: Dict[str, List[Tuple[int, float, float, float]]],
    total_time_diff: float
) -> Dict[str, List[Tuple[int, float, float, float]]]:

    behaviour_data = deepcopy(behaviour_data)
    for behaviour, instances in behaviour_data.items():
        adjusted_instances: List[Tuple[int, float, float, float]] = []
        for instance_number, start_time, end_time, duration in instances:
            adjusted_start_time = start_time + total_time_diff
            adjusted_end_time = end_time + total_time_diff
            if adjusted_start_time >= 0 and adjusted_end_time >= 0:
                adjusted_instances.append(
                    (instance_number, adjusted_start_time, adjusted_end_time, duration))
            else:
                log_info(
                    f'''Skipping behavior instance {instance_number} due to negative start or end time.''')
        behaviour_data[behaviour] = adjusted_instances
    return behaviour_data
