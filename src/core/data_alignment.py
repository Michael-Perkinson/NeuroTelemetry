from typing import Tuple
from typing import Optional, Tuple, Dict, List
from copy import deepcopy
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import detrend, savgol_filter

from src.core.adaptive_algorithms import butter_lowpass_filter, compute_first_derivative
from src.core.logger import log_info, log_exception
from src.core.data_file_parser import detect_skip_rows

def extract_and_process_data(
    data: pd.DataFrame,
    behaviour_data: Optional[Dict[str, List[Tuple[int, float, float, float]]]],
    probe_date_time: str,
    video_date_time: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, datetime]:

    probe_reference_timestamp: datetime = datetime.strptime(
        probe_date_time, '%m/%d/%Y %I:%M:%S %p')
    video_reference_timestamp: datetime = datetime.strptime(
        video_date_time, '%m/%d/%Y %I:%M:%S %p')

    log_info(f'''Probe timestamp: {probe_reference_timestamp}''')
    log_info(f'''Video timestamp: {video_reference_timestamp}''')

    skip_rows = detect_skip_rows(data)

    meta_data, numerical_data = split_and_parse_data(data, skip_rows)

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

    pressure_sample_rate, temp_sample_rate, activity_sample_rate = extract_sample_rates(
        meta_data)

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


def split_and_parse_data(
    data: pd.DataFrame,
    skip_rows: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = data[0].str.split(',', expand=True)

    meta_data = data.iloc[:skip_rows]
    numerical_data = data.iloc[skip_rows:]

    numerical_data.reset_index(drop=True, inplace=True)
    
    # Ensure numerical_data has exactly 4 columns
    if numerical_data.shape[1] != 4:
        if numerical_data.shape[1] > 4:
            numerical_data = numerical_data.iloc[1:, :4]
        else:
            missing_cols = 4 - numerical_data.shape[1]
            for i in range(missing_cols):
                numerical_data[f'pad{i}'] = 0

    numerical_data.columns = ['DateTime',
                              'Pressure', 'Temp', 'Activity']

    numerical_data['DateTime'] = numerical_data['DateTime'].str.strip()

    numerical_data['Pressure'] = pd.to_numeric(
        numerical_data['Pressure'], errors='coerce')
    numerical_data['Temp'] = pd.to_numeric(
        numerical_data['Temp'], errors='coerce')
    numerical_data['Activity'] = pd.to_numeric(
        numerical_data['Activity'], errors='coerce')

    try:
        numerical_data['DateTime'] = pd.to_datetime(
            numerical_data['DateTime'],
            format='%m/%d/%Y %I:%M:%S.%f %p',
            errors='raise'
        )
    except ValueError as e:
        print(f"Parsing failed: {e}")
        numerical_data['DateTime'] = pd.to_datetime(
            numerical_data['DateTime'], errors='coerce')

    # Step 7: Warn if any timestamps couldn't be parsed
    invalid_dates = numerical_data[numerical_data['DateTime'].isna()]
    if not invalid_dates.empty:
        print("Some DateTime values could not be parsed. Here are the first few invalid entries:")
        print(invalid_dates.head())

    return meta_data, numerical_data


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
