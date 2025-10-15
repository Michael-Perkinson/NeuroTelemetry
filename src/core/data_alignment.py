import re
from copy import deepcopy
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import detrend, savgol_filter

from src.core.adaptive_algorithms import butter_lowpass_filter, compute_first_derivative
from src.core.data_file_parser import detect_skip_rows
from src.core.logger import log_info


def _to_datetime_with_ms(
    series: pd.Series,
    dayfirst: bool,
    sample_rate_hz: float,
) -> pd.Series:
    """
    Parse timestamps with optional milliseconds (before or after AM/PM).
    Falls back to reconstructing ms using sample rate if missing.
    """
    s = series.astype(str).str.strip()

    # Normalize weird AM/PM notations like "a.m." / "p.m."
    s = (
        s.str.replace("a.m.", "AM", case=False, regex=False)
        .str.replace("p.m.", "PM", case=False, regex=False)
        .str.replace("..", ".", regex=False)  # collapse double dots
    )

    first = s.iloc[0]

    # Case 1: ms before AM/PM
    if re.search(r"\.\d{1,6}\s*(AM|PM)\b", first, re.IGNORECASE):
        fmt = "%d/%m/%Y %I:%M:%S.%f %p" if dayfirst else "%m/%d/%Y %I:%M:%S.%f %p"
        return pd.to_datetime(s, format=fmt, errors="raise")

    # Case 2: ms after AM/PM
    if re.search(r"(AM|PM)\.\d{1,6}\b", first, re.IGNORECASE):
        fmt = "%d/%m/%Y %I:%M:%S %p.%f" if dayfirst else "%m/%d/%Y %I:%M:%S %p.%f"
        return pd.to_datetime(s, format=fmt, errors="raise")

    # Case 3: reconstruct
    fmt0 = "%d/%m/%Y %I:%M:%S %p" if dayfirst else "%m/%d/%Y %I:%M:%S %p"
    first_dt = pd.to_datetime(first, format=fmt0, errors="raise")
    step_us = int(round(1_000_000 / sample_rate_hz))
    return pd.date_range(start=first_dt, periods=len(s), freq=f"{step_us}us").to_series(
        index=series.index
    )


def _decide_dayfirst_from_probe(
    first_timestamp_str: str, probe_reference: datetime
) -> bool:
    """
    Decide if a date string is day-first (dd/mm/yyyy) or month-first (mm/dd/yyyy),
    using the first timestamp and a known probe reference date.
    """
    # Clean input string (handles uppercase + stray spaces/milliseconds)
    ts_str = str(first_timestamp_str).strip().upper()

    # Extract the two leading numbers (potentially day/month in some order)
    match = re.match(r"\s*(\d{1,2})/(\d{1,2})/", ts_str)
    if not match:
        return False  # fallback: assume month/day

    first_number = int(match.group(1))
    second_number = int(match.group(2))
    ref_month, ref_day = probe_reference.month, probe_reference.day

    # Case 1: Direct match to reference date
    if (first_number, second_number) == (ref_month, ref_day):
        return False  # month/day
    if (first_number, second_number) == (ref_day, ref_month):
        return True  # day/month

    # Case 2: Disambiguate with >12 rule
    if first_number > 12:
        return True  # must be day/month
    if second_number > 12:
        return False  # must be month/day

    # Case 3: Still ambiguous → default to month/day
    return False


def split_data(data: pd.DataFrame, skip_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into meta and numerical data
    """
    data = data[0].str.split(",", expand=True)
    meta_data = data.iloc[:skip_rows]
    all_numerical_data = data.iloc[skip_rows:].copy().reset_index(drop=True)

    return meta_data, all_numerical_data


def prepare_numerical_data(
    meta_data: pd.DataFrame, all_numerical_data: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Reorder numerical data columns to [DateTime, Pressure, Temp, Activity]
    based on metadata. If duplicate signals exist, only the first occurrence
    is kept. Returns the cleaned numerical data and sample rates.
    """
    # Normalize signal names
    name_map = {
        "temp": "Temp",
        "temperature": "Temp",
        "pressure": "Pressure",
        "activity": "Activity",
        "act": "Activity",
    }

    # Filter metadata rows starting with "# Col"
    mask = meta_data[0].str.startswith("# Col")
    cols = meta_data.loc[mask, [0, 1, 4]]

    # Extract column numbers (1-based → 0-based)
    col_nums = cols[0].str.extract(r"# Col\s+(\d+):").astype(int)[0] - 1

    # Normalize raw signal names
    raw_names = cols[1].astype(str).str.strip().str.lower()

    # Map to canonical names
    signal_names = raw_names.map(name_map).dropna()

    # Build column_order, keeping only first occurrence of each canonical signal
    column_order = {}
    for sig, col in zip(signal_names, col_nums.loc[signal_names.index], strict=False):
        if sig not in column_order:
            column_order[sig] = col

    # Reorder columns: DateTime first, then Pressure, Temp, Activity
    available_signals = [
        sig for sig in ["Pressure", "Temp", "Activity"] if sig in column_order
    ]
    new_order = [0] + [column_order[sig] for sig in available_signals]
    all_numerical_data = all_numerical_data.iloc[:, new_order]

    all_numerical_data.columns = ["DateTime"] + available_signals

    # Get sample rates for the first occurrence of each signal
    sample_rates: dict[str, float] = {}
    for sig in available_signals:
        idx = signal_names[signal_names == sig].index[0]
        rate_raw = cols.loc[idx, 4]

        # Force string type so Pylance stops inferring mixed dtypes
        rate_str: str = str(rate_raw)

        # Optional sanity check
        if ":" not in rate_str:
            raise ValueError(
                f"Unexpected rate format in metadata: {rate_str!r}")

        parts = rate_str.split(":")
        sample_rates[sig] = float(parts[1].strip())

    # Drop header row
    all_numerical_data = all_numerical_data.iloc[1:].reset_index(drop=True)

    return all_numerical_data, sample_rates


def extract_and_process_data(
    data: pd.DataFrame,
    behaviour_data: dict[str, list[tuple[int, float, float, float]]] | None,
    probe_date_time: str,
    alignment_date_time: str,
) -> dict[str, Any]:
    probe_ref, align_ref = parse_reference_timestamps(
        probe_date_time, alignment_date_time
    )

    skip_rows = detect_skip_rows(data)
    meta_data, all_numerical_data = split_data(data, skip_rows)

    all_numerical_data, sample_rates = prepare_numerical_data(
        meta_data, all_numerical_data
    )

    numerical_data = parse_numerical_data(
        all_numerical_data, probe_ref, sample_rates)

    numerical_data, new_ref, removed_nan_offset = align_and_clean_datetime(
        numerical_data, probe_ref
    )

    # bookkeeping only
    alignment_offset, total_offset = compute_time_offsets(
        align_ref, probe_ref, removed_nan_offset
    )
    log_info(
        f"Alignment offset: {alignment_offset}s | Total offset: {total_offset}s")

    # master timebase
    numerical_data["TimeSinceReference"] = (
        pd.to_datetime(numerical_data["DateTime"]) - pd.Timestamp(new_ref)
    ).dt.total_seconds()

    # build once, after finalised timeline
    processed_data: dict[str, Any] = build_output_frames(
        numerical_data, sample_rates)

    if behaviour_data is not None:
        processed_data["Behaviours"] = adjust_behaviours(
            behaviour_data, total_offset)

    processed_data["ReferenceTimestamp"] = new_ref

    return processed_data


def compute_time_offsets(video_ref, probe_ref, removed_nan):
    video_diff = (video_ref - probe_ref).total_seconds()
    total = video_diff + removed_nan
    return video_diff, total


def parse_reference_timestamps(
    probe_date_time: str, alignment_date_time: str
) -> tuple[datetime, datetime]:
    """Parse probe and alignment timestamps into datetime objects."""
    probe_ref = datetime.strptime(probe_date_time, "%d/%m/%Y %I:%M:%S %p")
    align_ref = datetime.strptime(alignment_date_time, "%d/%m/%Y %I:%M:%S %p")
    return probe_ref, align_ref


def build_output_frames(numerical_data: pd.DataFrame,
                        sample_rates: dict[str, float]) -> dict[str, pd.DataFrame]:
    processed = {}

    if "Pressure" not in numerical_data:
        raise ValueError("Pressure channel missing; cannot build timeline.")

    pressure_df = numerical_data[["TimeSinceReference", "Pressure"]].copy()
    prate = sample_rates["Pressure"]
    processed["Pressure"] = preprocess_pressure_data(pressure_df, prate)

    time_axis = processed["Pressure"]["TimeSinceReference"]

    for sig in ["Temp", "Activity"]:
        if sig in numerical_data:
            processed[sig] = safe_interpolate(numerical_data, time_axis, sig)

    return processed


def safe_interpolate(
    source_df: pd.DataFrame,
    time_axis: pd.Series,
    signal_name: str,
) -> pd.DataFrame:
    """
    Interpolates a given signal DataFrame onto a common time axis.
    Returns a DataFrame with TimeSinceReference and the interpolated signal.
    """
    if source_df.empty or signal_name not in source_df.columns:
        return pd.DataFrame(columns=["TimeSinceReference", signal_name])

    clean_df = source_df[["TimeSinceReference", signal_name]].dropna().copy()
    if clean_df.empty:
        return pd.DataFrame(columns=["TimeSinceReference", signal_name])

    interp_vals = np.interp(
        time_axis,
        clean_df["TimeSinceReference"],
        clean_df[signal_name],
    )
    return pd.DataFrame({"TimeSinceReference": time_axis, signal_name: interp_vals})


def parse_numerical_data(
    all_numerical_data: pd.DataFrame,
    probe_reference_timestamp: datetime,  # always m/d/Y %I:%M:%S %p
    sample_rates: dict[str, float],
) -> pd.DataFrame:
    """Parse telemetry dataframe into numeric values with timestamps."""

    # Copy and ensure canonical column names already set
    numerical_data = all_numerical_data.copy()

    # Convert numeric columns (skip DateTime)
    for col in numerical_data.columns:
        if col != "DateTime":
            numerical_data[col] = pd.to_numeric(
                numerical_data[col], errors="coerce")

    # Decide date parsing order
    first_str = str(numerical_data["DateTime"].iloc[0])
    dayfirst = _decide_dayfirst_from_probe(
        first_str, probe_reference_timestamp)

    # Pick master rate (prefer Pressure > Temp > Activity)
    if "Pressure" in sample_rates:
        master_rate = sample_rates["Pressure"]
    elif "Temp" in sample_rates:
        master_rate = sample_rates["Temp"]
    elif "Activity" in sample_rates:
        master_rate = sample_rates["Activity"]
    else:
        master_rate = 1.0  # fallback

    # Parse timestamps with ms reconstruction if needed
    numerical_data["DateTime"] = _to_datetime_with_ms(
        numerical_data["DateTime"], dayfirst=dayfirst, sample_rate_hz=master_rate
    )

    # Debug check
    invalid = numerical_data["DateTime"].isna().sum()
    if invalid:
        print(f"{invalid} DateTime values could not be parsed. First few bad rows:")
        print(numerical_data[numerical_data["DateTime"].isna()].head())

    return numerical_data


def align_and_clean_datetime(
    numerical_data: pd.DataFrame, probe_reference_timestamp: datetime
) -> tuple[pd.DataFrame, datetime, float]:
    """
    Clean numerical data by dropping duplicates/NaNs
    and shift the reference timestamp to the first valid sample.
    Returns the cleaned data, the new reference timestamp,
    and the offset in seconds relative to the original probe reference.
    """

    # Pick anchor column
    for candidate in ["Pressure", "Temp", "Activity"]:
        if candidate in numerical_data.columns:
            anchor_col = candidate
            break
    else:
        raise ValueError(
            "No usable signal found (Pressure/Temp/Activity missing).")

    numerical_data = numerical_data.drop_duplicates(
        subset="DateTime").reset_index(drop=True)

    first_valid_index = numerical_data[anchor_col].first_valid_index()
    if first_valid_index is None:
        raise ValueError(f"No valid {anchor_col} data found.")

    # Explicit conversion for Pylance
    first_valid_time = pd.to_datetime(
        str(numerical_data.at[first_valid_index, "DateTime"])).to_pydatetime()
    last_valid_time = pd.to_datetime(
        str(numerical_data["DateTime"].iloc[-1])).to_pydatetime()

    # Sanity check for reference timestamp
    if not (first_valid_time <= probe_reference_timestamp <= last_valid_time):
        raise ValueError(
            f"Reference timestamp {probe_reference_timestamp} not within "
            f"data range ({first_valid_time} → {last_valid_time})."
        )

    new_reference_timestamp: datetime = first_valid_time
    removed_nan_time_diff = (
        first_valid_time - probe_reference_timestamp).total_seconds()

    numerical_data = numerical_data.loc[first_valid_index:].reset_index(
        drop=True)
    
    return numerical_data, new_reference_timestamp, removed_nan_time_diff


def preprocess_pressure_data(
    pressure_data: pd.DataFrame, pressure_sample_rate: float
) -> pd.DataFrame:
    """Preprocess pressure signal by detrending, low-pass filtering, and smoothing."""

    # Always copy to avoid SettingWithCopyWarning
    pressure_data = pressure_data.copy()

    # Interpolate missing values
    pressure_data["Pressure"] = pressure_data["Pressure"].interpolate(
        method="linear")

    # Drop trailing NaNs if any
    last_valid_index = pressure_data["Pressure"].last_valid_index()
    log_info(f"Last valid index (non-NaN in Pressure): {last_valid_index}")
    pressure_data = pressure_data.loc[:last_valid_index]

    # Detrend
    detrended = detrend(
        pressure_data["Pressure"].to_numpy(dtype="float64"),
        type="linear",
    )

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
    pressure_data["SmoothedPressure"] = smoothed
    pressure_data["dvdt"] = dvdt

    pressure_data.reset_index(drop=True, inplace=True)

    return pressure_data


def adjust_behaviours(
    behaviour_data: dict[str, list[tuple[int, float, float, float]]],
    total_time_diff: float,
) -> dict[str, list[tuple[int, float, float, float]]]:
    behaviour_data = deepcopy(behaviour_data)
    for behaviour, instances in behaviour_data.items():
        adjusted_instances: list[tuple[int, float, float, float]] = []
        for instance_number, start_time, end_time, duration in instances:
            adjusted_start_time = start_time + total_time_diff
            adjusted_end_time = end_time + total_time_diff
            if adjusted_start_time >= 0 and adjusted_end_time >= 0:
                adjusted_instances.append(
                    (instance_number, adjusted_start_time,
                     adjusted_end_time, duration)
                )
            else:
                log_info(
                    f"Skipping behavior instance {instance_number} due to "
                    f"negative start or end time."
                )

        behaviour_data[behaviour] = adjusted_instances
    return behaviour_data


def prepare_raw_data(
    photometry_df: pd.DataFrame,
    temp_data: pd.DataFrame | None,
    activity_data: pd.DataFrame | None,
    injection_sec: float,
) -> pd.DataFrame:
    """
    Continuous raw data aligned to injection time.
    Columns: TimeRel, dFoF_465, Temp, Activity
    """
    df_raw = pd.DataFrame()
    df_raw["TimeRel"] = photometry_df["TimeSinceReference"] - injection_sec
    df_raw["dFoF_465"] = photometry_df["dFoF_465"]

    if temp_data is not None and not temp_data.empty:
        df_raw["Temp"] = np.interp(
            df_raw["TimeRel"],
            temp_data["TimeSinceReference"] - injection_sec,
            temp_data["Temp"],
        )

    if activity_data is not None and not activity_data.empty:
        df_raw["Activity"] = np.interp(
            df_raw["TimeRel"],
            activity_data["TimeSinceReference"] - injection_sec,
            activity_data["Activity"],
        )

    return df_raw
