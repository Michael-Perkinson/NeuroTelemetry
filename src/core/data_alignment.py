import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.core.adaptive_algorithms import (
    butter_highpass_filter,
    butter_lowpass_filter,
    compute_first_derivative,
)
from src.core.data_file_parser import detect_skip_rows
from src.core.logger import log_info, log_warning

PRESSURE_COVERAGE_MAX_GAP_SECONDS = 1.0
PRESSURE_FILTER_EDGE_MARGIN_SECONDS = 1.0


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

    # Case 3: reconstruct sub-second positions while preserving visible jumps.
    fmt0 = "%d/%m/%Y %I:%M:%S %p" if dayfirst else "%m/%d/%Y %I:%M:%S %p"
    parsed_seconds = pd.to_datetime(s, format=fmt0, errors="raise")
    step_us = int(round(1_000_000 / sample_rate_hz))
    if parsed_seconds.nunique() == 1:
        return pd.date_range(
            start=parsed_seconds.iloc[0], periods=len(s), freq=f"{step_us}us"
        ).to_series(index=series.index)

    within_second = parsed_seconds.groupby(parsed_seconds).cumcount()
    return parsed_seconds + pd.to_timedelta(
        within_second.to_numpy() * step_us,
        unit="us",
    )


def _decide_dayfirst_from_reference(
    first_timestamp_str: str, reference_timestamp: datetime
) -> bool:
    """
    Decide if a date string is day-first (dd/mm/yyyy) or month-first (mm/dd/yyyy),
    choosing the valid interpretation closest to the supplied reference date.
    """
    ts_str = str(first_timestamp_str).strip().upper()

    # Extract the two leading numbers (potentially day/month in some order)
    match = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", ts_str)
    if not match:
        return False  # fallback: assume month/day

    first_number = int(match.group(1))
    second_number = int(match.group(2))
    year = int(match.group(3))

    # Case 1: Direct match to reference date
    if (first_number, second_number) == (
        reference_timestamp.month,
        reference_timestamp.day,
    ):
        return False  # month/day
    if (first_number, second_number) == (
        reference_timestamp.day,
        reference_timestamp.month,
    ):
        return True  # day/month

    # Case 2: Disambiguate with >12 rule
    if first_number > 12:
        return True  # must be day/month
    if second_number > 12:
        return False  # must be month/day

    candidates: list[tuple[bool, datetime]] = []
    try:
        candidates.append((False, datetime(year, first_number, second_number)))
    except ValueError:
        pass
    try:
        candidates.append((True, datetime(year, second_number, first_number)))
    except ValueError:
        pass

    if candidates:
        reference_date = datetime(
            reference_timestamp.year,
            reference_timestamp.month,
            reference_timestamp.day,
        )
        return min(
            candidates,
            key=lambda candidate: abs((candidate[1] - reference_date).total_seconds()),
        )[0]

    # No valid candidate is available: default to month/day.
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

    name_map = {
        "temp": "Temp",
        "temperature": "Temp",
        "pressure": "Pressure",
        "activity": "Activity",
        "act": "Activity",
        "apr": "AtmPressure",
    }

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
        sig
        for sig in ["Pressure", "Temp", "Activity", "AtmPressure"]
        if sig in column_order
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
            raise ValueError(f"Unexpected rate format in metadata: {rate_str!r}")

        parts = rate_str.split(":")
        sample_rates[sig] = float(parts[1].strip())

    all_numerical_data = all_numerical_data.iloc[1:].reset_index(drop=True)

    return all_numerical_data, sample_rates


def extract_and_process_data(
    data: pd.DataFrame,
    behaviour_data: dict[str, list[tuple[int, float, float, float]]] | None,
    alignment_date_time: str,
    *,
    timeline_origin: Literal["first_valid", "alignment"] = "first_valid",
) -> dict[str, Any]:
    """Parse telemetry and express it on the requested timeline."""
    if timeline_origin not in {"first_valid", "alignment"}:
        raise ValueError("timeline_origin must be 'first_valid' or 'alignment'.")

    align_ref = parse_reference_timestamp(alignment_date_time)

    skip_rows = detect_skip_rows(data)
    meta_data, all_numerical_data = split_data(data, skip_rows)

    all_numerical_data, sample_rates = prepare_numerical_data(
        meta_data, all_numerical_data
    )

    numerical_data = parse_numerical_data(all_numerical_data, align_ref, sample_rates)

    numerical_data, first_valid_ref = align_and_clean_datetime(numerical_data)
    timeline_ref = align_ref if timeline_origin == "alignment" else first_valid_ref
    alignment_offset = (align_ref - first_valid_ref).total_seconds()
    log_info(
        f"Alignment offset from first valid sample: {alignment_offset}s | "
        f"Timeline origin: {timeline_origin}"
    )

    # master timebase
    numerical_data["TimeSinceReference"] = (
        pd.to_datetime(numerical_data["DateTime"]) - pd.Timestamp(timeline_ref)
    ).dt.total_seconds()

    pressure_coverage = continuous_signal_coverage(
        numerical_data,
        "Pressure",
        max_gap_seconds=PRESSURE_COVERAGE_MAX_GAP_SECONDS,
    )
    analyzable_pressure_coverage = trim_coverage_intervals(
        pressure_coverage,
        margin_seconds=PRESSURE_FILTER_EDGE_MARGIN_SECONDS,
    )

    # build once, after finalised timeline
    processed_data: dict[str, Any] = build_output_frames(numerical_data, sample_rates)

    if behaviour_data is not None:
        behaviour_offset = (align_ref - timeline_ref).total_seconds()
        processed_data["Behaviours"] = adjust_behaviours(
            behaviour_data, behaviour_offset
        )

    processed_data["ReferenceTimestamp"] = timeline_ref
    processed_data["FirstValidTimestamp"] = first_valid_ref
    processed_data["PressureCoverageIntervals"] = pressure_coverage
    processed_data["PressureAnalyzableIntervals"] = analyzable_pressure_coverage
    processed_data["PressureCoverageMaxGapSeconds"] = PRESSURE_COVERAGE_MAX_GAP_SECONDS
    processed_data["PressureFilterEdgeMarginSeconds"] = (
        PRESSURE_FILTER_EDGE_MARGIN_SECONDS
    )

    return processed_data


def continuous_signal_coverage(
    numerical_data: pd.DataFrame,
    signal_name: str,
    *,
    max_gap_seconds: float,
) -> list[tuple[float, float]]:
    """Return valid-signal intervals, splitting outages longer than a threshold."""
    if signal_name not in numerical_data or "TimeSinceReference" not in numerical_data:
        return []

    signal = pd.to_numeric(numerical_data[signal_name], errors="coerce")
    times = pd.to_numeric(numerical_data["TimeSinceReference"], errors="coerce")
    valid_times = (
        times[signal.notna() & times.notna()].drop_duplicates().sort_values().tolist()
    )
    if not valid_times:
        return []

    intervals: list[tuple[float, float]] = []
    interval_start = float(valid_times[0])
    previous = interval_start
    for raw_time in valid_times[1:]:
        current = float(raw_time)
        if current - previous > max_gap_seconds:
            intervals.append((interval_start, previous))
            interval_start = current
        previous = current
    intervals.append((interval_start, previous))
    return intervals


def trim_coverage_intervals(
    intervals: list[tuple[float, float]],
    *,
    margin_seconds: float,
) -> list[tuple[float, float]]:
    """Exclude filter-edge regions from continuous signal coverage."""
    return [
        (start + margin_seconds, end - margin_seconds)
        for start, end in intervals
        if end - start >= 2 * margin_seconds
    ]


def parse_reference_timestamp(reference_date_time: str) -> datetime:
    """Parse a day-first external alignment timestamp."""
    return datetime.strptime(reference_date_time, "%d/%m/%Y %I:%M:%S %p")


def build_output_frames(
    numerical_data: pd.DataFrame, sample_rates: dict[str, float]
) -> dict[str, pd.DataFrame]:
    processed = {}

    if "Pressure" in numerical_data:
        pressure_df = numerical_data[["TimeSinceReference", "Pressure"]].copy()
        prate = sample_rates["Pressure"]
        processed["Pressure"] = preprocess_pressure_data(pressure_df, prate)
        time_axis = processed["Pressure"]["TimeSinceReference"]
        for signal in ["Temp", "Activity"]:
            if signal in numerical_data:
                processed[signal] = safe_interpolate(numerical_data, time_axis, signal)
    else:
        optional_signals = [
            signal
            for signal in ("Temp", "Activity")
            if signal in numerical_data and numerical_data[signal].notna().any()
        ]
        if not optional_signals:
            raise ValueError("No supported telemetry channel available for timeline.")
        for signal in optional_signals:
            signal_df = numerical_data[["TimeSinceReference", signal]].dropna().copy()
            processed[signal] = signal_df.reset_index(drop=True)

    # AtmPressure is 1 Hz — downsample to native rate using the recorded sample rate
    # Ponemah repeats the last value at 500 Hz rather than leaving NaNs, so dropna()
    # won't help — instead bin by sample period and keep one value per period
    if "AtmPressure" in numerical_data:
        atm_rate = sample_rates.get("AtmPressure", 1.0)
        period = 1.0 / atm_rate
        atm_df = numerical_data[["TimeSinceReference", "AtmPressure"]].copy()
        atm_df["_bin"] = (atm_df["TimeSinceReference"] / period).round()
        atm_df = atm_df.drop_duplicates(subset="_bin").drop(columns="_bin")
        processed["AtmPressure"] = atm_df.reset_index(drop=True)

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
        left=np.nan,
        right=np.nan,
    )
    return pd.DataFrame({"TimeSinceReference": time_axis, signal_name: interp_vals})


def parse_numerical_data(
    all_numerical_data: pd.DataFrame,
    reference_timestamp: datetime,
    sample_rates: dict[str, float],
) -> pd.DataFrame:
    """Parse telemetry dataframe into numeric values with timestamps."""

    numerical_data = all_numerical_data.copy()

    for col in numerical_data.columns:
        if col != "DateTime":
            numerical_data[col] = pd.to_numeric(numerical_data[col], errors="coerce")

    # Decide date parsing order
    first_str = str(numerical_data["DateTime"].iloc[0])
    dayfirst = _decide_dayfirst_from_reference(first_str, reference_timestamp)

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
        log_warning(
            f"{invalid} DateTime values could not be parsed. First few bad rows:\n"
            f"{numerical_data[numerical_data['DateTime'].isna()].head().to_string()}"
        )

    return numerical_data


def align_and_clean_datetime(
    numerical_data: pd.DataFrame,
) -> tuple[pd.DataFrame, datetime]:
    """
    Clean numerical data by dropping duplicates/NaNs
    and shift the reference timestamp to the first valid sample.
    Returns the cleaned data and its first valid sample timestamp. External
    alignment timestamps are deliberately not range-checked here.
    """

    # Pick anchor column
    for candidate in ["Pressure", "Temp", "Activity"]:
        if (
            candidate in numerical_data.columns
            and numerical_data[candidate].notna().any()
        ):
            anchor_col = candidate
            break
    else:
        raise ValueError("No usable signal found (Pressure/Temp/Activity missing).")

    numerical_data = numerical_data.drop_duplicates(subset="DateTime").reset_index(
        drop=True
    )

    first_valid_index = numerical_data[anchor_col].first_valid_index()
    if first_valid_index is None:
        raise ValueError(f"No valid {anchor_col} data found.")

    # Explicit conversion for Pylance
    first_valid_time = pd.to_datetime(
        numerical_data.at[first_valid_index, "DateTime"]
    ).to_pydatetime()
    numerical_data = numerical_data.loc[first_valid_index:].reset_index(drop=True)

    return numerical_data, first_valid_time


def preprocess_pressure_data(
    pressure_data: pd.DataFrame, pressure_sample_rate: float
) -> pd.DataFrame:
    """Filter continuous pressure segments without bridging long outages."""

    pressure_data = pressure_data.copy()
    pressure_data["Pressure"] = pd.to_numeric(
        pressure_data["Pressure"], errors="coerce"
    )

    # Drop trailing NaNs if any.
    last_valid_index = pressure_data["Pressure"].last_valid_index()
    log_info(f"Last valid index (non-NaN in Pressure): {last_valid_index}")
    pressure_data = pressure_data.loc[:last_valid_index]
    pressure_data["PressureHighpass"] = np.nan
    pressure_data["SmoothedPressure"] = np.nan
    pressure_data["dvdt"] = np.nan

    valid_rows = pressure_data["Pressure"].notna()
    valid_times = pressure_data.loc[valid_rows, "TimeSinceReference"].astype(float)
    split_points = np.flatnonzero(
        valid_times.diff().fillna(0).to_numpy() > PRESSURE_COVERAGE_MAX_GAP_SECONDS
    )
    valid_indices = valid_times.index.to_numpy()

    for group in np.split(valid_indices, split_points):
        if len(group) == 0:
            continue
        segment_index = pressure_data.loc[group[0] : group[-1]].index
        segment = pressure_data.loc[segment_index, "Pressure"].interpolate(
            method="linear"
        )
        if len(segment) < 36 or segment.isna().any():
            log_warning(
                "Skipping pressure preprocessing for a continuous segment "
                f"with only {len(segment)} rows."
            )
            continue

        raw_pressure = segment.to_numpy(dtype="float64")
        highpassed = butter_highpass_filter(
            raw_pressure, fs=pressure_sample_rate, cutoff_hz=0.02
        )
        filtered = butter_lowpass_filter(
            raw_pressure, cutoff_hz=10, fs=pressure_sample_rate
        )
        smoothed = savgol_filter(filtered, window_length=35, polyorder=2)
        dvdt = compute_first_derivative(filtered, pressure_sample_rate)

        pressure_data.loc[segment_index, "Pressure"] = raw_pressure
        segment_times = pressure_data.loc[segment_index, "TimeSinceReference"].astype(
            float
        )
        usable = (
            segment_times >= segment_times.iloc[0] + PRESSURE_FILTER_EDGE_MARGIN_SECONDS
        ) & (
            segment_times
            <= segment_times.iloc[-1] - PRESSURE_FILTER_EDGE_MARGIN_SECONDS
        )
        usable_index = segment_index[usable.to_numpy()]
        pressure_data.loc[usable_index, "PressureHighpass"] = highpassed[usable]
        pressure_data.loc[usable_index, "SmoothedPressure"] = smoothed[usable]
        pressure_data.loc[usable_index, "dvdt"] = dvdt[usable]

    pressure_data.reset_index(drop=True, inplace=True)

    return pressure_data


def adjust_behaviours(
    behaviour_data: dict[str, list[tuple[int, float, float, float]]],
    total_time_diff: float,
) -> dict[str, list[tuple[int, float, float, float]]]:
    """
    Shift start and end times of behaviour instances by a given offset.

    Negative times are retained so downstream coverage checks can distinguish
    partially covered and unavailable behavior windows.
    """
    behaviour_data = deepcopy(behaviour_data)
    for behaviour, instances in behaviour_data.items():
        adjusted_instances: list[tuple[int, float, float, float]] = []
        for instance_number, start_time, end_time, duration in instances:
            adjusted_start_time = start_time + total_time_diff
            adjusted_end_time = end_time + total_time_diff
            adjusted_instances.append(
                (instance_number, adjusted_start_time, adjusted_end_time, duration)
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
            left=np.nan,
            right=np.nan,
        )

    if activity_data is not None and not activity_data.empty:
        df_raw["Activity"] = np.interp(
            df_raw["TimeRel"],
            activity_data["TimeSinceReference"] - injection_sec,
            activity_data["Activity"],
            left=np.nan,
            right=np.nan,
        )

    return df_raw
