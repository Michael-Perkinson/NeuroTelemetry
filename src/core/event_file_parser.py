from pathlib import Path
import pandas as pd
import math

REQUIRED_COLUMNS = {
    'event': 'behavior',
    'start': 'start (s)',
    'stop': 'stop (s)',
    'duration': 'duration (s)'
}


def read_and_process_event_file(event_file_path: Path) -> pd.DataFrame:
    """Read and standardize a behavior event CSV (case-insensitive)."""
    df = pd.read_csv(event_file_path)

    # Lowercase and strip whitespace from all columns
    df.columns = [col.strip().lower() for col in df.columns]

    column_map = {}
    for key, match_str in REQUIRED_COLUMNS.items():
        match_str_lower = match_str.lower()
        matches = [col for col in df.columns if match_str_lower in col]
        if not matches:
            raise ValueError(
                f"Missing required column containing: '{match_str}' (case-insensitive)"
            )
        column_map[key] = matches[0]

    df['event'] = df[column_map['event']]
    df['instance'] = df.groupby('event').cumcount() + 1
    df['start'] = df[column_map['start']]
    df['end'] = df[column_map['stop']]
    df['duration'] = df[column_map['duration']]

    return df[['event', 'instance', 'start', 'end', 'duration']]


def select_time_windows(
    behaviour_to_plot: str,
    behaviour_data: dict,
    min_duration: float,
    reference_timestamp: pd.Timestamp
) -> list[tuple[float, float]]:
    """Filter and extract time windows for a specific behavior."""
    if isinstance(min_duration, tuple):
        min_duration = min_duration[0]

    cutoff_seconds = (reference_timestamp +
                      pd.Timedelta(minutes=90)).timestamp()

    if behaviour_to_plot not in behaviour_data:
        return []

    behaviour_list = behaviour_data[behaviour_to_plot]

    filtered = [
        (inst, start, end, dur) for inst, start, end, dur in behaviour_list
        if start <= cutoff_seconds and end <= cutoff_seconds
        and isinstance(dur, (float, int)) and not math.isnan(dur) and dur >= min_duration
    ]

    return [(start, end) for _, start, end, _ in filtered]
