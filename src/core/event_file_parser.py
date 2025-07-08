from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    'event': 'behavior',
    'start': 'start (s)',
    'stop': 'stop (s)',
    'duration': 'duration (s)'
}


def read_and_process_event_file(event_file_path: Path) -> pd.DataFrame:
    """Read the CSV file and return a DataFrame with events formatted."""
    df = pd.read_csv(event_file_path)
    df.columns = [col.lower() for col in df.columns]

    column_map = {}
    for key, expected_substring in REQUIRED_COLUMNS.items():
        matches = [col for col in df.columns if expected_substring in col]
        if not matches:
            raise ValueError(
                f"Missing required column containing: '{expected_substring}'")
        column_map[key] = matches[0]

    df['event'] = df[column_map['event']]
    df['instance'] = df.groupby('event').cumcount() + 1
    df['start'] = df[column_map['start']]
    df['end'] = df[column_map['stop']]
    df['duration'] = df[column_map['duration']]

    return df[['event', 'instance', 'start', 'end', 'duration']]
