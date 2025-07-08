from pathlib import Path
import pandas as pd
from collections import defaultdict

from file_handling import list_files

REQUIRED_COLUMNS = {
    'event': 'behavior',
    'start': 'start (s)',
    'stop': 'stop (s)',
    'duration': 'duration (s)'
}


def read_and_process_event_file(file_path: Path) -> pd.DataFrame:
    """Read the CSV file and return a DataFrame with events formatted."""
    df = pd.read_csv(file_path)
    df.columns = [col.lower() for col in df.columns]

    column_map = {}
    for key, expected_substring in REQUIRED_COLUMNS.items():
        matches = [col for col in df.columns if expected_substring in col]
        if not matches:
            raise ValueError(
                f"Missing required column containing: '{expected_substring}'")
        column_map[key] = matches[0]

    data = []
    counter = defaultdict(int)

    for _, row in df.iterrows():
        event = row[column_map['event']]
        counter[event] += 1
        data.append((event, counter[event],
                     row[column_map['start']],
                     row[column_map['stop']],
                     row[column_map['duration']]))

    return pd.DataFrame(data, columns=["event", "instance", "start", "end", "duration"])


