import pandas as pd
from pathlib import Path

def retrieve_telemetry_data(data_file_path: Path) -> pd.DataFrame:
    """Read the full telemetry file (including metadata)."""
    return pd.read_csv(data_file_path, sep='\t', header=None)


def detect_skip_rows(data: pd.DataFrame) -> int:
    """Detect the index of the first row containing actual telemetry data."""
    for i, line in enumerate(data.iloc[:, 0]):
        if str(line).strip().startswith('Time'):
            return i
    return 0
