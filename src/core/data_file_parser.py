import pandas as pd
from pathlib import Path

# Temp paths (you can replace these with real input later)
data_file_path = Path(
    'data/Day 2 (25-04-24) Pro - this is the NaNs file/B1 virgin Day 2 (25-04-24) ponemah.csv')
behaviour_file_path = Path(
    'data/Day 2 (25-04-24) Pro - this is the NaNs file/B1 virgin Day 2 (25-04-24).csv')


def retrieve_telemetry_data(data_file_path: Path) -> pd.DataFrame:
    """Read telemetry data by skipping to the actual header and ignoring meta data."""
    if not data_file_path.exists():
        raise FileNotFoundError(f"File not found: {data_file_path}")

    with open(data_file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if line.strip().startswith('Time'):
                skiprows = i
                break
        else:
            skiprows = 0  # fallback if 'Time' not found

    return pd.read_csv(data_file_path, sep='\t', skiprows=skiprows)

