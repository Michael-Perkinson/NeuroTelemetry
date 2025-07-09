import pandas as pd
from pathlib import Path

from file_handling import list_files

# Temp paths (you can replace these with real input later)
data_file_path = Path(
    'data/Day 2 (25-04-24) Pro - this is the NaNs file/B1 virgin Day 2 (25-04-24) ponemah.csv')
behaviour_file_path = Path(
    'data/Day 2 (25-04-24) Pro - this is the NaNs file/B1 virgin Day 2 (25-04-24).csv')


def retrieve_telemetry_data() -> pd.DataFrame:
    """Read telemetry data by skipping to the actual header and ignoring meta data."""
    if data_file_path.exists():
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip().startswith('Time'):
                        skiprows = i
                        break
                else:
                    skiprows = 0  # fallback if 'Time' not found
            return pd.read_csv(data_file_path, sep='\t', skiprows=skiprows)
        except Exception as e:
            print(f"Failed to read telemetry file: {e}")
            list_files(data_file_path.parent, 'ascii')
            return pd.DataFrame()
    else:
        print("No valid data file path provided.")
        list_files(Path.cwd(), 'ascii')
        return pd.DataFrame()