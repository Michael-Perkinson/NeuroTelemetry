from pathlib import Path
import pandas as pd


def read_and_process_file(file_path: Path) -> pd.DataFrame | None:
    """Read a tab-delimited file into a DataFrame."""
    try:
        return pd.read_csv(file_path, sep='\t', header=None)
    except Exception as e:
        print(f"Failed to read the file at {file_path}. Error: {e}")
        return None


def list_files(directory: Path, extension: str = 'csv') -> None:
    """List all files with a given extension in a directory."""
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return
    print(f"Listing *.{extension} files in: {directory}")
    for file in directory.glob(f'*.{extension}'):
        print(file)
