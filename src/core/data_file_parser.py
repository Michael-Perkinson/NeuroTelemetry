import pandas as pd
from pathlib import Path

def retrieve_telemetry_data(data_file_path: Path) -> pd.DataFrame:
    """Read the full telemetry file (including metadata)."""
    return pd.read_csv(data_file_path, sep='\t', header=None)


def read_and_process_photometry_file(photometry_file_path: Path) -> pd.DataFrame:
    """
    Read a photometry CSV file and return a cleaned dataframe
    with only relevant columns for downstream analysis.

    Keeps:
        TimeSinceReference (seconds),
        dFoF_465,
        dFoF_405 (if present),
        Z_465 (if present).
    """
    df = pd.read_csv(photometry_file_path)

    if "# t_min" not in df.columns:
        raise ValueError(
            "Photometry file must contain a '# t_min' column (minutes).")

    # Add TimeSinceReference in seconds
    df["TimeSinceReference"] = pd.to_numeric(
        df["# t_min"], errors="coerce") * 60.0

    # Build cleaned dataframe
    keep_cols = ["TimeSinceReference"]
    for col in ["dFoF_465", "dFoF_405", "Z_465"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep_cols.append(col)

    clean_df = df[keep_cols].dropna().reset_index(drop=True)
    
    return clean_df


def detect_skip_rows(data: pd.DataFrame) -> int:
    """Detect the index of the first row containing actual telemetry data."""
    for i, line in enumerate(data.iloc[:, 0]):
        if str(line).strip().startswith('Time'):
            return i
    return 0
