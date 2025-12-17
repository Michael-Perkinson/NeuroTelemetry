from pathlib import Path
from typing import Any

import pandas as pd


def retrieve_telemetry_data(data_file_path: Path) -> pd.DataFrame:
    """Read the full telemetry file (including metadata)."""
    return pd.read_csv(data_file_path, sep="\t", header=None)


def read_and_process_photometry_file(photometry_file_path: Path) -> pd.DataFrame:
    """
    Read a photometry CSV file and return a cleaned dataframe
    with only relevant columns for downstream analysis.
    """
    df = pd.read_csv(photometry_file_path)

    if "# t_min" not in df.columns:
        raise ValueError("Photometry file must contain a '# t_min' column (minutes).")

    # Add TimeSinceReference in seconds
    df["TimeSinceReference"] = pd.to_numeric(df["# t_min"], errors="coerce") * 60.0

    keep_cols = ["TimeSinceReference"]
    for col in ["dFoF_465", "dFoF_405", "Z_465"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep_cols.append(col)

    clean_df = df[keep_cols].dropna().reset_index(drop=True)

    if clean_df.empty:
        raise ValueError(f"No valid data found in {photometry_file_path.name}.")

    return clean_df


def detect_skip_rows(data: pd.DataFrame) -> int:
    """Detect the index of the first row containing actual telemetry data."""
    for i, line in enumerate(data.iloc[:, 0]):
        if isinstance(line, str) and line.strip().lower().startswith("time"):
            return i
    return 0


def safe_get_df(data: dict[str, Any], key: str) -> pd.DataFrame:
    df = data.get(key)
    if key == "Pressure" and df is None:
        raise ValueError("Pressure data is required but not found in processed data.")
    return df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()
