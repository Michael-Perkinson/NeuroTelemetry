import numpy as np
import pandas as pd
from typing import Optional


def bin_signal(
    signal_df: pd.DataFrame,
    bin_size_sec: float,
    signal_col: str
) -> pd.DataFrame:
    """
    Bin signal data into fixed time windows relative to the clipped window start.
    """
    t_abs = signal_df["TimeSinceReference"].to_numpy()
    y = signal_df[signal_col].to_numpy()

    # Anchor bins to the first timestamp (absolute), but export relative
    t0 = t_abs.min()
    bins_abs = np.arange(t0, t_abs.max() + bin_size_sec, bin_size_sec)
    bin_indices = np.digitize(t_abs, bins_abs) - 1

    results = []
    for i in range(len(bins_abs) - 1):
        mask = bin_indices == i
        vals = y[mask]
        if len(vals) == 0:
            mean, sem = np.nan, np.nan
        else:
            mean = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
        results.append([bins_abs[i] - t0, bins_abs[i+1] - t0, mean, sem])

    return pd.DataFrame(
        results, columns=["BinStart", "BinEnd", "Mean", "SEM"]
    )


def combine_signal_bins(
    photometry_binned: pd.DataFrame,
    temp_binned: Optional[pd.DataFrame],
    activity_binned: Optional[pd.DataFrame]
) -> pd.DataFrame:
    # Base bins from photometry
    df = photometry_binned.rename(
        columns={
            "Mean": "dFoF_Mean",
            "SEM": "dFoF_SEM",
        }
    ).copy()

    key_cols = ["BinStart", "BinEnd"]

    if temp_binned is not None:
        temp_binned = temp_binned.rename(
            columns={
                "Mean": "Temp_Mean",
                "SEM": "Temp_SEM",
            }
        )[key_cols + ["Temp_Mean", "Temp_SEM"]]
        df = df.merge(temp_binned, on=key_cols, how="left")

    if activity_binned is not None:
        activity_binned = activity_binned.rename(
            columns={
                "Mean": "Act_Mean",
                "SEM": "Act_SEM",
            }
        )[key_cols + ["Act_Mean", "Act_SEM"]]
        df = df.merge(activity_binned, on=key_cols, how="left")

    # Final column order without counts
    return df[
        ["BinStart", "BinEnd",
         "dFoF_Mean", "dFoF_SEM",
         "Temp_Mean", "Temp_SEM",
         "Act_Mean", "Act_SEM"]
    ]
