import numpy as np
import pandas as pd


def bin_signal(
    signal_df: pd.DataFrame,
    bin_edges: np.ndarray,
    signal_col: str,
    injection_sec: float,
) -> pd.DataFrame:
    """
    Bin signal data into fixed time windows relative to injection time.
    """
    if signal_df is None or signal_df.empty:
        return pd.DataFrame(columns=["BinStart", "BinEnd", "Mean", "SEM"])

    # Convert absolute time → relative (0 = injection)
    t_rel = signal_df["TimeSinceReference"].to_numpy() - injection_sec
    y = signal_df[signal_col].to_numpy()

    bin_indices = np.digitize(t_rel, bin_edges) - 1

    results = []
    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        vals = y[mask]
        if len(vals) == 0:
            mean, sem = np.nan, np.nan
        else:
            mean = float(np.nanmean(vals))
            sem = float(np.nanstd(vals, ddof=1)) / np.sqrt(np.sum(~np.isnan(vals)))

        results.append([bin_edges[i], bin_edges[i + 1], mean, sem])

    return pd.DataFrame(results, columns=["BinStart", "BinEnd", "Mean", "SEM"])


def combine_signal_bins(
    photometry_binned: pd.DataFrame,
    temp_binned: pd.DataFrame | None,
    activity_binned: pd.DataFrame | None,
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

    # Columns now carry bins aligned to injection (negative to positive)
    return df[
        [
            "BinStart",
            "BinEnd",
            "dFoF_Mean",
            "dFoF_SEM",
            "Temp_Mean",
            "Temp_SEM",
            "Act_Mean",
            "Act_SEM",
        ]
    ]
