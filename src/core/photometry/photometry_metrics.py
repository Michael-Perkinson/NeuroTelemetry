import numpy as np
import numpy.typing as npt
import pandas as pd


def trim_to_window(
    df: pd.DataFrame | None, start: float, end: float
) -> pd.DataFrame | None:
    """Return subset of dataframe between time window or None if df is None."""
    if df is None:
        return None
    return df.query("@start <= TimeSinceReference <= @end")


def make_bin_edges(pre_min: int, post_min: int, bin_min: int) -> np.ndarray:
    """Return bin edges in seconds relative to 0 (injection time)."""
    bin_size_sec = bin_min * 60
    bin_start = -pre_min * 60
    bin_end = post_min * 60
    edges = np.arange(bin_start, bin_end, bin_size_sec)
    return np.append(edges, bin_end)


def bin_signal(
    signal_df: pd.DataFrame | None,
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
    y = signal_df[signal_col].to_numpy(dtype=float)

    bin_indices: npt.NDArray[np.intp] = np.asarray(
        np.digitize(t_rel, bin_edges) - 1, dtype=np.intp
    )
    n_bins = len(bin_edges) - 1
    in_range = (bin_indices >= 0) & (bin_indices < n_bins)
    grouped = (
        pd.DataFrame({"Bin": bin_indices[in_range], "Value": y[in_range]})
        .groupby("Bin")["Value"]
        .agg(["count", "mean", "std"])
        .reindex(range(n_bins))
    )

    return pd.DataFrame(
        {
            "BinStart": bin_edges[:-1],
            "BinEnd": bin_edges[1:],
            "Mean": grouped["mean"].to_numpy(),
            "SEM": (grouped["std"] / np.sqrt(grouped["count"])).to_numpy(),
        }
    )


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

    if temp_binned is not None and not temp_binned.empty:
        temp_binned = temp_binned.rename(
            columns={
                "Mean": "Temp_Mean",
                "SEM": "Temp_SEM",
            }
        )[key_cols + ["Temp_Mean", "Temp_SEM"]]
        df = df.merge(temp_binned, on=key_cols, how="left")
    else:
        df["Temp_Mean"] = np.nan
        df["Temp_SEM"] = np.nan

    if activity_binned is not None and not activity_binned.empty:
        activity_binned = activity_binned.rename(
            columns={
                "Mean": "Act_Mean",
                "SEM": "Act_SEM",
            }
        )[key_cols + ["Act_Mean", "Act_SEM"]]
        df = df.merge(activity_binned, on=key_cols, how="left")
    else:
        df["Act_Mean"] = np.nan
        df["Act_SEM"] = np.nan

    for column in ["Temp_Mean", "Temp_SEM", "Act_Mean", "Act_SEM"]:
        if column not in df:
            df[column] = np.nan

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
