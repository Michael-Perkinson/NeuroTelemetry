from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_summary_data(
    valid_peak_times_all: list[float],
    updated_valid_periods: list[tuple[float, float]],
    time_windows: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """
    Create a summary of valid respiratory periods and total peaks per time window.

    Returns a list of dictionaries suitable for DataFrame export.
    """
    summary_data = []

    for window_start_time, window_end_time in time_windows:
        # Time window block
        summary_data.append(
            {
                "Description": "Overall Time Window",
                "Start Time": window_start_time,
                "End Time": window_end_time,
                "Number of Peaks": None,
                "Duration (s)": window_end_time - window_start_time,
                "Peaks per Minute": None,
            }
        )

        # Valid periods within this window
        for valid_start, valid_end in updated_valid_periods:
            if (
                window_start_time <= valid_start <= window_end_time
                and window_start_time <= valid_end <= window_end_time
            ):
                num_valid_peaks = sum(
                    valid_start <= peak <= valid_end for peak in valid_peak_times_all
                )
                duration = valid_end - valid_start
                ppm = (num_valid_peaks / duration) * 60 if duration > 0 else None

                summary_data.append(
                    {
                        "Description": "Valid Period",
                        "Start Time": valid_start,
                        "End Time": valid_end,
                        "Number of Peaks": num_valid_peaks,
                        "Duration (s)": duration,
                        "Peaks per Minute": ppm,
                    }
                )

    return summary_data


def insert_blank_rows(dataframes: list[pd.DataFrame], group_key: str) -> pd.DataFrame:
    """Insert blank rows between group changes (e.g., Period or Window)."""
    rows = []
    last_group = None
    for df in dataframes:
        if df.empty:
            continue
        current_group = df[group_key].iloc[0]
        if last_group is not None and current_group != last_group:
            # Append one blank row
            rows.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))
        rows.append(df)
        last_group = current_group
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_export_data(
    all_metrics: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare per-bin, per-period, and per-window dataframes with spacing."""
    per_bin_rows, per_period_rows, per_window_rows = [], [], []

    for window_key, window_content in all_metrics.items():
        if window_key == "GlobalSummary":
            continue

        # --- Window-level summary
        window_summary = window_content.get("WindowSummary", {})
        if window_summary:
            per_window_rows.append({"Window": window_key, **window_summary})

        # --- Per-period and per-bin data
        for period_name, period_data in window_content.get("Periods", {}).items():
            # Binned data
            binned_df = period_data.get("Binned", pd.DataFrame())

            if not binned_df.empty:
                binned_df = binned_df.copy()
                binned_df.insert(0, "Period", period_name)
                binned_df.insert(0, "Window", window_key)
                per_bin_rows.append(binned_df)
            else:
                continue
                # # Placeholder row for too-short or empty periods
                # placeholder = pd.DataFrame([{
                #     "Window": window_key,
                #     "Period": period_name,
                #     "Note": "Not enough time for binning"
                # }])
                # per_bin_rows.append(placeholder)

            # Summary data
            period_summary = period_data.get("Summary", {})
            if period_summary:
                summary_df = pd.DataFrame(
                    [{"Window": window_key, "Period": period_name, **period_summary}]
                )
                per_period_rows.append(summary_df)

    # --- Add spacing between logical groups
    per_bin_df = insert_blank_rows(per_bin_rows, "Period")
    per_period_df = insert_blank_rows(per_period_rows, "Window")
    per_window_df = pd.DataFrame(per_window_rows)

    return per_bin_df, per_period_df, per_window_df


def export_data_to_excel(
    summary_data: list[dict], all_metrics: dict, analysis_folder: Path,
    session_overall_df: pd.DataFrame | None = None,
    session_binned_df: pd.DataFrame | None = None,
) -> None:
    """Export data to Excel into the shared analysis folder alongside the graphs."""
    try:
        summary_df = pd.DataFrame(summary_data)
        global_summary = all_metrics.get("GlobalSummary", {})
        global_summary_df = (
            pd.DataFrame([global_summary]) if global_summary else pd.DataFrame()
        )

        per_bin_df, per_period_df, per_window_df = build_export_data(all_metrics)

        date_str = datetime.now().strftime("%Y%m%d")
        excel_path = analysis_folder / f"{analysis_folder.name}_{date_str}.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary Data", index=False)

            start_row = 0
            if session_overall_df is not None and not session_overall_df.empty:
                # Title row, then df one row below
                session_overall_df.to_excel(
                    writer, sheet_name="Atmospheric Pressure", index=False, startrow=start_row + 1
                )
                ws = writer.sheets["Atmospheric Pressure"]
                ws.cell(row=start_row + 1, column=1).value = "Overall Session"
                start_row += len(session_overall_df) + 3
            if session_binned_df is not None and not session_binned_df.empty:
                session_binned_df.to_excel(
                    writer, sheet_name="Atmospheric Pressure", index=False, startrow=start_row + 1
                )
                ws = writer.sheets["Atmospheric Pressure"]
                ws.cell(row=start_row + 1, column=1).value = "Binned"

            global_summary_df.to_excel(writer, sheet_name="Global Summary", index=False)
            per_bin_df.to_excel(writer, sheet_name="Per Bin", index=False)
            per_period_df.to_excel(writer, sheet_name="Per Period", index=False)
            per_window_df.to_excel(writer, sheet_name="Per Window", index=False)

        print(f"Exported to {excel_path}")

    except Exception as e:
        print(f"Export failed: {e}")


def export_binned_data_to_excel(
    output_folder: Path,
    excel_filename: str,
    peaks_binned: pd.DataFrame,
    signal_binned: pd.DataFrame,
    df_raw: pd.DataFrame,
):
    output_folder.mkdir(parents=True, exist_ok=True)
    out_file = output_folder / excel_filename

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        # Peaks + signals (same sheet as before)
        peaks_binned.to_excel(writer, sheet_name="Binned Data", index=False, startrow=0)
        start_row = len(peaks_binned) + 3
        signal_binned.to_excel(
            writer, sheet_name="Binned Data", index=False, startrow=start_row
        )

        # Raw data in a separate sheet
        df_raw.to_excel(writer, sheet_name="Raw Data", index=False)

    print(f"Exported binned + raw data to {out_file}")
