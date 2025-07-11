import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any


def create_summary_data(
    valid_peak_times_all: List[float],
    updated_valid_periods: List[Tuple[float, float]],
    time_windows: List[Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """
    Create a summary of valid respiratory periods and total peaks per time window.

    Returns a list of dictionaries suitable for DataFrame export.
    """
    summary_data = []

    for window_start_time, window_end_time in time_windows:
        # Time window block
        summary_data.append({
            'Description': 'Overall Time Window',
            'Start Time': window_start_time,
            'End Time': window_end_time,
            'Number of Peaks': None,
            'Duration (s)': window_end_time - window_start_time,
            'Peaks per Minute': None,
        })

        # Valid periods within this window
        for valid_start, valid_end in updated_valid_periods:
            if window_start_time <= valid_start <= window_end_time and window_start_time <= valid_end <= window_end_time:
                num_valid_peaks = sum(
                    valid_start <= peak <= valid_end for peak in valid_peak_times_all)
                duration = valid_end - valid_start
                ppm = (num_valid_peaks / duration) * \
                    60 if duration > 0 else None

                summary_data.append({
                    'Description': 'Valid Period',
                    'Start Time': valid_start,
                    'End Time': valid_end,
                    'Number of Peaks': num_valid_peaks,
                    'Duration (s)': duration,
                    'Peaks per Minute': ppm,
                })

    return summary_data


def export_data_to_excel(
    summary_data: List[Dict[str, Any]],
    all_metrics: Dict[str, Dict[str, Any]],
    data_file_path: str
) -> None:
    """
    Export all metrics to Excel with 4 clear sheets:
    1. Summary Data
    2. Per Bin
    3. Per Period
    4. Per Window
    """
    try:
        summary_df = pd.DataFrame(summary_data)

        file_base = os.path.splitext(os.path.basename(data_file_path))[0]
        extracted_data_folder = 'extracted_data'
        analysis_folder = os.path.join(
            extracted_data_folder, f'{file_base}_MRP_analysis')
        os.makedirs(analysis_folder, exist_ok=True)

        current_date = datetime.now().strftime('%Y%m%d')
        excel_filename = f'{file_base}_MRP_analysis_{current_date}_sleep_analysis.xlsx'
        excel_path = os.path.join(analysis_folder, excel_filename)

        # Prepare the 3 detailed sheets
        per_bin_rows = []
        per_period_rows = []
        per_window_rows = []

        for window_key, window_content in all_metrics.items():
            # Window-level summary
            window_summary = window_content.get("CombinedSummary", {})
            if window_summary:
                window_row = {"Window": window_key, **window_summary}
                per_window_rows.append(window_row)

            for period_name, period_data in window_content.get("Periods", {}).items():
                # Binned rows
                binned_df = period_data.get("Binned", pd.DataFrame())
                if not binned_df.empty:
                    binned_df = binned_df.copy()
                    binned_df.insert(0, "Period", period_name)
                    binned_df.insert(0, "Window", window_key)
                    per_bin_rows.append(binned_df)

                # Period summary
                period_summary = period_data.get("Summary", {})
                if period_summary:
                    period_row = {
                        "Window": window_key,
                        "Period": period_name,
                        **period_summary
                    }
                    per_period_rows.append(period_row)

        # Combine binned rows into one DataFrame
        per_bin_df = pd.concat(
            per_bin_rows, ignore_index=True) if per_bin_rows else pd.DataFrame()
        per_period_df = pd.DataFrame(per_period_rows)
        per_window_df = pd.DataFrame(per_window_rows)

        # Write to Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary Data', index=False)
            per_bin_df.to_excel(writer, sheet_name='Per Bin', index=False)
            per_period_df.to_excel(
                writer, sheet_name='Per Period', index=False)
            per_window_df.to_excel(
                writer, sheet_name='Per Window', index=False)

        print(f'Data successfully exported to {excel_path}')

    except Exception as e:
        print(f'Failed to export data to Excel: {e}')
