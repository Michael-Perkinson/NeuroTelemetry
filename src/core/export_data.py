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
            'Start Time': window_start_time,
            'End Time': window_end_time,
            'Number of Peaks': None,
            'Duration (s)': window_end_time - window_start_time,
            'Peaks per Minute': None,
            'Description': 'Overall Time Window'
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
                    'Start Time': valid_start,
                    'End Time': valid_end,
                    'Number of Peaks': num_valid_peaks,
                    'Duration (s)': duration,
                    'Peaks per Minute': ppm,
                    'Description': 'Valid Period'
                })

    return summary_data


def export_mrp_data_to_excel(
    summary_data: List[Dict[str, Any]],
    all_metrics: Dict[str, Dict[str, pd.DataFrame]],
    data_file_path: str
) -> None:
    """
    Export summary and detailed respiratory analysis results to Excel.
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

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary Data', index=False)

            for window, periods in all_metrics.items():
                window_data = []
                for period_index, (period_name, metrics_df) in enumerate(periods.items(), start=1):
                    window_data.append([f'Valid Period {period_index}'])
                    window_data.append(metrics_df.columns.tolist())
                    window_data.extend(metrics_df.values.tolist())
                    window_data.append([])

                window_df = pd.DataFrame(window_data)

                clean_window = ''.join(c if c.isalnum() or c in (
                    ' ', '_') else '_' for c in window)
                clean_window = clean_window[:31]  # Excel limit

                window_df.to_excel(
                    writer, sheet_name=clean_window, index=False, header=False)

        print(f'Data successfully exported to {excel_path}')

    except Exception as e:
        print(f'Failed to export data to Excel: {e}')



# summary_data.append({
#     'Start Time': valid_start,
#     'End Time': valid_end,
#     'Number of Peaks': num_valid_peaks,
#     'Duration (s)': valid_end - valid_start,
#     'Peaks per Minute': (num_valid_peaks / (valid_end - valid_start)) * 60,
#     'Description': 'Valid Period'
# })
