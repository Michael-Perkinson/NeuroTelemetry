from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.export_data import create_summary_data, export_data_to_excel, export_binned_data_to_excel
from src.core.export_graphs import (
    export_full_time_range_plot,
    export_behavior_images_interactive
)
from src.core.file_handling import create_folders_for_graphs, list_files
from src.core.adaptive_algorithms import get_time_bounds
from src.core.period_analysis import find_valid_periods
from src.core.event_file_parser import read_and_process_event_file, select_time_windows

from src.core.data_file_parser import retrieve_telemetry_data, read_and_process_photometry_file
from src.core.data_alignment import extract_and_process_data, prepare_raw_data
from src.core.logger import log_info, log_exception
from src.core.photometry_peaks import analyse_photometry_peaks, bin_peaks
from src.core.photometry_metrics import bin_signal, combine_signal_bins

def load_photometry_data(photometry_path: Path, telemetry_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load photometry and event files with logging + error context."""
    try:
        log_info(f"Loading photometry: {photometry_path}")
        log_info(f"Loading telemetry file: {telemetry_path}")
        photometry_df = read_and_process_photometry_file(
            photometry_path)
        telemetry_df = retrieve_telemetry_data(telemetry_path)
        log_info("Data loaded successfully.")
        return photometry_df, telemetry_df

    except Exception as e:
        log_exception(e)
        photometry_dir_files = list_files(
            photometry_path.parent, ['csv', 'txt'])
        telemetry_dir_files = list_files(telemetry_path.parent, ['csv', 'txt'])

        error_message = (
            f"Data loading failed.\n\n"
            f"Photometry file attempted:\n{photometry_path}\n\n"
            f"{photometry_dir_files}\n\n"
            f"Telemetry file attempted:\n{telemetry_path}\n\n"
            f"{telemetry_dir_files}"
        )
        raise RuntimeError(error_message)


def run_photometry_pipeline(
    telemetry_df: pd.DataFrame,
    photometry_df: pd.DataFrame,
    photometry_align_time: str,
    injection_sec: int,
    pre_minutes: int,
    post_minutes: int,
    bin_minutes: int,
    output_path: Path,
    log_callback=None,
):
    """End-to-end pipeline for analyzing fiber photometry data."""
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    # ---- Setup analysis folder ----
    file_base = Path(output_path).stem
    analysis_folder = Path("extracted_data") / \
        f"{file_base}_PhotometryAnalysis"
    analysis_folder.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y%m%d')
    excel_filename = f"{file_base}_photometry_{date_str}.xlsx"

    # ---- Process telemetry ----
    log("Processing telemetry...")
    processed_telemetry = extract_and_process_data(
        telemetry_df,
        behaviour_data=None,
        probe_date_time=photometry_align_time,
        alignment_date_time=photometry_align_time
    )
    temp_data = processed_telemetry.get("Temp")
    activity_data = processed_telemetry.get("Activity")

    # ---- Process photometry ----
    log("Processing photometry data...")
    photo_min_time, photo_max_time = get_time_bounds(photometry_df)
    start_time = injection_sec - pre_minutes * 60
    end_time = injection_sec + post_minutes * 60
    window_start = max(photo_min_time, start_time)
    window_end = min(photo_max_time, end_time)

    photometry_df = photometry_df[
        (photometry_df["TimeSinceReference"] >= window_start) &
        (photometry_df["TimeSinceReference"] <= window_end)
    ].reset_index(drop=True)

    if temp_data is not None:
        temp_data = temp_data.query(
            "@window_start <= TimeSinceReference <= @window_end")
    if activity_data is not None:
        activity_data = activity_data.query(
            "@window_start <= TimeSinceReference <= @window_end")

    per_peak_df, summary = analyse_photometry_peaks(
        photometry_df, main_signal_col="dFoF_465"
    )
    
    df_raw = prepare_raw_data(
        photometry_df,
        temp_data,
        activity_data,
        injection_sec
    )

    # ---- Summaries & binning ----
    log("Summarizing data...")
    bin_size_sec = bin_minutes * 60

    # Build bins relative to injection (0 = injection point)
    bin_start = -pre_minutes * 60
    bin_end = post_minutes * 60
    bin_edges = np.arange(bin_start, bin_end + bin_size_sec, bin_size_sec)

    peak_times = per_peak_df["PeakTime"].tolist(
    ) if not per_peak_df.empty else []

    photometry_binned = bin_signal(
        photometry_df, bin_edges, "dFoF_465", injection_sec)
    temp_binned = bin_signal(temp_data, bin_edges, "Temp", injection_sec)
    activity_binned = bin_signal(
        activity_data, bin_edges, "Activity", injection_sec)
    peaks_binned = bin_peaks(per_peak_df, bin_edges, injection_sec)

    signal_binned = combine_signal_bins(
        photometry_binned,
        temp_binned,
        activity_binned
    )

    # ---- Plots ----
    log("Creating output folders...")
    html_folder, svg_folder, full_trace_folder, _ = create_folders_for_graphs(
        analysis_folder
    )
    log("Exporting full time-range plot...")
    export_full_time_range_plot(
        photometry_df,
        temp_data,
        activity_data,
        peak_times,
        [],
        window_start,
        window_end,
        "Injection Window",
        full_trace_folder,
        file_base,
        main_signal_col="dFoF_465",
        main_signal_label="Photometry"
    )

    # ---- Excel ----
    log("Saving Excel output...")
    export_binned_data_to_excel(
        output_folder=analysis_folder,
        excel_filename=excel_filename,
        peaks_binned=peaks_binned,
        signal_binned=signal_binned,
        df_raw=df_raw
    )

    log("Photometry analysis complete.")
