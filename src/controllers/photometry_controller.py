from pathlib import Path
import pandas as pd
import numpy as np

from src.core.export_data import create_summary_data, export_data_to_excel
from src.core.export_graphs import (
    export_full_time_range_plot,
    export_behavior_images_interactive
)
from src.core.file_handling import create_folders_for_graphs, list_files
from src.core.adaptive_algorithms import get_time_bounds
from src.core.period_analysis import find_valid_periods
from src.core.event_file_parser import read_and_process_event_file, select_time_windows
# or a photometry-specific reader
from src.core.data_file_parser import retrieve_telemetry_data
from src.core.data_alignment import extract_and_process_data
from src.core.logger import log_info, log_exception


def load_data(photometry_path: Path, event_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load photometry and event files with logging + error context."""
    try:
        log_info(f"Loading photometry: {photometry_path}")
        log_info(f"Loading event file: {event_path}")
        photometry_df = retrieve_telemetry_data(photometry_path)
        event_df = read_and_process_event_file(event_path)
        log_info("Data loaded successfully.")
        return photometry_df, event_df

    except Exception as e:
        log_exception(e)
        photometry_dir_files = list_files(
            photometry_path.parent, ['csv', 'txt'])
        event_dir_files = list_files(event_path.parent, ['csv', 'txt'])

        error_message = (
            f"Data loading failed.\n\n"
            f"Photometry file attempted:\n{photometry_path}\n\n"
            f"{photometry_dir_files}\n\n"
            f"Event file attempted:\n{event_path}\n\n"
            f"{event_dir_files}"
        )
        raise RuntimeError(error_message)


def run_photometry_pipeline(
    telemetry_df: pd.DataFrame,
    photometry_df: pd.DataFrame,
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

    log("Preparing behavior data...")

    log("Processing photometry data...")
    processed_data = extract_and_process_data(
        photometry_df,
        behaviour_data,
        probe_time,
        video_time,
    )

    # Unpack — adapt depending on what extract_and_process_data returns
    signal_data = processed_data["Photometry"]  # e.g. raw/cleaned signals
    new_reference_timestamp = processed_data["ReferenceTimestamp"]

    log("Selecting time windows...")
    time_windows = select_time_windows(
        behaviour_to_plot,
        behaviour_data,
        new_reference_timestamp
    )

    if not time_windows:
        log("No valid time windows found.")
        return

    log(f"Found {len(time_windows)} time windows.")
    log("Performing ΔF/F processing...")

    # --- PLACEHOLDER: implement proper ΔF/F ---
    baseline = signal_data["Signal"].rolling(
        window=1000, min_periods=1).median()
    signal_data["dFF"] = (signal_data["Signal"] - baseline) / baseline

    log("Finding valid periods...")
    min_time, max_time = get_time_bounds(signal_data)

    # Placeholder summary until you add specific photometry metrics
    summary_data = create_summary_data([], [], time_windows)

    log("Creating output folders...")
    html_folder, svg_folder, full_trace_folder, file_base = create_folders_for_graphs(
        output_path)

    log("Exporting full time-range plot...")
    export_full_time_range_plot(
        photometry_data,
        temp_data,
        activity_data,
        valid_peak_times_all,
        valid_pre_peak_times_all,
        min_time,
        max_time,
        behaviour_to_plot,
        full_trace_folder,
        file_base,
        main_signal_col="dFF",
        main_signal_label="Photometry"
    )
    
    # log("Exporting per-behavior plots...")
    # export_behavior_images_interactive(
    #     time_windows,
    #     signal_data,
    #     None,
    #     None,
    #     [], [],
    #     behaviour_to_plot,
    #     html_folder,
    #     svg_folder,
    #     file_base
    # )

    log("Saving Excel output...")
    export_data_to_excel(summary_data, {}, output_path)
    log("✅ Photometry analysis complete.")
