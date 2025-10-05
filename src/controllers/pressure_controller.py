from pathlib import Path

import pandas as pd

from src.core.adaptive_algorithms import get_time_bounds
from src.core.data_alignment import extract_and_process_data
from src.core.data_file_parser import retrieve_telemetry_data
from src.core.event_file_parser import read_and_process_event_file, select_time_windows
from src.core.export_data import create_summary_data, export_data_to_excel
from src.core.export_graphs import (
    export_behavior_images_interactive,
    export_full_time_range_plot,
)
from src.core.file_handling import create_folders_for_graphs, list_files
from src.core.logger import log_exception, log_info
from src.core.peak_detection import analyse_peaks
from src.core.period_analysis import find_valid_periods


def load_data(
    telemetry_path: Path,
    event_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load telemetry and event data from disk into DataFrames."""
    try:
        log_info(f"Loading telemetry: {telemetry_path}")
        log_info(f"Loading event file: {event_path}")
        telemetry_df = retrieve_telemetry_data(telemetry_path)
        event_df = read_and_process_event_file(event_path)
        log_info("Data loaded successfully.")
        return telemetry_df, event_df

    except Exception as e:
        log_exception(e)

        telemetry_dir_files = list_files(telemetry_path.parent, ["csv", "txt", "ascii"])
        event_dir_files = list_files(event_path.parent, ["csv", "txt", "ascii"])

        error_message = (
            f"Data loading failed.\n\n"
            f"Telemetry file attempted:\n{telemetry_path}\n\n"
            f"{telemetry_dir_files}\n\n"
            f"Event file attempted:\n{event_path}\n\n"
            f"{event_dir_files}"
        )

        raise RuntimeError(error_message) from e


def run_pressure_pipeline(
    telemetry_df: pd.DataFrame,
    event_df: pd.DataFrame,
    behaviour_to_plot: str,
    probe_time: str,
    video_time: str,
    bin_size_sec: int,
    output_path: Path,
    log_callback=None,
):
    """Run full analysis pipeline for pressure telemetry aligned to behaviors."""

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log("Preparing behavior data...")
    behaviour_data = {}
    for event, group in event_df.groupby("event"):
        tuples = group[["instance", "start", "end", "duration"]].itertuples(
            index=False, name=None
        )
        behaviour_data[event] = list(tuples)

    log("Processing telemetry...")
    processed_data = extract_and_process_data(
        telemetry_df,
        behaviour_data,
        probe_time,
        video_time,
    )

    pressure_data = processed_data.get("Pressure")
    temp_data = processed_data.get("Temp")
    activity_data = processed_data.get("Activity")
    new_reference_timestamp = processed_data["ReferenceTimestamp"]

    log("Selecting time windows...")
    time_windows = select_time_windows(
        behaviour_to_plot, processed_data["Behaviours"], new_reference_timestamp
    )

    if not time_windows:
        log("No valid time windows found.")
        return

    log(f"Found {len(time_windows)} time windows.")
    log("Analyzing shoulders...")

    results = {
        f"{start}-{end}": analyse_peaks(
            time_windows=[(start, end)],
            pressure_data=pressure_data,
        )
        for start, end in time_windows
    }

    log("Extracting peak timings...")
    all_peak_times = []
    all_pre_peak_times = []
    for window_results in results.values():
        for result in window_results:
            _, _, _, peak_times, _, pre_peak_times = result
            all_peak_times.extend(peak_times)
            all_pre_peak_times.extend(pre_peak_times)

    log("Finding valid periods...")
    (
        valid_peak_times_all,
        valid_pre_peak_times_all,
        updated_valid_periods,
        all_metrics,
    ) = find_valid_periods(
        results, pressure_data, temp_data, activity_data, time_windows, bin_size_sec
    )

    min_time, max_time = get_time_bounds(pressure_data)

    summary_data = create_summary_data(
        valid_peak_times_all, updated_valid_periods, time_windows
    )

    log("Creating output folders...")
    html_folder, svg_folder, full_trace_folder, file_base = create_folders_for_graphs(
        output_path
    )

    log("Exporting full time-range plot...")
    export_full_time_range_plot(
        pressure_data,
        temp_data,
        activity_data,
        valid_peak_times_all,
        valid_pre_peak_times_all,
        min_time,
        max_time,
        behaviour_to_plot,
        full_trace_folder,
        file_base,
        main_signal_col="SmoothedPressure",
        main_signal_label="Pressure",
    )

    log("Exporting per-behavior plots...")
    export_behavior_images_interactive(
        time_windows,
        pressure_data,
        temp_data,
        activity_data,
        valid_peak_times_all,
        valid_pre_peak_times_all,
        behaviour_to_plot,
        html_folder,
        svg_folder,
        file_base,
        main_signal_col="SmoothedPressure",
        main_signal_label="Pressure",
    )

    log("Saving Excel output...")
    export_data_to_excel(summary_data, all_metrics, output_path)
    log("✅ Analysis complete.")
