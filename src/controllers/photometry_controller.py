import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.adaptive_algorithms import compute_time_window, get_time_bounds
from src.core.data_alignment import extract_and_process_data, prepare_raw_data
from src.core.data_file_parser import (
    read_and_process_photometry_file,
    retrieve_telemetry_data,
)
from src.core.export_data import (
    export_binned_data_to_excel,
)
from src.core.export_graphs import (
    export_full_time_range_plot,
)
from src.core.file_handling import create_folders_for_graphs, list_files
from src.core.logger import log_exception, log_info
from src.core.photometry_metrics import (
    bin_signal,
    combine_signal_bins,
    make_bin_edges,
    trim_to_window,
)
from src.core.photometry_peaks import analyse_photometry_peaks, bin_peaks

MAIN_SIGNAL = "dFoF_465"


def load_photometry_data(
    photometry_path: Path,
    telemetry_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load photometry and event files with logging + error context."""
    try:
        log_info(f"Loading photometry: {photometry_path}")
        log_info(f"Loading telemetry file: {telemetry_path}")
        photometry_df = read_and_process_photometry_file(photometry_path)
        telemetry_df = retrieve_telemetry_data(telemetry_path)
        log_info("Data loaded successfully.")
        return photometry_df, telemetry_df

    except Exception as e:
        log_exception(e)
        photometry_dir_files = list_files(photometry_path.parent, ["csv", "txt"])
        telemetry_dir_files = list_files(telemetry_path.parent, ["csv", "txt"])

        error_message = (
            f"Data loading failed.\n\n"
            f"Photometry file attempted:\n{photometry_path}\n\n"
            f"{photometry_dir_files}\n\n"
            f"Telemetry file attempted:\n{telemetry_path}\n\n"
            f"{telemetry_dir_files}"
        )
        raise RuntimeError(error_message) from e


def run_photometry_pipeline(
    telemetry_df: pd.DataFrame,
    photometry_df: pd.DataFrame,
    photometry_align_time: str,
    injection_sec: int,
    pre_min: int,
    post_min: int,
    bin_min: int,
    telemetry_path: Path,
    log_callback=None,
) -> dict | None:
    """Run full analysis pipeline for photometry aligned to injection events."""

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    if pre_min <= 0 or post_min <= 0 or bin_min <= 0:
        raise ValueError("Pre, Post, and Bin durations must all be positive.")

    start_time = datetime.now()

    # ---- Setup analysis folder ----
    file_base = telemetry_path.stem
    analysis_folder = (
        telemetry_path.parent / "extracted_data" / f"{file_base}_PhotometryAnalysis"
    )
    analysis_folder.mkdir(parents=True, exist_ok=True)

    date_str = start_time.strftime("%Y%m%d_%H%M%S")

    metadata = {
        "RunDate": date_str,
        "PipelineVersion": "1.0.0",
        "TelemetryFile": str(telemetry_path),
        "InjectionTime_s": injection_sec,
        "PreWindow_min": pre_min,
        "PostWindow_min": post_min,
        "BinSize_min": bin_min,
        "AlignTime": photometry_align_time,
    }

    (analysis_folder / f"analysis_config_{date_str}.json").write_text(
        json.dumps(metadata, indent=2)
    )
    excel_filename = f"{file_base}_photometry_{date_str}.xlsx"

    log("Processing telemetry...")
    processed_telemetry = extract_and_process_data(
        telemetry_df,
        behaviour_data=None,
        probe_date_time=photometry_align_time,
        alignment_date_time=photometry_align_time,
    )
    temp_data = pd.DataFrame(processed_telemetry.get("Temp"))
    activity_data = pd.DataFrame(processed_telemetry.get("Activity"))

    log("Processing photometry data...")
    photo_min_time, photo_max_time = get_time_bounds(photometry_df)
    window_start, window_end = compute_time_window(
        photo_min_time, photo_max_time, injection_sec, pre_min, post_min
    )

    photometry_df = photometry_df[
        (photometry_df["TimeSinceReference"] >= window_start)
        & (photometry_df["TimeSinceReference"] <= window_end)
    ].reset_index(drop=True)

    if temp_data is not None:
        temp_data = trim_to_window(temp_data, window_start, window_end)
    if activity_data is not None:
        activity_data = trim_to_window(activity_data, window_start, window_end)

    if photometry_df.empty:
        log("Photometry data is empty after time-window trimming; aborting.")
        return None

    per_peak_df, peak_times, summary = analyse_photometry_peaks(
        photometry_df, main_signal_col=MAIN_SIGNAL
    )

    df_raw = prepare_raw_data(photometry_df, temp_data, activity_data, injection_sec)

    log("Summarizing data...")
    bin_edges = make_bin_edges(pre_min, post_min, bin_min)
    photometry_binned = bin_signal(photometry_df, bin_edges, MAIN_SIGNAL, injection_sec)
    temp_binned = bin_signal(temp_data, bin_edges, "Temp", injection_sec)
    activity_binned = bin_signal(activity_data, bin_edges, "Activity", injection_sec)
    peaks_binned = bin_peaks(per_peak_df, bin_edges, injection_sec)
    signal_binned = combine_signal_bins(photometry_binned, temp_binned, activity_binned)

    log("Creating output folders...")
    html_folder, svg_folder, full_trace_folder, _ = create_folders_for_graphs(
        analysis_folder
    )

    log("Exporting full time-range plot...")
    export_full_time_range_plot(
        photometry_df,
        temp_data if temp_data is not None else pd.DataFrame(),
        activity_data if activity_data is not None else pd.DataFrame(),
        peak_times,
        [],
        window_start,
        window_end,
        "Injection Window",
        str(full_trace_folder),
        file_base,
        main_signal_col=MAIN_SIGNAL,
        main_signal_label="Photometry",
    )

    # ---- Excel ----
    try:
        log("Saving Excel output...")
        export_binned_data_to_excel(
            output_folder=analysis_folder,
            excel_filename=excel_filename,
            peaks_binned=peaks_binned,
            signal_binned=signal_binned,
            df_raw=df_raw,
        )
    except Exception as e:
        log(f"Failed to save Excel output: {e}")
        raise

    log(f"Total runtime: {(datetime.now() - start_time).total_seconds():.1f}s")

    return {
        "summary": summary,
        "per_peak_df": per_peak_df,
        "signal_binned": signal_binned,
        "peaks_binned": peaks_binned,
        "analysis_folder": analysis_folder,
        "excel_path": analysis_folder / excel_filename,
        "config_path": analysis_folder / f"analysis_config_{date_str}.json",
    }
