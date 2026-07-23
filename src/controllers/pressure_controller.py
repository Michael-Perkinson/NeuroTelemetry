import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.adaptive_algorithms import get_time_bounds
from src.core.data_alignment import extract_and_process_data
from src.core.data_file_parser import retrieve_telemetry_data, safe_get_df
from src.core.event_file_parser import (
    classify_behaviour_windows,
    read_and_process_event_file,
    select_time_windows,
    structure_behaviour_events,
)
from src.core.export_data import create_summary_data, export_data_to_excel
from src.core.export_graphs import (
    export_behavior_images_interactive,
    export_full_time_range_plot,
)
from src.core.file_handling import create_folders_for_graphs, list_files
from src.core.logger import log_exception, log_info
from src.core.peak_detection import analyse_peaks
from src.core.period_analysis import (
    compute_respiratory_metrics_for_periods,
    find_valid_periods,
)
from src.core.power_spectral_analysis import (
    DEFAULT_PSD_AUC_BAND_HZ,
    DEFAULT_PSD_END_TRIM_SAMPLES,
    DEFAULT_PSD_NFFT,
    DEFAULT_PSD_OVERLAP_FRACTION,
    DEFAULT_PSD_RESAMPLE_HZ,
    DEFAULT_PSD_SEGMENT_SECONDS,
    DEFAULT_PSD_SMOOTH_NPERSEG,
    DEFAULT_PSD_WELCH_WINDOWS,
    DEFAULT_PSD_WINDOW,
    analyze_ttot_psd,
    export_ttot_traces,
    plot_psd_results,
)
from src.core.respiratory_metrics import (
    compute_atm_pressure_session_summary,
    compute_atm_pressure_time_bins,
)


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
    video_time: str,
    bin_size_sec: int,
    output_path: Path,
    atm_bin_size_sec: int = 300,
    export_atm_summary: bool = True,
    log_callback=None,
    analysis_root: Path | None = None,
    event_path: Path | None = None,
) -> dict | None:
    """Run full analysis pipeline for pressure telemetry aligned to behaviors."""

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            log_info(msg)

    start_time = datetime.now()
    file_base = output_path.stem
    output_root = analysis_root or (output_path.parent / "extracted_data")
    analysis_folder = output_root / f"{file_base}_PressureAnalysis"
    analysis_folder.mkdir(parents=True, exist_ok=True)

    date_str = start_time.strftime("%Y%m%d_%H%M%S")

    metadata = {
        "ConfigSchemaVersion": 3,
        "RunDate": date_str,
        "BehaviourToPlot": behaviour_to_plot,
        "VideoStartTime": video_time,
        "VideoTime": video_time,
        "AlignmentMethod": "absolute_video_start",
        "AnalysisStatus": "started",
        "TelemetryFile": str(output_path.resolve()),
        "BinSize_s": bin_size_sec,
        "AtmPressureBinSize_s": atm_bin_size_sec,
        "ExportAtmPressureSummary": export_atm_summary,
        "PSDMetric": "Ttot",
        "PSDResampleHz": DEFAULT_PSD_RESAMPLE_HZ,
        "PSDSegmentDuration_s": DEFAULT_PSD_SEGMENT_SECONDS,
        "PSDEndTrimSamples": DEFAULT_PSD_END_TRIM_SAMPLES,
        "PSDWelchSmoothNperseg": DEFAULT_PSD_SMOOTH_NPERSEG,
        "PSDWelchComparisonDivisor": DEFAULT_PSD_WELCH_WINDOWS,
        "PSDWelchOverlapFraction": DEFAULT_PSD_OVERLAP_FRACTION,
        "PSDWelchWindow": DEFAULT_PSD_WINDOW,
        "PSDWelchWindowSampling": "symmetric",
        "PSDNominalWelchResolutionHz": (
            DEFAULT_PSD_RESAMPLE_HZ / DEFAULT_PSD_SMOOTH_NPERSEG
        ),
        "PSDMeanCentered": True,
        "PSDDetrended": False,
        "PSDNFFT": DEFAULT_PSD_NFFT,
        "PSDAUCBandHz": DEFAULT_PSD_AUC_BAND_HZ,
        "PSDAUCBandStatus": "Exploratory mouse adaptation",
    }
    if event_path is not None:
        metadata["EventFile"] = str(event_path.resolve())

    logs_folder = analysis_folder / "logs"
    logs_folder.mkdir(parents=True, exist_ok=True)
    config_path = logs_folder / f"analysis_config_{date_str}.json"
    config_path.write_text(json.dumps(metadata, indent=2))

    log("Preparing behavior data...")
    behaviour_data = structure_behaviour_events(event_df)

    log("Processing telemetry...")
    processed_data = extract_and_process_data(
        telemetry_df,
        behaviour_data,
        video_time,
        timeline_origin="alignment",
    )

    pressure_data = safe_get_df(processed_data, "Pressure")
    temp_data = safe_get_df(processed_data, "Temp")
    activity_data = safe_get_df(processed_data, "Activity")
    atm_pressure_data = safe_get_df(processed_data, "AtmPressure")
    n_atm = len(atm_pressure_data)
    log(f"AtmPressure data: {n_atm} rows, empty={atm_pressure_data.empty}")

    new_reference_timestamp = processed_data["ReferenceTimestamp"]
    metadata["TelemetryReferenceTime"] = pd.Timestamp(
        new_reference_timestamp
    ).isoformat()

    min_time, max_time = get_time_bounds(pressure_data)
    coverage = classify_behaviour_windows(
        behaviour_to_plot,
        processed_data["Behaviours"],
        telemetry_bounds=(min_time, max_time),
        telemetry_intervals=processed_data["PressureAnalyzableIntervals"],
    )
    coverage_counts = Counter(record["status"] for record in coverage)
    metadata["TelemetryFirstValidTimestamp"] = pd.Timestamp(
        processed_data["FirstValidTimestamp"]
    ).isoformat()
    metadata["TelemetryCoverageStart_s"] = float(min_time)
    metadata["TelemetryCoverageEnd_s"] = float(max_time)
    metadata["TelemetryContinuousCoverageIntervals_s"] = processed_data[
        "PressureCoverageIntervals"
    ]
    metadata["TelemetryCoverageMaxGap_s"] = processed_data[
        "PressureCoverageMaxGapSeconds"
    ]
    metadata["TelemetryAnalyzableIntervals_s"] = processed_data[
        "PressureAnalyzableIntervals"
    ]
    metadata["PressureFilterEdgeMargin_s"] = processed_data[
        "PressureFilterEdgeMarginSeconds"
    ]
    metadata["BehaviourCoverageCounts"] = dict(coverage_counts)
    metadata["BehaviourCoverage"] = coverage
    config_path.write_text(json.dumps(metadata, indent=2))

    if coverage:
        log(
            "Behavior coverage: "
            + ", ".join(
                f"{status}={coverage_counts.get(status, 0)}"
                for status in (
                    "fully_covered",
                    "partially_covered",
                    "unavailable",
                    "filtered",
                )
            )
        )
        for record in coverage:
            if record["status"] != "fully_covered":
                start_label = (
                    "missing" if record["start"] is None else f"{record['start']:.3f}s"
                )
                end_label = (
                    "missing" if record["end"] is None else f"{record['end']:.3f}s"
                )
                log(
                    f"Behavior instance {record['instance']} excluded: "
                    f"{record['status']} ({record['reason']}; "
                    f"{start_label} to {end_label})."
                )
    elif behaviour_to_plot not in processed_data["Behaviours"]:
        available = sorted(processed_data["Behaviours"])
        metadata["BehaviourSelectionStatus"] = "label_not_found"
        metadata["AvailableBehaviours"] = available
        config_path.write_text(json.dumps(metadata, indent=2))
        log(
            f"Behavior label {behaviour_to_plot!r} was not found. "
            f"Available labels: {available or 'none'}."
        )

    log("Selecting time windows...")
    time_windows = select_time_windows(
        behaviour_to_plot,
        processed_data["Behaviours"],
        new_reference_timestamp,
        telemetry_bounds=(min_time, max_time),
        telemetry_intervals=processed_data["PressureAnalyzableIntervals"],
    )

    if not time_windows:
        metadata["AnalysisStatus"] = "no_eligible_windows"
        metadata["CompletedAt"] = datetime.now().isoformat()
        config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        no_results_path = analysis_folder / f"NO_RESULTS_{date_str}.txt"
        no_results_path.write_text(
            "No fully covered eligible behavior windows were available.\n"
            f"See logs/{config_path.name} for coverage details.\n",
            encoding="utf-8",
        )
        log("No fully covered eligible time windows found.")
        log(f"No-results marker: {no_results_path}")
        return None

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
        window_periods,
    ) = find_valid_periods(
        results,
        pressure_data,
        time_windows,
    )

    log("Computing respiratory metrics...")
    all_metrics = compute_respiratory_metrics_for_periods(
        window_periods,
        pressure_data,
        temp_data,
        activity_data,
        time_windows,
        bin_size_sec,
    )

    atm_overall_df = None
    atm_binned_df = None
    if export_atm_summary:
        log("Computing atmospheric pressure session summary...")
        atm_overall_df = compute_atm_pressure_session_summary(
            atm_pressure_data=atm_pressure_data,
        )
        atm_binned_df = compute_atm_pressure_time_bins(
            pressure_data=pressure_data,
            atm_pressure_data=atm_pressure_data,
            bin_size_sec=atm_bin_size_sec,
        )
        log(f"Atm overall: {atm_overall_df.shape}, binned: {atm_binned_df.shape}")

    summary_data = create_summary_data(
        valid_peak_times_all, updated_valid_periods, time_windows
    )

    log("Creating output folders...")
    html_folder, svg_folder, full_trace_folder, base_name = create_folders_for_graphs(
        analysis_folder
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
        str(full_trace_folder),
        base_name,
        main_signal_col="SmoothedPressure",
        main_signal_label="Pressure",
        atm_pressure_data=atm_pressure_data,
        behavior_windows=time_windows,
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
        str(html_folder),
        str(svg_folder),
        base_name,
        main_signal_col="SmoothedPressure",
        main_signal_label="Pressure",
    )

    log("Computing Ttot variability PSD...")
    psd_export_folder = analysis_folder / "PSD"
    psd_results = analyze_ttot_psd(
        window_periods=window_periods,
        output_folder=psd_export_folder,
        log=log,
    )
    plot_psd_results(psd_results, psd_export_folder, show=False, log=log)

    log("Saving Excel output...")
    export_data_to_excel(
        summary_data,
        all_metrics,
        analysis_folder,
        session_overall_df=atm_overall_df,
        session_binned_df=atm_binned_df,
        psd_results=psd_results,
    )

    log("Exporting Ttot breath-by-breath traces...")
    ttot_export_folder = analysis_folder / "Ttot_Traces"
    export_ttot_traces(
        window_periods=window_periods, output_folder=ttot_export_folder, log=log
    )

    log(f"Total runtime: {(datetime.now() - start_time).total_seconds():.1f}s")

    metadata["AnalysisStatus"] = "complete"
    metadata["CompletedAt"] = datetime.now().isoformat()
    config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "summary": summary_data,
        "metrics": all_metrics,
        "psd": psd_results,
        "analysis_folder": analysis_folder,
        "time_windows": time_windows,
        "behaviour_coverage": coverage,
        "config_path": config_path,
    }
