"""
Batch runner for the pressure analysis pipeline - no GUI required.

Usage:
    python scripts/batch_run.py batch_config.csv

batch_config.csv must have these columns (one row per recording):
    telemetry_file   - full path to the Ponemah .csv or .ascii file
    event_file       - full path to the BORIS .csv file
    behaviour        - behaviour name to analyse (must match BORIS label exactly)
    probe_time       - probe reference timestamp  e.g. "15/05/2024 10:46:38 AM"
    video_time       - video reference timestamp  e.g. "15/05/2024 10:46:46 AM"
    bin_size         - bin size in seconds (integer, e.g. 10)

Output goes to <telemetry_file_dir>/extracted_data/<name>_PressureAnalysis/
exactly as if run from the GUI.

Example batch_config.csv:
    telemetry_file,event_file,behaviour,probe_time,video_time,bin_size
    C:/path/to/recording_1_ponemah.csv,C:/path/to/recording_1_boris.csv,Time spent sleeping,15/05/2024 10:46:38 AM,15/05/2024 10:46:46 AM,10
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controllers.pressure_controller import load_data, run_pressure_pipeline

REQUIRED_COLUMNS = {
    "telemetry_file",
    "event_file",
    "behaviour",
    "probe_time",
    "video_time",
    "bin_size",
}


def run_batch(config_path: Path) -> None:
    config = pd.read_csv(config_path)

    missing = REQUIRED_COLUMNS - set(config.columns)
    if missing:
        print(f"ERROR: batch_config.csv is missing columns: {missing}")
        sys.exit(1)

    n_total = len(config)
    n_ok = 0
    n_fail = 0

    for i, row in config.iterrows():
        telemetry_path = Path(str(row["telemetry_file"]).strip())
        event_path = Path(str(row["event_file"]).strip())
        behaviour = str(row["behaviour"]).strip()
        probe_time = str(row["probe_time"]).strip()
        video_time = str(row["video_time"]).strip()
        bin_size = int(row["bin_size"])

        print(f"\n{'='*70}")
        print(f"[{i+1}/{n_total}] {telemetry_path.name}")
        print(f"  Behaviour : {behaviour}")
        print(f"  Probe time: {probe_time}")
        print(f"  Video time: {video_time}")
        print(f"  Bin size  : {bin_size} s")
        print(f"{'='*70}")

        if not telemetry_path.exists():
            print(f"  ERROR: telemetry file not found: {telemetry_path}")
            n_fail += 1
            continue
        if not event_path.exists():
            print(f"  ERROR: event file not found: {event_path}")
            n_fail += 1
            continue

        try:
            telemetry_df, event_df = load_data(telemetry_path, event_path)
            result = run_pressure_pipeline(
                telemetry_df=telemetry_df,
                event_df=event_df,
                behaviour_to_plot=behaviour,
                probe_time=probe_time,
                video_time=video_time,
                bin_size_sec=bin_size,
                output_path=telemetry_path,
                log_callback=lambda msg: print(f"  {msg}"),
            )
            if result:
                print(f"  Done. Output: {result['analysis_folder']}")
                n_ok += 1
            else:
                print("  WARNING: pipeline returned no result (no valid time windows?).")
                n_fail += 1
        except Exception:
            print(f"  ERROR running pipeline for {telemetry_path.name}:")
            traceback.print_exc()
            n_fail += 1

    print(f"\n{'='*70}")
    print(f"Batch complete: {n_ok}/{n_total} succeeded, {n_fail} failed.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run the pressure analysis pipeline without the GUI."
    )
    parser.add_argument(
        "config_csv",
        help="Path to batch_config.csv",
    )
    args = parser.parse_args()

    config_path = Path(args.config_csv)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    run_batch(config_path)


if __name__ == "__main__":
    main()
