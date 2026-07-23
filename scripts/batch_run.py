"""
Batch runner for the pressure analysis pipeline - no GUI required.

Usage:
    python scripts/batch_run.py batch_config.xlsx --output-dir C:/path/to/results

The batch config may be .xlsx, .xlsm, or .csv and must have these columns
(one row per recording):
    telemetry_file   - full path to the Ponemah .csv or .ascii file
    event_file       - full path to the BORIS .csv file
    behaviour        - behaviour name to analyse (must match BORIS label exactly)
    video_time       - absolute time at video elapsed time zero,
                       e.g. "15/05/2024 10:46:46 AM"
    bin_size         - bin size in seconds (integer, e.g. 10)

By default, output goes beside each telemetry file. Use --output-dir to place
all per-recording analysis folders under one chosen directory.

Example batch_config.xlsx columns/row:
    telemetry_file,event_file,behaviour,video_time,bin_size
    telemetry.csv,events.csv,Sleep,15/05/2024 10:46:46 AM,10
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from numbers import Real
from pathlib import Path

import pandas as pd
from openpyxl.utils.datetime import from_excel

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controllers.pressure_controller import load_data, run_pressure_pipeline

REQUIRED_COLUMNS = {
    "telemetry_file",
    "event_file",
    "behaviour",
    "video_time",
    "bin_size",
}

VIDEO_TIME_FORMAT = "%d/%m/%Y %I:%M:%S %p"
ACCEPTED_TEXT_FORMATS = (
    VIDEO_TIME_FORMAT,
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
)


def read_batch_config(config_path: Path) -> pd.DataFrame:
    """Load a CSV or native Excel batch configuration."""
    suffix = config_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(config_path)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(config_path, engine="openpyxl")
    raise ValueError("Batch config must be a .csv, .xlsx, or .xlsm file.")


def normalize_video_time(value: object) -> str:
    """Normalize supported Excel/text timestamps to the pipeline format."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        raise ValueError("video_time is blank")

    parsed: datetime
    if isinstance(value, pd.Timestamp):
        parsed = value.to_pydatetime()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, Real) and not isinstance(value, bool):
        converted = from_excel(float(value))
        if not isinstance(converted, datetime):
            raise ValueError("Excel serial video_time does not contain a date")
        parsed = converted
    else:
        raw = str(value).strip()
        for date_format in ACCEPTED_TEXT_FORMATS:
            try:
                parsed = datetime.strptime(raw, date_format)
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                "video_time must include exact seconds, for example "
                "25/04/2024 10:07:06 AM or 2024-04-25T10:07:06"
            )

    return parsed.strftime(VIDEO_TIME_FORMAT)


def run_batch(config_path: Path, output_dir: Path | None = None) -> bool:
    config = read_batch_config(config_path)

    missing = REQUIRED_COLUMNS - set(config.columns)
    if missing:
        print(f"ERROR: batch config is missing columns: {missing}")
        return False

    if "probe_time" in config.columns:
        print(
            "WARNING: probe_time is deprecated and ignored; telemetry timestamps "
            "define the telemetry reference."
        )

    destinations: dict[str, list[int]] = {}
    for row_number, raw_path in enumerate(config["telemetry_file"], start=1):
        telemetry_path = Path(str(raw_path).strip())
        destination_root = output_dir or telemetry_path.parent / "extracted_data"
        destination = destination_root / f"{telemetry_path.stem}_PressureAnalysis"
        destinations.setdefault(str(destination.resolve()).casefold(), []).append(
            row_number
        )
    collisions = [rows for rows in destinations.values() if len(rows) > 1]
    if collisions:
        rows_label = "; ".join(
            ", ".join(str(row) for row in rows) for rows in collisions
        )
        print(
            "ERROR: batch rows would write to the same analysis folder "
            f"(config rows: {rows_label}). Rename those telemetry files or "
            "use separate batch runs/output directories."
        )
        return False

    n_total = len(config)
    n_ok = 0
    n_fail = 0

    for i, row in config.iterrows():
        try:
            telemetry_path = Path(str(row["telemetry_file"]).strip())
            event_path = Path(str(row["event_file"]).strip())
            behaviour = str(row["behaviour"]).strip()
            video_time = normalize_video_time(row["video_time"])
            bin_size = int(row["bin_size"])
        except (TypeError, ValueError) as exc:
            print(f"[{i + 1}/{n_total}] CONFIG ERROR: {exc}")
            n_fail += 1
            continue

        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{n_total}] {telemetry_path.name}")
        print(f"  Behaviour : {behaviour}")
        print(f"  Video time: {video_time}")
        print(f"  Bin size  : {bin_size} s")
        print(f"{'=' * 70}")

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
                video_time=video_time,
                bin_size_sec=bin_size,
                output_path=telemetry_path,
                analysis_root=output_dir,
                event_path=event_path,
                log_callback=lambda msg: print(f"  {msg}"),
            )
            if result:
                print(f"  Done. Output: {result['analysis_folder']}")
                n_ok += 1
            else:
                print(
                    "  WARNING: pipeline returned no result (no valid time windows?)."
                )
                n_fail += 1
        except Exception:
            print(f"  ERROR running pipeline for {telemetry_path.name}:")
            traceback.print_exc()
            n_fail += 1

    print(f"\n{'=' * 70}")
    print(f"Batch complete: {n_ok}/{n_total} succeeded, {n_fail} failed.")
    return n_fail == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run the pressure analysis pipeline without the GUI."
    )
    parser.add_argument(
        "config_file",
        help="Path to a .xlsx, .xlsm, or .csv batch config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Place every recording's analysis folder under this directory. "
            "Defaults to extracted_data beside each telemetry file."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    if not run_batch(config_path, output_dir=args.output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
