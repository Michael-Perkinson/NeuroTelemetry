"""
Diagnostic script — visualise raw vs filtered pressure signal at each pipeline stage.

Shows: Raw → Highpass → Lowpass → Savitzky-Golay (SmoothedPressure)
Does NOT modify any existing pipeline code.

Usage:
    python scripts/diagnose_filtering.py <telemetry_csv> [--seconds 10] [--start 0]

Example:
    python scripts/diagnose_filtering.py "X:/path/to/Ponemah.csv"
        --seconds 10 --start 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, savgol_filter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_alignment import extract_and_process_data
from src.core.data_file_parser import retrieve_telemetry_data
from src.core.event_file_parser import (
    read_and_process_event_file,
    structure_behaviour_events,
)


def _butter(data, cutoff, fs, btype, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype=btype, analog=False)
    return filtfilt(b, a, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry_csv", type=Path)
    parser.add_argument("event_csv", type=Path)
    parser.add_argument("probe_time", type=str)
    parser.add_argument("video_time", type=str)
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="Window to display (s)"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start offset (s) into the file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save HTML to this path (default: show in browser)",
    )
    args = parser.parse_args()

    print(f"Loading: {args.telemetry_csv}")
    telemetry_df = retrieve_telemetry_data(args.telemetry_csv)
    event_df = read_and_process_event_file(args.event_csv)
    behaviour_data = structure_behaviour_events(event_df)

    processed = extract_and_process_data(
        telemetry_df, behaviour_data, args.probe_time, args.video_time
    )
    pressure_data = processed["Pressure"]

    fs = 500.0  # known from file header
    print(f"Using sample rate: {fs} Hz")

    # Raw pressure (before any filtering)
    raw = pressure_data["Pressure"].to_numpy(dtype=float)

    # Slice to requested window
    i0 = int(args.start * fs)
    i1 = int((args.start + args.seconds) * fs)
    i1 = min(i1, len(raw))
    t = np.arange(i0, i1) / fs

    raw_slice = raw[i0:i1]

    # Reproduce each filter step exactly as the pipeline does
    highpassed = _butter(raw, cutoff=0.02, fs=fs, btype="high")[i0:i1]
    lowpassed = _butter(raw, cutoff=20.0, fs=fs, btype="low")[i0:i1]

    window = min(35, len(raw) - 1)
    if window % 2 == 0:
        window -= 1
    savgol = savgol_filter(
        _butter(raw, cutoff=20.0, fs=fs, btype="low"), window_length=window, polyorder=2
    )[i0:i1]

    # Also grab the actual SmoothedPressure the pipeline produced
    smoothed_actual = pressure_data["SmoothedPressure"].to_numpy(dtype=float)[i0:i1]

    print(f"Savitzky-Golay window: {window} samples = {window / fs * 1000:.1f} ms")
    display_end = args.start + args.seconds
    print(f"Displaying {args.start:.1f}s – {display_end:.1f}s ({i1 - i0} samples)")

    # Build 5-panel figure
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "1. Raw pressure",
            "2. After high-pass (0.02 Hz) — drift removed",
            "3. After low-pass (20 Hz)",
            "4. After Savitzky-Golay "
            f"(window={window} samples = {window / fs * 1000:.0f} ms)",
            "5. Actual SmoothedPressure from pipeline",
        ],
        vertical_spacing=0.06,
    )

    for row, (label, sig) in enumerate(
        [
            ("Raw", raw_slice),
            ("Highpass 0.02 Hz", highpassed),
            ("Lowpass 20 Hz", lowpassed),
            (f"SavGol w={window}", savgol),
            ("SmoothedPressure (pipeline)", smoothed_actual),
        ],
        start=1,
    ):
        fig.add_trace(
            go.Scatter(
                x=t, y=sig, mode="lines", name=label, line=dict(width=1, color="white")
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=(
            f"Filter pipeline diagnosis — {args.telemetry_csv.name} | "
            f"{args.start:.0f}–{display_end:.0f}s"
        ),
        height=1100,
        showlegend=False,
        template="plotly_dark",
    )
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)

    if args.out:
        fig.write_html(str(args.out))
        print(f"Saved: {args.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
