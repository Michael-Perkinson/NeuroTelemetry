"""
Diagnostic: compare a range of low-pass filter cutoffs on a single recording.
Does NOT modify any pipeline code.

Usage:
    python scripts/compare_lowpass.py <telemetry_csv> <event_csv>
        <video_time> [--seconds 10] [--start 120]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_alignment import extract_and_process_data
from src.core.data_file_parser import retrieve_telemetry_data
from src.core.event_file_parser import (
    read_and_process_event_file,
    structure_behaviour_events,
)

CUTOFFS = [20, 30, 40, 50, 60, 75, 100]


def _lowpass(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data)


def load_pressure(telemetry_csv, event_csv, video_time):
    telemetry_df = retrieve_telemetry_data(Path(telemetry_csv))
    event_df = read_and_process_event_file(Path(event_csv))
    behaviour_data = structure_behaviour_events(event_df)
    processed = extract_and_process_data(
        telemetry_df,
        behaviour_data,
        video_time,
        timeline_origin="first_valid",
    )
    return processed["Pressure"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("event", type=Path)
    parser.add_argument("video", type=str)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--start", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    fs = 500.0
    rows = 1 + len(CUTOFFS)

    print(f"Loading: {args.telemetry.name}")
    pdata = load_pressure(args.telemetry, args.event, args.video)

    raw = pdata["Pressure"].to_numpy(dtype=float)
    i0 = int(args.start * fs)
    i1 = min(int((args.start + args.seconds) * fs), len(raw))
    t = np.arange(i0, i1) / fs

    titles = [f"Raw  ({args.start:.0f}–{args.start + args.seconds:.0f}s)"] + [
        f"Lowpass {c} Hz" for c in CUTOFFS
    ]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.03,
    )

    signals = [raw[i0:i1]] + [_lowpass(raw, c, fs)[i0:i1] for c in CUTOFFS]

    for row, sig in enumerate(signals, start=1):
        fig.add_trace(
            go.Scatter(x=t, y=sig, mode="lines", line=dict(width=1, color="white")),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=f"Lowpass cutoff sweep — {args.telemetry.stem}",
        height=200 * rows,
        showlegend=False,
        template="plotly_dark",
    )

    if args.out:
        fig.write_html(str(args.out))
        print(f"Saved: {args.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
