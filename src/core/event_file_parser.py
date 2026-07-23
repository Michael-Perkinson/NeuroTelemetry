import math
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "event": "behavior",
    "start": "start (s)",
    "stop": "stop (s)",
    "duration": "duration (s)",
}

MIN_DURATION = 30  # seconds
MAX_BEHAVIOUR_TIME_SECONDS = 90 * 60


def read_and_process_event_file(event_file_path: Path) -> pd.DataFrame:
    """Read and standardize a behavior event CSV (case-insensitive)."""
    df = pd.read_csv(event_file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    column_map = {}
    for key, match_str in REQUIRED_COLUMNS.items():
        match_str_lower = match_str.lower()
        matches = [col for col in df.columns if match_str_lower in col]
        if not matches:
            raise ValueError(f"Missing column with '{match_str}' (case-insensitive)")
        column_map[key] = matches[0]

    df["event"] = df[column_map["event"]]
    df["instance"] = df.groupby("event").cumcount() + 1
    df["start"] = pd.to_numeric(df[column_map["start"]], errors="coerce")
    df["end"] = pd.to_numeric(df[column_map["stop"]], errors="coerce")
    df["duration"] = pd.to_numeric(df[column_map["duration"]], errors="coerce")
    df = df.sort_values(["event", "instance"]).reset_index(drop=True)

    return df[["event", "instance", "start", "end", "duration"]]


def select_time_windows(
    behaviour_to_plot: str,
    behaviour_data: dict[str, list[tuple[int, float, float, float]]],
    reference_timestamp: pd.Timestamp,
    telemetry_bounds: tuple[float, float] | None = None,
    telemetry_intervals: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Return fully covered, eligible windows for a specific behavior."""
    del reference_timestamp  # retained for compatibility with existing callers
    coverage = classify_behaviour_windows(
        behaviour_to_plot,
        behaviour_data,
        telemetry_bounds=telemetry_bounds,
        telemetry_intervals=telemetry_intervals,
    )
    return [
        (record["start"], record["end"])
        for record in coverage
        if record["status"] == "fully_covered"
    ]


def classify_behaviour_windows(
    behaviour_to_plot: str,
    behaviour_data: dict[str, list[tuple[int, float, float, float]]],
    telemetry_bounds: tuple[float, float] | None = None,
    telemetry_intervals: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Classify behavior windows against telemetry coverage on video time."""
    instances = behaviour_data.get(behaviour_to_plot, [])
    if telemetry_bounds is None:
        telemetry_start, telemetry_end = float("-inf"), float("inf")
    else:
        telemetry_start, telemetry_end = map(float, telemetry_bounds)

    if telemetry_intervals is None:
        intervals = [(telemetry_start, telemetry_end)]
    else:
        intervals = [
            (float(interval_start), float(interval_end))
            for interval_start, interval_end in telemetry_intervals
        ]

    def finite_float(value: object) -> float | None:
        try:
            converted = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    records: list[dict[str, Any]] = []
    for instance, start, end, duration in instances:
        start_value = finite_float(start)
        end_value = finite_float(end)
        duration_value = finite_float(duration)
        record: dict[str, Any] = {
            "instance": int(instance),
            "start": start_value,
            "end": end_value,
            "duration": duration_value,
        }

        if start_value is None or end_value is None:
            status, reason = "filtered", "missing start or end"
        elif duration_value is None or duration_value < MIN_DURATION:
            status, reason = "filtered", f"duration below {MIN_DURATION}s"
        elif start_value > end_value:
            status, reason = "filtered", "start is after end"
        elif (
            start_value > MAX_BEHAVIOUR_TIME_SECONDS
            or end_value > MAX_BEHAVIOUR_TIME_SECONDS
        ):
            status, reason = "filtered", "outside the 90-minute analysis limit"
        elif any(
            interval_start <= start_value and end_value <= interval_end
            for interval_start, interval_end in intervals
        ):
            status, reason = "fully_covered", "fully covered by telemetry"
        elif any(
            end_value >= interval_start and start_value <= interval_end
            for interval_start, interval_end in intervals
        ):
            status, reason = "partially_covered", "crosses a telemetry boundary or gap"
        else:
            status, reason = "unavailable", "outside continuous telemetry coverage"

        record["status"] = status
        record["reason"] = reason
        records.append(record)

    return records


def structure_behaviour_events(
    event_df: pd.DataFrame,
) -> dict[str, list[tuple[int, float, float, float]]]:
    """Convert the dataframe of events into a dictionary grouped by event name."""
    return {
        str(event): list(
            group[["instance", "start", "end", "duration"]].itertuples(
                index=False, name=None
            )
        )
        for event, group in event_df.groupby("event")
    }
