from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from src.core.event_file_parser import (
    classify_behaviour_windows,
    read_and_process_event_file,
    select_time_windows,
    structure_behaviour_events,
)


def test_read_and_process_event_file_standardizes_case_insensitive_columns(
    local_tmpdir: Path,
) -> None:
    event_path = local_tmpdir / "events.csv"
    event_path.write_text(
        "\n".join(
            [
                "Behavior,Start (s),Stop (s),Duration (s)",
                "sleep,10,50,40",
                "eat,60,70,10",
                "sleep,90,130,40",
            ]
        )
    )

    df = read_and_process_event_file(event_path)

    assert df.to_dict("list") == {
        "event": ["eat", "sleep", "sleep"],
        "instance": [1, 1, 2],
        "start": [60, 10, 90],
        "end": [70, 50, 130],
        "duration": [10, 40, 40],
    }


def test_read_and_process_event_file_requires_core_columns(local_tmpdir: Path) -> None:
    event_path = local_tmpdir / "events.csv"
    event_path.write_text("Behavior,Start (s),Duration (s)\nsleep,10,40\n")

    with pytest.raises(ValueError, match="stop"):
        read_and_process_event_file(event_path)


def test_structure_behaviour_events_groups_by_event() -> None:
    event_df = pd.DataFrame(
        {
            "event": ["eat", "sleep", "sleep"],
            "instance": [1, 1, 2],
            "start": [60.0, 10.0, 90.0],
            "end": [70.0, 50.0, 130.0],
            "duration": [10.0, 40.0, 40.0],
        }
    )

    assert structure_behaviour_events(event_df) == {
        "eat": [(1, 60.0, 70.0, 10.0)],
        "sleep": [(1, 10.0, 50.0, 40.0), (2, 90.0, 130.0, 40.0)],
    }


def test_select_time_windows_filters_by_behaviour_duration_and_cutoff() -> None:
    behaviour_data = {
        "sleep": [
            (1, 10.0, 50.0, 40.0),
            (2, 90.0, 100.0, 10.0),
            (3, 5_350.0, 5_400.0, 50.0),
            (4, 5_390.0, 5_410.0, 40.0),
            (5, 6_000.0, 6_050.0, 50.0),
        ],
        "eat": [(1, 30.0, 70.0, 40.0)],
    }

    windows = select_time_windows(
        behaviour_to_plot="sleep",
        behaviour_data=behaviour_data,
        reference_timestamp=pd.Timestamp("2025-01-01T00:00:00"),
    )

    assert windows == [(10.0, 50.0), (5_350.0, 5_400.0)]


def test_select_time_windows_returns_empty_for_missing_behaviour() -> None:
    assert (
        select_time_windows(
            behaviour_to_plot="sleep",
            behaviour_data={},
            reference_timestamp=pd.Timestamp("1970-01-01T00:00:00"),
        )
        == []
    )


def test_classify_behaviour_windows_reports_telemetry_coverage() -> None:
    behaviour_data = {
        "sleep": [
            (1, 0.0, 40.0, 40.0),
            (2, -20.0, 20.0, 40.0),
            (3, 80.0, 120.0, 40.0),
            (4, -100.0, -50.0, 50.0),
            (5, 150.0, 200.0, 50.0),
            (6, 20.0, 25.0, 5.0),
            (7, float("nan"), 40.0, 40.0),
        ]
    }

    coverage = classify_behaviour_windows(
        "sleep", behaviour_data, telemetry_bounds=(-10.0, 100.0)
    )

    assert [record["status"] for record in coverage] == [
        "fully_covered",
        "partially_covered",
        "partially_covered",
        "unavailable",
        "unavailable",
        "filtered",
        "filtered",
    ]
    assert select_time_windows(
        "sleep",
        behaviour_data,
        pd.Timestamp("2025-01-01"),
        telemetry_bounds=(-10.0, 100.0),
    ) == [(0.0, 40.0)]


def test_classify_behaviour_windows_detects_internal_telemetry_gap() -> None:
    behaviour_data = {
        "sleep": [
            (1, 10.0, 40.0, 30.0),
            (2, 80.0, 120.0, 40.0),
            (3, 40.0, 80.0, 40.0),
        ]
    }
    intervals = [(0.0, 50.0), (100.0, 150.0)]

    coverage = classify_behaviour_windows(
        "sleep",
        behaviour_data,
        telemetry_bounds=(0.0, 150.0),
        telemetry_intervals=intervals,
    )

    assert [record["status"] for record in coverage] == [
        "fully_covered",
        "partially_covered",
        "partially_covered",
    ]
    assert select_time_windows(
        "sleep",
        behaviour_data,
        pd.Timestamp("2025-01-01"),
        telemetry_bounds=(0.0, 150.0),
        telemetry_intervals=intervals,
    ) == [(10.0, 40.0)]


def test_classify_behaviour_windows_filters_malformed_values() -> None:
    coverage = classify_behaviour_windows(
        "sleep",
        cast(
            Any,
            {"sleep": [(1, pd.NA, 40.0, 40.0), (2, "bad", 40.0, 40.0)]},
        ),
    )

    assert [record["status"] for record in coverage] == ["filtered", "filtered"]
    assert all(record["start"] is None for record in coverage)
