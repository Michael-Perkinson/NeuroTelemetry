from datetime import datetime

import pandas as pd
import pytest

from src.core.data_alignment import (
    align_and_clean_datetime,
    build_output_frames,
    extract_and_process_data,
)


def _temp_telemetry(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build the one-column raw dataframe returned by retrieve_telemetry_data."""
    return pd.DataFrame(
        {
            0: [
                "# Col 2:,Temp,x,x,Rate: 1",
                "Time,Temp",
                *(f"{timestamp},{value}" for timestamp, value in rows),
            ]
        }
    )


def test_behaviour_alignment_uses_video_start_and_first_valid_sample_directly() -> None:
    telemetry = _temp_telemetry(
        [
            ("01/01/2025 10:04:59 AM", "x"),
            ("01/01/2025 10:05:00 AM", "36.0"),
            ("01/01/2025 10:05:01 AM", "36.1"),
        ]
    )

    processed = extract_and_process_data(
        telemetry,
        {"sleep": [(1, 360.0, 400.0, 40.0)]},
        alignment_date_time="01/01/2025 10:00:00 AM",
    )

    # Video elapsed 360 s is 10:06 absolute, hence 60 s after telemetry zero.
    assert processed["Behaviours"] == {"sleep": [(1, 60.0, 100.0, 40.0)]}
    assert processed["ReferenceTimestamp"] == datetime(2025, 1, 1, 10, 5)


def test_behaviour_alignment_accepts_video_minutes_after_first_valid_sample() -> None:
    telemetry = _temp_telemetry(
        [
            ("25/04/2024 10:02:08.000 AM", "x"),
            ("25/04/2024 10:05:18.428 AM", "36.0"),
            ("25/04/2024 10:05:19.428 AM", "36.1"),
        ]
    )

    processed = extract_and_process_data(
        telemetry,
        {"sleep": [(1, 0.0, 40.0, 40.0)]},
        alignment_date_time="25/04/2024 10:07:06 AM",
    )

    event = processed["Behaviours"]["sleep"][0]
    assert event[1] == pytest.approx(107.572)
    assert event[2] == pytest.approx(147.572)
    assert processed["FirstValidTimestamp"] == datetime(2024, 4, 25, 10, 5, 18, 428000)


def test_alignment_origin_keeps_absolute_offset_when_reference_precedes_data() -> None:
    telemetry = _temp_telemetry(
        [
            ("01/01/2025 10:05:00 AM", "36.0"),
            ("01/01/2025 10:05:01 AM", "36.1"),
        ]
    )

    processed = extract_and_process_data(
        telemetry,
        behaviour_data=None,
        alignment_date_time="01/01/2025 10:00:00 AM",
        timeline_origin="alignment",
    )

    assert processed["Temp"]["TimeSinceReference"].tolist() == [300.0, 301.0]
    assert processed["ReferenceTimestamp"] == datetime(2025, 1, 1, 10, 0)


def test_alignment_origin_uses_video_zero_for_telemetry_and_behaviour() -> None:
    telemetry = _temp_telemetry(
        [
            ("01/01/2025 10:05:00 AM", "36.0"),
            ("01/01/2025 10:05:01 AM", "36.1"),
        ]
    )

    processed = extract_and_process_data(
        telemetry,
        {"sleep": [(1, 280.0, 320.0, 40.0)]},
        alignment_date_time="01/01/2025 10:00:00 AM",
        timeline_origin="alignment",
    )

    assert processed["Temp"]["TimeSinceReference"].tolist() == [300.0, 301.0]
    assert processed["Behaviours"] == {"sleep": [(1, 280.0, 320.0, 40.0)]}


def test_alignment_origin_keeps_negative_time_and_trims_leading_nan() -> None:
    telemetry = _temp_telemetry(
        [
            ("01/01/2025 09:59:58 AM", "x"),
            ("01/01/2025 09:59:59 AM", "36.0"),
            ("01/01/2025 10:00:00 AM", "36.1"),
        ]
    )

    processed = extract_and_process_data(
        telemetry,
        behaviour_data=None,
        alignment_date_time="01/01/2025 10:00:00 AM",
        timeline_origin="alignment",
    )

    assert processed["Temp"]["TimeSinceReference"].tolist() == [-1.0, 0.0]
    assert processed["Temp"]["Temp"].tolist() == [36.0, 36.1]
    assert processed["ReferenceTimestamp"] == datetime(2025, 1, 1, 10, 0)


def test_extract_rejects_unknown_timeline_origin() -> None:
    telemetry = _temp_telemetry([("01/01/2025 10:00:00 AM", "36.0")])

    with pytest.raises(ValueError, match="timeline_origin"):
        extract_and_process_data(
            telemetry,
            behaviour_data=None,
            alignment_date_time="01/01/2025 10:00:00 AM",
            timeline_origin="probe",  # type: ignore[arg-type]
        )


def test_build_output_frames_supports_temp_and_activity_without_pressure() -> None:
    numerical_data = pd.DataFrame(
        {
            "TimeSinceReference": [0.0, 1.0, 2.0],
            "Temp": [36.0, 36.1, 36.2],
            "Activity": [0.0, 1.0, 0.0],
        }
    )

    processed = build_output_frames(
        numerical_data,
        sample_rates={"Temp": 1.0, "Activity": 1.0},
    )

    assert set(processed) == {"Temp", "Activity"}
    assert processed["Temp"].to_dict("list") == {
        "TimeSinceReference": [0.0, 1.0, 2.0],
        "Temp": [36.0, 36.1, 36.2],
    }
    assert processed["Activity"]["Activity"].tolist() == [0.0, 1.0, 0.0]


def test_no_pressure_frames_preserve_each_optional_signal_time_axis() -> None:
    numerical_data = pd.DataFrame(
        {
            "TimeSinceReference": [0.0, 1.0, 2.0],
            "Temp": [36.0, float("nan"), float("nan")],
            "Activity": [0.0, 1.0, 2.0],
        }
    )

    processed = build_output_frames(
        numerical_data,
        sample_rates={"Temp": 1.0, "Activity": 1.0},
    )

    assert processed["Temp"]["TimeSinceReference"].tolist() == [0.0]
    assert processed["Activity"]["TimeSinceReference"].tolist() == [0.0, 1.0, 2.0]


def test_datetime_alignment_skips_present_but_empty_anchor_channels() -> None:
    numerical_data = pd.DataFrame(
        {
            "DateTime": pd.date_range("2025-01-01", periods=3, freq="s"),
            "Pressure": [float("nan")] * 3,
            "Activity": [0.0, 1.0, 2.0],
        }
    )

    cleaned, reference = align_and_clean_datetime(numerical_data)

    assert len(cleaned) == 3
    assert reference == datetime(2025, 1, 1)
