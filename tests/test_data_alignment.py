from datetime import datetime

import pandas as pd

from src.core.data_alignment import (
    adjust_behaviours,
    align_and_clean_datetime,
    build_output_frames,
    compute_time_offsets,
)


def test_compute_time_offsets_aligns_video_events_to_first_valid_sample() -> None:
    probe = datetime(2025, 1, 1, 10, 5)
    video = datetime(2025, 1, 1, 10, 0)

    video_offset, total_offset = compute_time_offsets(
        video_ref=video,
        probe_ref=probe,
        removed_nan=-600.0,
    )

    assert video_offset == -300.0
    assert total_offset == 300.0
    adjusted = adjust_behaviours({"sleep": [(1, 60.0, 100.0, 40.0)]}, total_offset)
    assert adjusted == {"sleep": [(1, 360.0, 400.0, 40.0)]}


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

    cleaned, reference, removed_offset = align_and_clean_datetime(
        numerical_data,
        probe_reference_timestamp=datetime(2025, 1, 1, 0, 0, 1),
    )

    assert len(cleaned) == 3
    assert reference == datetime(2025, 1, 1)
    assert removed_offset == -1.0
