from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.batch_run import (
    REQUIRED_COLUMNS,
    normalize_video_time,
    read_batch_config,
    run_batch,
)


def test_batch_schema_no_longer_requires_probe_time() -> None:
    assert "video_time" in REQUIRED_COLUMNS
    assert "probe_time" not in REQUIRED_COLUMNS


@pytest.mark.parametrize("include_legacy_probe", [False, True])
def test_batch_accepts_new_and_legacy_extra_probe_columns(
    local_tmpdir: Path,
    capsys: pytest.CaptureFixture[str],
    include_legacy_probe: bool,
) -> None:
    telemetry_path = local_tmpdir / "telemetry.csv"
    event_path = local_tmpdir / "events.csv"
    telemetry_path.touch()
    event_path.touch()
    config_path = local_tmpdir / "batch.csv"

    row = {
        "telemetry_file": str(telemetry_path),
        "event_file": str(event_path),
        "behaviour": "Sleep",
        "video_time": "15/05/2024 10:46:46 AM",
        "bin_size": 10,
    }
    if include_legacy_probe:
        row["probe_time"] = "15/05/2024 10:46:38 AM"
    pd.DataFrame([row]).to_csv(config_path, index=False)

    with (
        patch("scripts.batch_run.load_data", return_value=(object(), object())),
        patch(
            "scripts.batch_run.run_pressure_pipeline",
            return_value={"analysis_folder": local_tmpdir / "output"},
        ) as run_pipeline,
    ):
        run_batch(config_path)

    assert run_pipeline.call_args.kwargs["video_time"] == row["video_time"]
    assert run_pipeline.call_args.kwargs["analysis_root"] is None
    assert "probe_time" not in run_pipeline.call_args.kwargs
    output = capsys.readouterr().out
    if include_legacy_probe:
        assert "probe_time is deprecated and ignored" in output
    else:
        assert "deprecated" not in output


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            datetime(2024, 4, 25, 10, 7, 6),
            "25/04/2024 10:07:06 AM",
        ),
        ("25/04/2024 10:07:06", "25/04/2024 10:07:06 AM"),
        ("2024-04-25T10:07:06", "25/04/2024 10:07:06 AM"),
    ],
)
def test_normalize_video_time_accepts_excel_and_unambiguous_text(
    value: object, expected: str
) -> None:
    assert normalize_video_time(value) == expected


def test_normalize_video_time_rejects_text_without_seconds() -> None:
    with pytest.raises(ValueError, match="include exact seconds"):
        normalize_video_time("25/04/2024 10:07 AM")


def test_batch_reads_native_excel_datetime_and_uses_chosen_output_root(
    local_tmpdir: Path,
) -> None:
    telemetry_path = local_tmpdir / "telemetry.csv"
    event_path = local_tmpdir / "events.csv"
    telemetry_path.touch()
    event_path.touch()
    output_root = local_tmpdir / "chosen-results"
    config_path = local_tmpdir / "batch.xlsx"
    pd.DataFrame(
        [
            {
                "telemetry_file": str(telemetry_path),
                "event_file": str(event_path),
                "behaviour": "Sleep",
                "video_time": datetime(2024, 4, 25, 10, 7, 6),
                "bin_size": 10,
            }
        ]
    ).to_excel(config_path, index=False)

    loaded = read_batch_config(config_path)
    assert isinstance(loaded.loc[0, "video_time"], pd.Timestamp)

    with (
        patch("scripts.batch_run.load_data", return_value=(object(), object())),
        patch(
            "scripts.batch_run.run_pressure_pipeline",
            return_value={
                "analysis_folder": output_root / "telemetry_PressureAnalysis"
            },
        ) as run_pipeline,
    ):
        run_batch(config_path, output_dir=output_root)

    assert run_pipeline.call_args.kwargs["video_time"] == ("25/04/2024 10:07:06 AM")
    assert run_pipeline.call_args.kwargs["analysis_root"] == output_root


def test_bad_video_time_fails_only_its_batch_row(
    local_tmpdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = local_tmpdir / "batch.csv"
    pd.DataFrame(
        [
            {
                "telemetry_file": "unused.csv",
                "event_file": "unused-events.csv",
                "behaviour": "Sleep",
                "video_time": "25/04/2024 10:07 AM",
                "bin_size": 10,
            }
        ]
    ).to_csv(config_path, index=False)

    with patch("scripts.batch_run.run_pressure_pipeline") as run_pipeline:
        run_batch(config_path)

    run_pipeline.assert_not_called()
    assert "CONFIG ERROR" in capsys.readouterr().out


def test_shared_output_rejects_duplicate_telemetry_stems(
    local_tmpdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = local_tmpdir / "batch.csv"
    pd.DataFrame(
        [
            {
                "telemetry_file": "animal-a/session.csv",
                "event_file": "animal-a/events.csv",
                "behaviour": "Sleep",
                "video_time": "25/04/2024 10:07:06 AM",
                "bin_size": 10,
            },
            {
                "telemetry_file": "animal-b/session.csv",
                "event_file": "animal-b/events.csv",
                "behaviour": "Sleep",
                "video_time": "25/04/2024 10:07:06 AM",
                "bin_size": 10,
            },
        ]
    ).to_csv(config_path, index=False)

    with patch("scripts.batch_run.run_pressure_pipeline") as run_pipeline:
        success = run_batch(config_path, output_dir=local_tmpdir / "results")

    run_pipeline.assert_not_called()
    assert success is False
    assert "same analysis folder" in capsys.readouterr().out
