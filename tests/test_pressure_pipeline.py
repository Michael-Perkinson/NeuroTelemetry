from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.controllers.pressure_controller import load_data, run_pressure_pipeline

DATA_ROOT = Path("data") / "Day 2 (25-04-24) Pro"
TELEMETRY_PATH = DATA_ROOT / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
EVENT_PATH = DATA_ROOT / "B5 Pro NIGHT 11-01-2025 BORIS.csv"


def _run_pipeline(output_path: Path, log_callback) -> dict | None:
    telemetry_df, event_df = load_data(TELEMETRY_PATH, EVENT_PATH)

    return run_pressure_pipeline(
        telemetry_df=telemetry_df,
        event_df=event_df,
        behaviour_to_plot="Time spent sleeping",
        video_time="01/11/2025 04:59:59 PM",
        bin_size_sec=10,
        output_path=output_path,
        log_callback=log_callback,
    )


@pytest.mark.skipif(
    os.getenv("RUN_PIPELINE_SMOKE") != "1"
    or not TELEMETRY_PATH.exists()
    or not EVENT_PATH.exists(),
    reason="Set RUN_PIPELINE_SMOKE=1 and provide sample pressure data to run.",
)
def test_pressure_pipeline_smoke(local_tmpdir: Path) -> None:
    output_path = local_tmpdir / f"pressure_run_{uuid4().hex}"
    results = _run_pipeline(output_path=output_path, log_callback=lambda _msg: None)

    assert results is not None
    assert "summary" in results
    assert "metrics" in results
    assert "psd" in results

    analysis_folder = (
        output_path.parent / "extracted_data" / f"{output_path.stem}_PressureAnalysis"
    )
    assert analysis_folder.exists()
    assert results["analysis_folder"] == analysis_folder
    assert results["config_path"].parent == analysis_folder / "logs"
    assert results["config_path"].is_file()


def test_pressure_pipeline_handles_failure_gracefully(local_tmpdir: Path) -> None:
    """Test that pipeline catches exceptions and marks status as failed."""
    import pandas as pd

    from src.controllers.pressure_controller import run_pressure_pipeline

    output_path = local_tmpdir / f"pressure_run_{uuid4().hex}"

    # Create minimal mock dataframes
    telemetry_df = pd.DataFrame({
        "Timestamp": [0.0, 1.0, 2.0],
        "Pressure": [100.0, 101.0, 99.0],
    })
    event_df = pd.DataFrame({
        "Behavior": ["Time spent sleeping", "Time spent sleeping"],
        "StartTime": [0.0, 1.5],
        "EndTime": [1.0, 2.5],
    })

    # Patch structure_behaviour_events to raise an exception early in the pipeline
    with patch(
        "src.controllers.pressure_controller.structure_behaviour_events"
    ) as mock_struct:
        mock_struct.side_effect = RuntimeError("Simulated behavior structure failure")

        # The exception should be re-raised
        with pytest.raises(RuntimeError, match="Simulated behavior structure failure"):
            run_pressure_pipeline(
                telemetry_df=telemetry_df,
                event_df=event_df,
                behaviour_to_plot="Time spent sleeping",
                video_time="01/11/2025 04:59:59 PM",
                bin_size_sec=10,
                output_path=output_path,
                log_callback=lambda x: None,
            )

        # But the config file should exist and have status="failed"
        analysis_folder = (
            output_path.parent
            / "extracted_data"
            / f"{output_path.stem}_PressureAnalysis"
        )
        logs_folder = analysis_folder / "logs"
        assert logs_folder.exists()

        # Find the config file (should be analysis_config_*.json)
        config_files = list(logs_folder.glob("analysis_config_*.json"))
        assert len(config_files) == 1, (
            f"Expected 1 config file, found {len(config_files)}"
        )

        config_path = config_files[0]
        config_data = json.loads(config_path.read_text())

        # Verify the status is marked as failed
        assert config_data["AnalysisStatus"] == "failed"
        assert "ErrorMessage" in config_data
        assert "Simulated behavior structure failure" in config_data["ErrorMessage"]
        assert "FailedAt" in config_data


if __name__ == "__main__":
    if not TELEMETRY_PATH.exists() or not EVENT_PATH.exists():
        raise SystemExit(f"Sample pressure dataset not available under {DATA_ROOT}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manual_output_path = Path("test_outputs") / f"pressure_run_{timestamp}"
    _run_pipeline(output_path=manual_output_path, log_callback=print)
