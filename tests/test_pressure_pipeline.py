from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
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


if __name__ == "__main__":
    if not TELEMETRY_PATH.exists() or not EVENT_PATH.exists():
        raise SystemExit(f"Sample pressure dataset not available under {DATA_ROOT}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manual_output_path = Path("test_outputs") / f"pressure_run_{timestamp}"
    _run_pipeline(output_path=manual_output_path, log_callback=print)
