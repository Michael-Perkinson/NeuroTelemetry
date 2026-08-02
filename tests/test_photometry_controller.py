from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from src.controllers.photometry_controller import (
    load_photometry_data,
    run_photometry_pipeline,
)

DATA_ROOT = Path("data") / "Andys"
PHOTOMETRY_PATH = DATA_ROOT / "2025-07-16 VP3 MET DIESTRUS SALINE_Data OPTION 2 INC.csv"
TELEMETRY_PATH = DATA_ROOT / "2025-07-16 VP3 Saline TEMP and ACT ascii.csv"


def _run_pipeline(output_path: Path, log_callback) -> dict | None:
    photometry_df, telemetry_df = load_photometry_data(
        PHOTOMETRY_PATH,
        TELEMETRY_PATH,
    )

    return run_photometry_pipeline(
        telemetry_df=telemetry_df,
        photometry_df=photometry_df,
        photometry_start_time="16/07/2025 11:08:00 AM",
        injection_sec=2820,
        pre_min=30,
        post_min=360,
        bin_min=30,
        telemetry_path=output_path,
        log_callback=log_callback,
    )


@pytest.mark.skipif(
    os.getenv("RUN_PIPELINE_SMOKE") != "1"
    or not PHOTOMETRY_PATH.exists()
    or not TELEMETRY_PATH.exists(),
    reason="Set RUN_PIPELINE_SMOKE=1 and provide sample photometry data to run.",
)
def test_photometry_pipeline_smoke(local_tmpdir: Path) -> None:
    output_path = local_tmpdir / "photometry"
    results = _run_pipeline(output_path=output_path, log_callback=lambda _msg: None)

    assert results is not None
    assert "summary" in results
    assert "signal_binned" in results
    assert "peaks_binned" in results
    assert results["analysis_folder"].exists()


if __name__ == "__main__":
    if not PHOTOMETRY_PATH.exists() or not TELEMETRY_PATH.exists():
        raise SystemExit(f"Sample photometry dataset not available under {DATA_ROOT}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manual_output_path = Path("test_outputs") / f"photometry_run_{timestamp}"
    _run_pipeline(output_path=manual_output_path, log_callback=print)
