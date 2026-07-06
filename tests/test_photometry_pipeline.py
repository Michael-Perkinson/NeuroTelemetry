from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path


DEFAULT_DATA_ROOT = Path.home() / "data" / "Andys"
DATA_ROOT = Path(os.environ.get("PHOTOMETRY_PIPELINE_DATA_ROOT", DEFAULT_DATA_ROOT))

PHOTOMETRY_PATH = DATA_ROOT / "2025-07-16 VP3 MET DIESTRUS SALINE_Data OPTION 2 INC.csv"
TELEMETRY_PATH = DATA_ROOT / "2025-07-16 VP3 Saline TEMP and ACT ascii.csv"


@unittest.skipUnless(
    PHOTOMETRY_PATH.exists() and TELEMETRY_PATH.exists(),
    "Sample photometry dataset not available.",
)
class TestPhotometryPipeline(unittest.TestCase):
    def test_photometry_pipeline_runs_on_sample_dataset(self) -> None:
        from src.controllers.photometry_controller import (
            load_photometry_data,
            run_photometry_pipeline,
        )

        photometry_df, telemetry_df = load_photometry_data(
            PHOTOMETRY_PATH,
            TELEMETRY_PATH,
        )
        output_path = Path("test_outputs") / "photometry"

        try:
            result = run_photometry_pipeline(
                telemetry_df=telemetry_df,
                photometry_df=photometry_df,
                photometry_align_time="16/07/2025 11:08:00 AM",
                injection_sec=2820,
                pre_min=30,
                post_min=360,
                bin_min=30,
                telemetry_path=output_path,
                log_callback=lambda _msg: None,
            )
            self.assertIsNotNone(result)
        finally:
            shutil.rmtree(output_path.parent / "extracted_data", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
