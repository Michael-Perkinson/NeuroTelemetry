from __future__ import annotations

import os
import shutil
import unittest
from datetime import datetime
from pathlib import Path


DEFAULT_DATA_ROOT = (
    Path.home()
    / "Documents"
    / "Code"
    / "NeuroTelemetry"
    / "data"
    / "Day 2 (25-04-24) Pro"
)

DATA_ROOT = Path(os.environ.get("PRESSURE_PIPELINE_DATA_ROOT", DEFAULT_DATA_ROOT))
TELEMETRY_PATH = DATA_ROOT / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
EVENT_PATH = DATA_ROOT / "B5 Pro NIGHT 11-01-2025 BORIS.csv"


@unittest.skipUnless(
    TELEMETRY_PATH.exists() and EVENT_PATH.exists(),
    "Sample pressure dataset not available.",
)
class TestPressurePipeline(unittest.TestCase):
    def test_pressure_pipeline_runs_on_sample_dataset(self) -> None:
        from src.controllers.pressure_controller import load_data, run_pressure_pipeline

        telemetry_df, event_df = load_data(TELEMETRY_PATH, EVENT_PATH)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path("test_outputs") / f"pressure_run_{timestamp}"

        try:
            result = run_pressure_pipeline(
                telemetry_df=telemetry_df,
                event_df=event_df,
                behaviour_to_plot="Time spent sleeping",
                probe_time="01/11/2025 05:05:09 PM",
                video_time="01/11/2025 04:59:59 PM",
                bin_size_sec=10,
                output_path=output_path,
                log_callback=lambda _msg: None,
            )
            self.assertIsNotNone(result)
        finally:
            shutil.rmtree(output_path.parent / "extracted_data", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
