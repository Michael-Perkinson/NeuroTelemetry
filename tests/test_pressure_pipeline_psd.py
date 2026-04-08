from __future__ import annotations

import unittest
from pathlib import Path
import shutil
from uuid import uuid4

from src.controllers.pressure_controller import load_data, run_pressure_pipeline


DATA_ROOT = Path("c:/Users/permi73p/Documents/Code/NeuroTelemetry/data/B5 night")
TELEMETRY_PATH = DATA_ROOT / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
EVENT_PATH = DATA_ROOT / "B5 Pro NIGHT 11-01-2025 BORIS.csv"


@unittest.skipUnless(
    TELEMETRY_PATH.exists() and EVENT_PATH.exists(),
    "Sample pressure dataset not available.",
)
class TestPressurePipelinePSD(unittest.TestCase):
    def test_pipeline_exports_psd_outputs(self) -> None:
        telemetry_df, event_df = load_data(TELEMETRY_PATH, EVENT_PATH)

        base_dir = Path("test_outputs")
        base_dir.mkdir(exist_ok=True)
        tmpdir = base_dir / f"pressure_psd_{uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            output_path = tmpdir / "pressure_psd_run"
            results = run_pressure_pipeline(
                telemetry_df=telemetry_df,
                event_df=event_df,
                behaviour_to_plot="Time spent sleeping",
                probe_time="01/11/2025 05:05:09 PM",
                video_time="01/11/2025 04:59:59 PM",
                bin_size_sec=10,
                output_path=output_path,
                log_callback=lambda _msg: None,
            )

            self.assertIsNotNone(results)
            assert results is not None
            self.assertIn("psd", results)
            self.assertFalse(results["psd"]["segment_summary"].empty)
            self.assertFalse(results["psd"]["per_window_psd"].empty)
            self.assertFalse(results["psd"]["pooled_psd"].empty)

            analysis_folder = (
                output_path.parent
                / "extracted_data"
                / f"{output_path.stem}_PressureAnalysis"
            )
            psd_folder = analysis_folder / "PSD"
            self.assertTrue(psd_folder.exists())
            self.assertTrue((psd_folder / "ttot_psd_per_window.csv").exists())
            self.assertTrue((psd_folder / "ttot_psd_pooled.csv").exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
