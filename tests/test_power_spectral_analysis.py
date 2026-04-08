from __future__ import annotations

import unittest
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

from src.core.power_spectral_analysis import (
    DEFAULT_PSD_NFFT,
    DEFAULT_PSD_OVERLAP_FRACTION,
    DEFAULT_PSD_RESAMPLE_HZ,
    DEFAULT_PSD_SEGMENT_SECONDS,
    DEFAULT_PSD_WELCH_WINDOWS,
    _compute_welch_psd,
    _iter_segment_bounds,
    _resample_segment,
    analyze_ttot_psd,
)


def _make_trace_df(duration_s: float, interval_s: float = 1.0) -> pd.DataFrame:
    times = np.arange(0.0, duration_s + interval_s, interval_s, dtype=float)
    ttot = 1.0 + 0.1 * np.sin(2.0 * np.pi * 0.2 * times)
    return pd.DataFrame({"TimeOfBreath_s": times, "Ttot_s": ttot})


def _make_period(duration_s: float, interval_s: float = 1.0) -> dict[str, object]:
    peak_times = pd.Series(np.arange(0.0, duration_s + interval_s, interval_s))
    return {
        "name": f"Period_0_{duration_s}",
        "start_time": 0.0,
        "end_time": duration_s,
        "peak_times": peak_times,
        "trough_times": peak_times.iloc[:-1],
        "peak_indices": peak_times,
        "trough_indices": peak_times.iloc[:-1],
    }


class TestPowerSpectralAnalysis(unittest.TestCase):
    def _make_local_tempdir(self) -> Path:
        base_dir = Path("test_outputs")
        base_dir.mkdir(exist_ok=True)
        tmpdir = base_dir / f"psd_test_{uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        return tmpdir

    def test_segments_shorter_than_sixty_seconds_are_discarded(self) -> None:
        self.assertEqual(_iter_segment_bounds(0.0, 59.9, 60.0), [])

    def test_longer_periods_split_into_contiguous_sixty_second_segments(self) -> None:
        bounds = _iter_segment_bounds(0.0, 180.0, 60.0)
        self.assertEqual(bounds, [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)])

    def test_resampled_segments_have_expected_length_and_centering(self) -> None:
        trace_df = _make_trace_df(duration_s=180.0, interval_s=1.0)

        resampled = _resample_segment(
            trace_df=trace_df,
            segment_start=0.0,
            segment_seconds=DEFAULT_PSD_SEGMENT_SECONDS,
            resample_hz=DEFAULT_PSD_RESAMPLE_HZ,
        )

        self.assertIsNotNone(resampled)
        assert resampled is not None
        self.assertEqual(
            len(resampled),
            int(DEFAULT_PSD_SEGMENT_SECONDS * DEFAULT_PSD_RESAMPLE_HZ),
        )
        self.assertAlmostEqual(float(resampled["TtotCentered_s"].mean()), 0.0, places=10)
        self.assertGreater(float(resampled["Ttot_s"].std()), 0.0)

    def test_welch_psd_recovers_expected_modulation_frequency(self) -> None:
        time_s = np.arange(
            0.0,
            DEFAULT_PSD_SEGMENT_SECONDS,
            1.0 / DEFAULT_PSD_RESAMPLE_HZ,
            dtype=float,
        )
        modulation_hz = 0.2
        centered_signal = np.sin(2.0 * np.pi * modulation_hz * time_s)

        freq_hz, psd = _compute_welch_psd(
            centered_signal=centered_signal,
            resample_hz=DEFAULT_PSD_RESAMPLE_HZ,
            nfft=DEFAULT_PSD_NFFT,
            welch_windows=DEFAULT_PSD_WELCH_WINDOWS,
            overlap_fraction=DEFAULT_PSD_OVERLAP_FRACTION,
        )

        peak_freq = float(freq_hz[int(np.argmax(psd))])
        self.assertAlmostEqual(peak_freq, modulation_hz, delta=0.03)

    def test_analyze_ttot_psd_exports_expected_outputs(self) -> None:
        window_periods = {
            "0.0-180.0": [_make_period(duration_s=180.0, interval_s=1.0)],
        }

        tmpdir = self._make_local_tempdir()
        try:
            output_folder = tmpdir / "PSD"
            results = analyze_ttot_psd(window_periods=window_periods, output_folder=output_folder)

            self.assertFalse(results["segment_summary"].empty)
            self.assertFalse(results["per_window_psd"].empty)
            self.assertFalse(results["pooled_psd"].empty)
            self.assertFalse(results["resampled_traces"].empty)

            expected_files = {
                "ttot_psd_segment_summary.csv",
                "ttot_psd_per_segment.csv",
                "ttot_psd_per_window.csv",
                "ttot_psd_pooled.csv",
                "ttot_resampled_segments.csv",
            }
            self.assertTrue(expected_files.issubset({p.name for p in output_folder.iterdir()}))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_analyze_ttot_psd_discards_periods_shorter_than_sixty_seconds(self) -> None:
        window_periods = {
            "0.0-40.0": [_make_period(duration_s=40.0, interval_s=1.0)],
        }

        tmpdir = self._make_local_tempdir()
        try:
            results = analyze_ttot_psd(
                window_periods=window_periods,
                output_folder=tmpdir / "PSD",
            )

            self.assertTrue(results["segment_summary"].empty)
            self.assertTrue(results["per_window_psd"].empty)
            self.assertTrue(results["pooled_psd"].empty)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
