from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.core.photometry.photometry_peaks import analyse_photometry_peaks, bin_peaks


class TestPhotometryPeaks(unittest.TestCase):
    def test_analyse_photometry_peaks_returns_empty_summary_when_no_peaks(self) -> None:
        photometry = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 1.0, 2.0, 3.0],
                "dFoF_465": [0.0, 0.0, 0.0, 0.0],
            }
        )

        per_peak, peak_times, summary = analyse_photometry_peaks(photometry)

        self.assertTrue(per_peak.empty)
        self.assertEqual(peak_times, [])
        self.assertEqual(summary, {"n_peaks": 0})

    def test_analyse_photometry_peaks_detects_peak_metrics(self) -> None:
        photometry = pd.DataFrame(
            {
                "TimeSinceReference": np.arange(7, dtype=float),
                "dFoF_465": [0.0, 0.2, 0.0, 0.4, 0.0, 0.3, 0.0],
            }
        )

        per_peak, peak_times, summary = analyse_photometry_peaks(
            photometry,
            prominence=0.05,
            amp_thresh=0.01,
        )

        self.assertEqual(peak_times, [1.0, 3.0, 5.0])
        self.assertEqual(per_peak["PeakCount"].tolist(), [1, 2, 3])
        self.assertEqual(summary["n_peaks"], 3)
        self.assertAlmostEqual(float(summary["mean_peak_interval"]), 2.0)

    def test_analyse_photometry_peaks_requires_time_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "TimeSinceReference"):
            analyse_photometry_peaks(pd.DataFrame({"dFoF_465": [0.0, 1.0]}))

    def test_analyse_photometry_peaks_resolves_whitespace_normalized_column(
        self,
    ) -> None:
        photometry = pd.DataFrame(
            {
                "TimeSinceReference": np.arange(5, dtype=float),
                " dFoF_465 ": [0.0, 0.2, 0.0, 0.3, 0.0],
            }
        )

        _, peak_times, summary = analyse_photometry_peaks(photometry)

        self.assertEqual(peak_times, [1.0, 3.0])
        self.assertEqual(summary["n_peaks"], 2)

    def test_bin_peaks_returns_schema_for_empty_input(self) -> None:
        binned = bin_peaks(pd.DataFrame(), np.array([0.0, 60.0]), injection_sec=0.0)

        self.assertEqual(
            binned.columns.tolist(),
            [
                "BinStart",
                "BinEnd",
                "PeakCount",
                "PeaksPerMinute",
                "MeanAmp",
                "SEMAmp",
                "MeanWidth",
                "SEMWidth",
                "MeanISI",
                "SEMISI",
            ],
        )
        self.assertTrue(binned.empty)

    def test_bin_peaks_counts_peaks_and_rates_per_bin(self) -> None:
        per_peak = pd.DataFrame(
            {
                "PeakTime": [90.0, 100.0, 130.0],
                "Amplitude": [1.0, 3.0, 5.0],
                "PeakInterval": [10.0, 30.0, np.nan],
                "Width": [2.0, 4.0, 6.0],
            }
        )
        bin_edges = np.array([-30.0, 0.0, 30.0])

        binned = bin_peaks(per_peak, bin_edges, injection_sec=120.0)

        self.assertEqual(binned["PeakCount"].tolist(), [2, 1])
        self.assertEqual(binned["PeaksPerMinute"].tolist(), [4.0, 2.0])
        self.assertAlmostEqual(float(binned.loc[0, "MeanAmp"]), 2.0)
        self.assertAlmostEqual(float(binned.loc[0, "MeanWidth"]), 3.0)

    def test_bin_peaks_retains_empty_bins_and_excludes_cross_bin_isi(self) -> None:
        per_peak = pd.DataFrame(
            {
                "PeakTime": [10.0, 20.0, 130.0, 140.0],
                "Amplitude": [1.0, 2.0, 3.0, 4.0],
                "PeakInterval": [10.0, 110.0, 10.0, np.nan],
                "Width": [1.0, 2.0, 3.0, 4.0],
            }
        )

        binned = bin_peaks(
            per_peak,
            np.array([0.0, 60.0, 120.0, 180.0]),
            injection_sec=0.0,
        )

        self.assertEqual(binned["PeakCount"].tolist(), [2, 0, 2])
        self.assertEqual(binned["MeanISI"].iloc[[0, 2]].tolist(), [10.0, 10.0])
        self.assertTrue(np.isnan(binned.loc[1, "MeanISI"]))


if __name__ == "__main__":
    unittest.main()
