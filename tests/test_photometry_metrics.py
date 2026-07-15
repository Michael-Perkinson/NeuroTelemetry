from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.core.photometry_metrics import (
    bin_signal,
    combine_signal_bins,
    make_bin_edges,
    trim_to_window,
)


class TestPhotometryMetrics(unittest.TestCase):
    def test_trim_to_window_is_inclusive(self) -> None:
        df = pd.DataFrame({"TimeSinceReference": [0.0, 1.0, 2.0, 3.0]})

        trimmed = trim_to_window(df, 1.0, 2.0)

        assert trimmed is not None
        self.assertEqual(trimmed["TimeSinceReference"].tolist(), [1.0, 2.0])

    def test_trim_to_window_returns_none_for_none_input(self) -> None:
        self.assertIsNone(trim_to_window(None, 1.0, 2.0))

    def test_make_bin_edges_returns_seconds_relative_to_injection(self) -> None:
        np.testing.assert_array_equal(make_bin_edges(1, 2, 1), [-60, 0, 60, 120])

    def test_bin_signal_calculates_mean_and_sem_per_relative_bin(self) -> None:
        signal_df = pd.DataFrame(
            {
                "TimeSinceReference": [90.0, 100.0, 110.0, 130.0, 150.0],
                "dFoF_465": [1.0, 3.0, 5.0, 7.0, 9.0],
            }
        )
        bin_edges = np.array([-30.0, 0.0, 30.0])

        binned = bin_signal(signal_df, bin_edges, "dFoF_465", injection_sec=120.0)

        self.assertEqual(binned["BinStart"].tolist(), [-30.0, 0.0])
        self.assertEqual(binned["BinEnd"].tolist(), [0.0, 30.0])
        self.assertAlmostEqual(float(binned.loc[0, "Mean"]), 3.0)
        self.assertAlmostEqual(float(binned.loc[0, "SEM"]), 2.0 / np.sqrt(3.0))
        self.assertAlmostEqual(float(binned.loc[1, "Mean"]), 7.0)
        self.assertTrue(np.isnan(binned.loc[1, "SEM"]))

    def test_bin_signal_returns_schema_for_empty_input(self) -> None:
        binned = bin_signal(None, np.array([0.0, 60.0]), "dFoF_465", 0.0)

        self.assertEqual(binned.columns.tolist(), ["BinStart", "BinEnd", "Mean", "SEM"])
        self.assertTrue(binned.empty)

    def test_combine_signal_bins_merges_temperature_and_activity(self) -> None:
        photometry = pd.DataFrame(
            {"BinStart": [0.0], "BinEnd": [60.0], "Mean": [1.5], "SEM": [0.2]}
        )
        temp = pd.DataFrame(
            {"BinStart": [0.0], "BinEnd": [60.0], "Mean": [37.5], "SEM": [0.1]}
        )
        activity = pd.DataFrame(
            {"BinStart": [0.0], "BinEnd": [60.0], "Mean": [12.0], "SEM": [2.0]}
        )

        combined = combine_signal_bins(photometry, temp, activity)

        self.assertEqual(float(combined.loc[0, "dFoF_Mean"]), 1.5)
        self.assertEqual(float(combined.loc[0, "Temp_Mean"]), 37.5)
        self.assertEqual(float(combined.loc[0, "Act_Mean"]), 12.0)

    def test_combine_signal_bins_keeps_missing_optional_columns(self) -> None:
        photometry = pd.DataFrame(
            {"BinStart": [0.0], "BinEnd": [60.0], "Mean": [1.5], "SEM": [0.2]}
        )

        combined = combine_signal_bins(photometry, None, None)

        self.assertEqual(
            combined.columns.tolist(),
            [
                "BinStart",
                "BinEnd",
                "dFoF_Mean",
                "dFoF_SEM",
                "Temp_Mean",
                "Temp_SEM",
                "Act_Mean",
                "Act_SEM",
            ],
        )
        self.assertTrue(np.isnan(combined.loc[0, "Temp_Mean"]))
        self.assertTrue(np.isnan(combined.loc[0, "Act_Mean"]))


if __name__ == "__main__":
    unittest.main()
