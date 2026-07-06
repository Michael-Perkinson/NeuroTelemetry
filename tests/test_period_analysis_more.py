from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.period_analysis import (
    compute_respiratory_metrics_for_periods,
    find_valid_periods,
)


class TestFindValidPeriods(unittest.TestCase):
    def test_find_valid_periods_builds_window_period_definitions(self) -> None:
        peak_times = pd.Series(np.arange(60, dtype=float), index=np.arange(60))
        trough_times = pd.Series(np.arange(60, dtype=float), index=np.arange(60))
        result = (
            0.0,
            59.0,
            np.arange(60),
            peak_times,
            np.arange(60),
            trough_times,
        )

        valid_peaks, valid_troughs, periods, window_periods = find_valid_periods(
            {"0.0-59.0": [result]},
            pd.DataFrame(),
            [(0.0, 59.0)],
        )

        self.assertEqual(len(valid_peaks), 60)
        self.assertEqual(len(valid_troughs), 60)
        self.assertEqual(periods, [(0.0, 59.0)])
        self.assertEqual(len(window_periods["0.0-59.0"]), 1)
        self.assertEqual(window_periods["0.0-59.0"][0]["name"], "Period_1_0.0-59.0")

    def test_find_valid_periods_skips_missing_results_and_short_periods(self) -> None:
        valid_peaks, valid_troughs, periods, window_periods = find_valid_periods(
            {},
            pd.DataFrame(),
            [(0.0, 10.0)],
        )

        self.assertEqual(valid_peaks, [])
        self.assertEqual(valid_troughs, [])
        self.assertEqual(periods, [])
        self.assertEqual(window_periods, {})

    def test_find_valid_periods_aligns_extra_peaks_and_troughs(self) -> None:
        peak_times = pd.Series(np.arange(61, dtype=float), index=np.arange(61))
        trough_times = pd.Series(np.arange(60, dtype=float), index=np.arange(60))
        result = (
            0.0,
            60.0,
            np.arange(61),
            peak_times,
            np.arange(60),
            trough_times,
        )

        valid_peaks, valid_troughs, periods, window_periods = find_valid_periods(
            {"0.0-60.0": [result]},
            pd.DataFrame(),
            [(0.0, 60.0)],
        )

        self.assertEqual(len(valid_peaks), 60)
        self.assertEqual(len(valid_troughs), 60)
        self.assertEqual(periods, [(0.0, 60.0)])
        self.assertEqual(len(window_periods["0.0-60.0"][0]["peak_times"]), 60)


class TestComputeRespiratoryMetricsForPeriods(unittest.TestCase):
    def test_compute_respiratory_metrics_for_periods_assembles_all_scopes(self) -> None:
        peaks = pd.Series([0.5, 1.5], index=[1, 3])
        troughs = pd.Series([0.25, 1.25], index=[0, 2])
        window_periods = {
            "0.0-2.0": [
                {
                    "name": "Period_1",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "peak_times": peaks,
                    "trough_times": troughs,
                }
            ]
        }
        pressure = pd.DataFrame({"SmoothedPressure": [10.0, 5.0, 10.0, 5.0]})
        binned = pd.DataFrame({"Bin_Start": [0.0], "T_I_mean": [0.25]})
        summary = {"T_I_mean": 0.25}
        window_summary = {"T_I_mean": 0.5}

        with (
            patch(
                "src.core.period_analysis.calculate_binned_period_metrics",
                return_value=binned,
            ) as calc_binned,
            patch(
                "src.core.period_analysis.calculate_valid_period_metrics",
                return_value=summary,
            ),
            patch(
                "src.core.period_analysis.summarize_respiratory_cycles",
                return_value=window_summary,
            ) as summarize,
        ):
            metrics = compute_respiratory_metrics_for_periods(
                window_periods=window_periods,
                pressure_data=pressure,
                temp_data=None,
                activity_data=None,
                time_windows=[(0.0, 2.0), (10.0, 20.0)],
                bin_size_sec=2,
            )

        self.assertIn("0.0-2.0", metrics)
        self.assertIn("GlobalSummary", metrics)
        pd.testing.assert_frame_equal(
            metrics["0.0-2.0"]["Periods"]["Period_1"]["Binned"],
            binned,
        )
        self.assertEqual(metrics["0.0-2.0"]["Periods"]["Period_1"]["Summary"], summary)
        calc_binned.assert_called_once()
        self.assertEqual(summarize.call_count, 2)


if __name__ == "__main__":
    unittest.main()
