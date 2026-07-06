from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.core.peak_detection import find_shoulders
from src.core.period_analysis import identify_new_periods
from src.core.respiratory_metrics import (
    calculate_binned_period_metrics,
    calculate_respiratory_metrics,
    calculate_respiratory_metrics_raw,
    compute_atm_pressure_session_summary,
    compute_atm_pressure_time_bins,
    summarize_respiratory_cycles,
)


def _pressure_data() -> pd.DataFrame:
    return pd.DataFrame({"SmoothedPressure": [10.0, 5.0, 10.0, 5.0, 10.0, 5.0]})


def _peak_series() -> pd.Series:
    return pd.Series([1.0, 3.0, 5.0], index=[1, 3, 5])


def _trough_series() -> pd.Series:
    return pd.Series([0.5, 2.5, 4.5], index=[0, 2, 4])


class TestPeakDetection(unittest.TestCase):
    def test_find_shoulders_returns_start_for_empty_segment(self) -> None:
        self.assertEqual(
            find_shoulders(np.array([]), start_index=10, peak_index=20),
            10,
        )

    def test_find_shoulders_uses_last_positive_to_negative_zero_crossing(self) -> None:
        dvdt = np.array([0.2, 0.1, -0.05, -0.5, -1.0, -0.2, 0.1])

        shoulder = find_shoulders(dvdt, start_index=10, peak_index=20)

        self.assertEqual(shoulder, 12)

    def test_find_shoulders_falls_back_to_nearest_zero_before_minimum(self) -> None:
        dvdt = np.array([-0.1, -0.2, -1.0, -0.3])

        shoulder = find_shoulders(dvdt, start_index=5, peak_index=9)

        self.assertEqual(shoulder, 5)


class TestPeriodAnalysis(unittest.TestCase):
    def test_identify_new_periods_splits_on_large_peak_gap(self) -> None:
        peak_times = pd.Series([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])

        periods = identify_new_periods(
            start_time=0.0,
            end_time=12.0,
            peak_times=peak_times,
            min_peaks=3,
            interval_window=3,
            break_multiplier=2.0,
        )

        self.assertEqual(periods, [(0.0, 2.0), (10.0, 12.0)])

    def test_identify_new_periods_returns_empty_with_too_few_peaks(self) -> None:
        peak_times = pd.Series([0.0, 1.0])

        self.assertEqual(
            identify_new_periods(0.0, 2.0, peak_times, min_peaks=3),
            [],
        )


class TestRespiratoryMetrics(unittest.TestCase):
    def test_calculate_respiratory_metrics_returns_expected_cycle_summary(self) -> None:
        metrics = calculate_respiratory_metrics(
            _peak_series(),
            _peak_series(),
            _trough_series(),
            _pressure_data(),
            include_std=True,
        )

        self.assertAlmostEqual(float(metrics["T_I_mean"]), 0.5)
        self.assertAlmostEqual(float(metrics["T_E_mean"]), 1.5)
        self.assertAlmostEqual(float(metrics["T_TOT_mean"]), 2.0)
        self.assertAlmostEqual(float(metrics["Duty_Cycle_mean"]), 0.25)
        self.assertAlmostEqual(float(metrics["Respiratory_Drive_mean"]), 10.0)
        self.assertAlmostEqual(float(metrics["Peak_to_Peak_mean"]), 2.0)
        self.assertAlmostEqual(float(metrics["Frequency_Hz"]), 0.5)
        self.assertAlmostEqual(float(metrics["Pressure Difference"]), 5.0)
        self.assertAlmostEqual(float(metrics["T_I_std"]), 0.0)

    def test_calculate_respiratory_metrics_returns_nan_when_too_short(self) -> None:
        metrics = calculate_respiratory_metrics(
            pd.Series([1.0], index=[1]),
            pd.Series([1.0], index=[1]),
            pd.Series([0.5], index=[0]),
            _pressure_data(),
            include_std=True,
        )

        self.assertTrue(np.isnan(metrics["T_I_mean"]))
        self.assertTrue(np.isnan(metrics["T_TOT_std"]))

    def test_calculate_respiratory_metrics_raw_returns_per_cycle_arrays(self) -> None:
        raw = calculate_respiratory_metrics_raw(
            _peak_series(),
            _peak_series(),
            _trough_series(),
            _pressure_data(),
        )

        np.testing.assert_allclose(raw["T_I"], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(raw["T_E"], [1.5, 1.5])
        np.testing.assert_allclose(raw["T_TOT"], [2.0, 2.0])
        self.assertAlmostEqual(float(raw["Freq"]), 0.5)

    def test_summarize_respiratory_cycles_aggregates_across_periods(self) -> None:
        summary = summarize_respiratory_cycles(
            [_peak_series()],
            [_trough_series()],
            _pressure_data(),
        )

        self.assertAlmostEqual(float(summary["T_I_mean"]), 0.5)
        self.assertAlmostEqual(float(summary["T_TOT_mean"]), 2.0)
        self.assertAlmostEqual(float(summary["Frequency_Hz"]), 0.5)

    def test_calculate_binned_period_metrics_flags_missing_channels(self) -> None:
        pressure = pd.DataFrame({"SmoothedPressure": [10.0, 5.0, 10.0, 5.0]})
        peaks = pd.Series([0.5, 1.5], index=[1, 3])
        troughs = pd.Series([0.25, 1.25], index=[0, 2])

        with self.assertRaises(KeyError):
            calculate_binned_period_metrics(
                period_start_time=0.0,
                period_end_time=2.0,
                bin_size_sec=2,
                peak_times=peaks,
                trough_times=troughs,
                pressure_data=pressure,
                temp_data=pd.DataFrame(),
                activity_data=pd.DataFrame(),
            )

    def test_compute_atm_pressure_session_summary_returns_basic_stats(self) -> None:
        atm = pd.DataFrame({"AtmPressure": [100.0, 102.0, np.nan, 104.0]})

        summary = compute_atm_pressure_session_summary(atm)

        self.assertEqual(float(summary.loc[0, "Mean"]), 102.0)
        self.assertEqual(float(summary.loc[0, "Min"]), 100.0)
        self.assertEqual(float(summary.loc[0, "Max"]), 104.0)

    def test_compute_atm_pressure_time_bins_bins_on_pressure_time_axis(self) -> None:
        pressure = pd.DataFrame({"TimeSinceReference": [0.0, 10.0, 20.0, 30.0]})
        atm = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 5.0, 10.0, 15.0, 25.0],
                "AtmPressure": [100.0, 102.0, 104.0, np.nan, 108.0],
            }
        )

        binned = compute_atm_pressure_time_bins(pressure, atm, bin_size_sec=10)

        self.assertEqual(binned["Bin_Start_s"].tolist(), [0.0, 10.0, 20.0])
        self.assertEqual(binned["N"].tolist(), [2, 1, 1])
        self.assertEqual(float(binned.loc[0, "Mean"]), 101.0)
        self.assertEqual(float(binned.loc[2, "Mean"]), 108.0)


if __name__ == "__main__":
    unittest.main()
