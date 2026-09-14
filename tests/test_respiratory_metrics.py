from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.core.respiratory.respiratory_metrics import (
    calculate_binned_period_metrics,
    calculate_period_respiratory_metrics,
    calculate_period_respiratory_metrics_raw,
    compute_atm_pressure_session_summary,
    compute_atm_pressure_time_bins,
    summarize_respiratory_cycles,
)


def _make_pressure_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TimeSinceReference": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "SmoothedPressure": [0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
        }
    )


def test_calculate_period_respiratory_metrics_returns_expected_cycle_means() -> None:
    peak_times = pd.Series([0.5, 1.5, 2.5], index=[1, 3, 5])
    trough_times = pd.Series([0.0, 1.0, 2.0], index=[0, 2, 4])

    metrics = calculate_period_respiratory_metrics(
        period_peaks=peak_times,
        peak_indices=peak_times,
        trough_indices=trough_times,
        pressure_data=_make_pressure_data(),
    )

    assert metrics["T_I_mean"] == 0.5
    assert metrics["T_E_mean"] == 0.5
    assert metrics["T_TOT_mean"] == 1.0
    assert metrics["Duty_Cycle_mean"] == 0.5
    assert metrics["Respiratory_Drive_mean"] == 4.0
    assert metrics["Peak_to_Peak_mean"] == 1.0
    assert metrics["Frequency_Hz"] == 1.0
    assert metrics["Pressure Difference"] == 2.0


def test_calculate_period_respiratory_metrics_returns_nan_for_insufficient_cycles() -> (
    None
):
    metrics = calculate_period_respiratory_metrics(
        period_peaks=pd.Series([0.5], index=[1]),
        peak_indices=pd.Series([0.5], index=[1]),
        trough_indices=pd.Series([0.0], index=[0]),
        pressure_data=_make_pressure_data(),
        include_std=True,
    )

    expected_keys = {
        "T_I_mean",
        "T_I_std",
        "T_E_mean",
        "T_E_std",
        "T_TOT_mean",
        "T_TOT_std",
        "Duty_Cycle_mean",
        "Duty_Cycle_std",
        "Respiratory_Drive_mean",
        "Respiratory_Drive_std",
        "Peak_to_Peak_mean",
        "Peak_to_Peak_std",
        "Frequency_Hz",
        "Frequency_Hz_std",
        "Pressure Difference",
        "Pressure Difference_std",
    }
    assert set(metrics) == expected_keys
    assert all(np.isnan(value) for value in metrics.values())


def test_calculate_period_respiratory_metrics_raw_returns_per_cycle_arrays() -> None:
    peak_times = pd.Series([0.5, 1.5, 2.5], index=[1, 3, 5])
    trough_times = pd.Series([0.0, 1.0, 2.0], index=[0, 2, 4])

    raw = calculate_period_respiratory_metrics_raw(
        period_peaks=peak_times,
        peak_indices=peak_times,
        trough_indices=trough_times,
        pressure_data=_make_pressure_data(),
    )

    assert np.asarray(raw["T_I"]).tolist() == [0.5, 0.5, 0.5]
    assert np.asarray(raw["T_E"]).tolist() == [0.5, 0.5]
    assert np.asarray(raw["T_TOT"]).tolist() == [1.0, 1.0]
    assert np.asarray(raw["Duty_Cycle"]).tolist() == [0.5, 0.5]
    assert np.asarray(raw["Drive"]).tolist() == [4.0, 4.0, 4.0]
    assert np.asarray(raw["Pressure_Diff"]).tolist() == [2.0, 2.0, 2.0]
    assert np.asarray(raw["Peak_to_Peak"]).tolist() == [1.0, 1.0]
    assert np.asarray(raw["Freq"]).tolist() == [1.0, 1.0]


def test_summary_metrics_match_aggregation_of_raw_cycle_metrics() -> None:
    peak_times = pd.Series([0.5, 1.7, 2.8], index=[1, 3, 5])
    trough_times = pd.Series([0.0, 1.1, 2.1], index=[0, 2, 4])
    pressure_data = _make_pressure_data()

    raw = calculate_period_respiratory_metrics_raw(
        peak_times, peak_times, trough_times, pressure_data
    )
    summary = calculate_period_respiratory_metrics(
        peak_times, peak_times, trough_times, pressure_data, include_std=True
    )

    names = {
        "T_I": "T_I",
        "T_E": "T_E",
        "T_TOT": "T_TOT",
        "Duty_Cycle": "Duty_Cycle",
        "Drive": "Respiratory_Drive",
        "Peak_to_Peak": "Peak_to_Peak",
        "Freq": "Frequency_Hz",
        "Pressure_Diff": "Pressure Difference",
    }
    for raw_name, summary_name in names.items():
        values = np.asarray(raw[raw_name])
        mean_key = (
            summary_name
            if raw_name in {"Freq", "Pressure_Diff"}
            else f"{summary_name}_mean"
        )
        assert np.isclose(summary[mean_key], values.mean())
        assert np.isclose(summary[f"{summary_name}_std"], values.std())


def test_compute_atm_pressure_session_summary_handles_present_and_empty_data() -> None:
    atm = pd.DataFrame({"AtmPressure": [100.0, 102.0, 104.0]})

    summary = compute_atm_pressure_session_summary(atm)
    empty_summary = compute_atm_pressure_session_summary(pd.DataFrame())

    assert summary.to_dict("records") == [
        {"Mean": 102.0, "SD": 2.0, "Min": 100.0, "Max": 104.0}
    ]
    assert empty_summary.empty
    assert empty_summary.columns.tolist() == ["Mean", "SD", "Min", "Max"]


def test_compute_atm_pressure_time_bins_uses_pressure_time_axis() -> None:
    pressure_data = pd.DataFrame({"TimeSinceReference": [0.0, 60.0, 120.0]})
    atm_pressure_data = pd.DataFrame(
        {
            "TimeSinceReference": [0.0, 10.0, 70.0, 120.0],
            "AtmPressure": [100.0, 102.0, 110.0, 114.0],
        }
    )

    binned = compute_atm_pressure_time_bins(
        pressure_data=pressure_data,
        atm_pressure_data=atm_pressure_data,
        bin_size_sec=60,
    )

    assert binned["Bin_Start_s"].tolist() == [0.0, 60.0]
    assert binned["Bin_End_s"].tolist() == [60.0, 120.0]
    assert binned["Mean"].tolist() == [101.0, 112.0]
    assert np.allclose(binned["SD"].iloc[0], np.sqrt(2.0))
    assert np.allclose(binned["SD"].iloc[1], np.sqrt(8.0))
    assert binned["Min"].tolist() == [100.0, 110.0]
    assert binned["Max"].tolist() == [102.0, 114.0]
    assert binned["N"].tolist() == [2, 2]


def _pressure_data() -> pd.DataFrame:
    return pd.DataFrame({"SmoothedPressure": [10.0, 5.0, 10.0, 5.0, 10.0, 5.0]})


def _peak_series() -> pd.Series:
    return pd.Series([1.0, 3.0, 5.0], index=[1, 3, 5])


def _trough_series() -> pd.Series:
    return pd.Series([0.5, 2.5, 4.5], index=[0, 2, 4])


class TestRespiratoryMetrics(unittest.TestCase):
    def test_calculate_period_respiratory_metrics_returns_expected_cycle_summary(
        self,
    ) -> None:
        metrics = calculate_period_respiratory_metrics(
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

    def test_calculate_period_respiratory_metrics_returns_nan_when_too_short(
        self,
    ) -> None:
        metrics = calculate_period_respiratory_metrics(
            pd.Series([1.0], index=[1]),
            pd.Series([1.0], index=[1]),
            pd.Series([0.5], index=[0]),
            _pressure_data(),
            include_std=True,
        )

        self.assertTrue(np.isnan(metrics["T_I_mean"]))
        self.assertTrue(np.isnan(metrics["T_TOT_std"]))

    def test_calculate_period_respiratory_metrics_raw_returns_per_cycle_arrays(
        self,
    ) -> None:
        raw = calculate_period_respiratory_metrics_raw(
            _peak_series(),
            _peak_series(),
            _trough_series(),
            _pressure_data(),
        )

        np.testing.assert_allclose(raw["T_I"], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(raw["T_E"], [1.5, 1.5])
        np.testing.assert_allclose(raw["T_TOT"], [2.0, 2.0])
        np.testing.assert_allclose(raw["Freq"], [0.5, 0.5])

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
