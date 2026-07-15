from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.respiratory_metrics import (
    calculate_respiratory_metrics,
    calculate_respiratory_metrics_raw,
    compute_atm_pressure_session_summary,
    compute_atm_pressure_time_bins,
)


def _make_pressure_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TimeSinceReference": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "SmoothedPressure": [0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
        }
    )


def test_calculate_respiratory_metrics_returns_expected_cycle_means() -> None:
    peak_times = pd.Series([0.5, 1.5, 2.5], index=[1, 3, 5])
    trough_times = pd.Series([0.0, 1.0, 2.0], index=[0, 2, 4])

    metrics = calculate_respiratory_metrics(
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


def test_calculate_respiratory_metrics_returns_nan_for_insufficient_cycles() -> None:
    metrics = calculate_respiratory_metrics(
        period_peaks=pd.Series([0.5], index=[1]),
        peak_indices=pd.Series([0.5], index=[1]),
        trough_indices=pd.Series([0.0], index=[0]),
        pressure_data=_make_pressure_data(),
        include_std=True,
    )

    assert np.isnan(metrics["T_I_mean"])
    assert np.isnan(metrics["T_I_std"])
    assert np.isnan(metrics["Frequency_Hz"])


def test_calculate_respiratory_metrics_raw_returns_per_cycle_arrays() -> None:
    peak_times = pd.Series([0.5, 1.5, 2.5], index=[1, 3, 5])
    trough_times = pd.Series([0.0, 1.0, 2.0], index=[0, 2, 4])

    raw = calculate_respiratory_metrics_raw(
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
    assert raw["Freq"] == 1.0


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
