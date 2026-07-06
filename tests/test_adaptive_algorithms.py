from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.core.adaptive_algorithms import (
    butter_highpass_filter,
    butter_lowpass_filter,
    calculate_dynamic_bins,
    compute_first_derivative,
    compute_time_window,
    get_nearest_points,
    get_time_bounds,
)


class TestAdaptiveAlgorithms(unittest.TestCase):
    def test_dynamic_bins_caps_at_array_length(self) -> None:
        self.assertEqual(calculate_dynamic_bins(10), 10)

    def test_dynamic_bins_returns_zero_for_empty_input(self) -> None:
        self.assertEqual(calculate_dynamic_bins(0), 0)

    def test_lowpass_filter_attenuates_high_frequency_component(self) -> None:
        fs = 100.0
        t = np.arange(0.0, 2.0, 1.0 / fs)
        low = np.sin(2.0 * np.pi * 2.0 * t)
        high = 0.5 * np.sin(2.0 * np.pi * 20.0 * t)

        filtered = butter_lowpass_filter(low + high, cutoff_hz=5.0, fs=fs)

        interior = slice(20, -20)
        residual = filtered[interior] - low[interior]
        self.assertLess(float(np.std(residual)), float(np.std(high[interior])) * 0.4)

    def test_highpass_filter_attenuates_low_frequency_component(self) -> None:
        fs = 100.0
        t = np.arange(0.0, 2.0, 1.0 / fs)
        low = np.sin(2.0 * np.pi * 1.0 * t)
        high = 0.5 * np.sin(2.0 * np.pi * 15.0 * t)

        filtered = butter_highpass_filter(low + high, fs=fs, cutoff_hz=5.0)

        interior = slice(20, -20)
        residual = filtered[interior] - high[interior]
        self.assertLess(float(np.std(residual)), float(np.std(low[interior])) * 0.4)

    def test_first_derivative_matches_quadratic_slope(self) -> None:
        fs = 50.0
        t = np.arange(0.0, 2.0, 1.0 / fs)
        signal = t**2

        derivative = compute_first_derivative(signal, fs)

        np.testing.assert_allclose(derivative[20:-20], 2.0 * t[20:-20], atol=1e-10)

    def test_get_time_bounds_returns_min_and_max(self) -> None:
        pressure_data = pd.DataFrame({"TimeSinceReference": [3.0, -1.0, 2.0]})

        self.assertEqual(get_time_bounds(pressure_data), (-1.0, 3.0))

    def test_compute_time_window_clips_to_available_data(self) -> None:
        self.assertEqual(
            compute_time_window(
                photo_min_time=100.0,
                photo_max_time=500.0,
                injection_sec=250.0,
                pre_min=5.0,
                post_min=10.0,
            ),
            (100.0, 500.0),
        )

    def test_get_nearest_points_uses_closest_sample_and_tie_goes_left(self) -> None:
        df = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 10.0, 20.0, 30.0],
                "Pressure": [1.0, 2.0, 3.0, 4.0],
            }
        )

        matched_times, matched_values = get_nearest_points(
            [-5.0, 5.0, 16.0, 35.0],
            df,
            "TimeSinceReference",
            "Pressure",
        )

        self.assertEqual(matched_times, [0.0, 0.0, 20.0, 30.0])
        self.assertEqual(matched_values, [1.0, 1.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
