from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.peak_detection import analyse_peaks, find_peaks_and_shoulders


class TestFindPeaksAndShoulders(unittest.TestCase):
    def test_find_peaks_and_shoulders_detects_pressure_troughs(self) -> None:
        time = pd.Series(np.arange(250, dtype=float) / 500.0)
        pressure: np.ndarray = np.zeros(250, dtype=float)
        pressure[[50, 125, 200]] = -8.0
        dvdt = np.gradient(pressure)

        _, _, _, peaks, shoulders = find_peaks_and_shoulders(time, pressure, dvdt)

        self.assertEqual(peaks.tolist(), [50, 125, 200])
        self.assertEqual(len(shoulders), 3)
        self.assertTrue(
            all(
                shoulder <= peak
                for shoulder, peak in zip(shoulders, peaks, strict=False)
            )
        )

    def test_find_peaks_and_shoulders_filters_shallow_troughs(self) -> None:
        time = pd.Series(np.arange(150, dtype=float) / 500.0)
        pressure: np.ndarray = np.zeros(150, dtype=float)
        pressure[50] = -8.0
        pressure[100] = -0.5
        dvdt = np.gradient(pressure)

        _, _, _, peaks, _ = find_peaks_and_shoulders(time, pressure, dvdt)

        self.assertEqual(peaks.tolist(), [50])


class TestAnalysePeaks(unittest.TestCase):
    def test_analyse_peaks_windows_data_and_returns_peak_times(self) -> None:
        pressure_data = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 1.0, 2.0, 3.0, 4.0],
                "SmoothedPressure": [0.0, -5.0, 0.0, -5.0, 0.0],
                "dvdt": [0.0, -1.0, 1.0, -1.0, 1.0],
            }
        )

        with patch(
            "src.core.peak_detection.find_peaks_and_shoulders",
            return_value=(
                np.array([1.0, 2.0, 3.0]),
                np.array([-5.0, 0.0, -5.0]),
                np.array([-1.0, 1.0, -1.0]),
                np.array([0, 2]),
                [0, 1, 2],
            ),
        ):
            results = analyse_peaks([(1.0, 3.0)], pressure_data)

        start, end, peaks, peak_times, shoulders, shoulder_times = results[0]
        self.assertEqual((start, end), (1.0, 3.0))
        self.assertEqual(peaks.tolist(), [0, 2])
        self.assertEqual(shoulders.tolist(), [0, 1])
        self.assertEqual(peak_times.tolist(), [1.0, 3.0])
        self.assertEqual(shoulder_times.tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
