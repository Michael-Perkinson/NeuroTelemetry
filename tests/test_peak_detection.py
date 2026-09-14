import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.core.respiratory.peak_detection as peak_detection
from src.core.respiratory.peak_detection import (
    analyse_peaks,
    find_peaks_and_shoulders,
    find_shoulder_index,
)


def test_find_peaks_and_shoulders_uses_sampling_rate_from_timestamps(
    monkeypatch,
) -> None:
    captured: dict[str, int] = {}

    def fake_find_peaks(_signal, *, prominence, distance):
        assert prominence == 3
        captured["distance"] = distance
        return np.array([], dtype=int), {}

    monkeypatch.setattr(peak_detection, "find_peaks", fake_find_peaks)
    time = pd.Series(np.arange(100, dtype=float) / 100.0)
    signal = pd.Series(np.zeros(100))

    peak_detection.find_peaks_and_shoulders(time, signal, np.zeros(100))

    assert captured["distance"] == 5


class TestFindPeaksAndShoulders(unittest.TestCase):
    def test_find_peaks_and_shoulders_detects_pressure_troughs(self) -> None:
        time = pd.Series(np.arange(250, dtype=float) / 500.0)
        pressure: np.ndarray = np.zeros(250, dtype=float)
        pressure[[50, 125, 200]] = -8.0
        dvdt = np.gradient(pressure)

        peaks, shoulders = find_peaks_and_shoulders(time, pressure, dvdt)

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

        peaks, _ = find_peaks_and_shoulders(time, pressure, dvdt)

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
            "src.core.respiratory.peak_detection.find_peaks_and_shoulders",
            return_value=(
                np.array([0, 2]),
                np.array([0, 1, 2]),
            ),
        ):
            results = analyse_peaks([(1.0, 3.0)], pressure_data)

        start, end, peaks, peak_times, shoulders, shoulder_times = results[0]
        self.assertEqual((start, end), (1.0, 3.0))
        self.assertEqual(peaks.tolist(), [0, 2])
        self.assertEqual(shoulders.tolist(), [0, 1])
        self.assertEqual(peak_times.tolist(), [1.0, 3.0])
        self.assertEqual(shoulder_times.tolist(), [1.0, 2.0])


class TestPeakDetection(unittest.TestCase):
    def test_find_shoulder_index_returns_start_for_empty_segment(self) -> None:
        self.assertEqual(
            find_shoulder_index(np.array([]), start_index=10, peak_index=20),
            10,
        )

    def test_find_shoulder_index_uses_last_positive_to_negative_zero_crossing(
        self,
    ) -> None:
        dvdt = np.array([0.2, 0.1, -0.05, -0.5, -1.0, -0.2, 0.1])

        shoulder = find_shoulder_index(dvdt, start_index=10, peak_index=20)

        self.assertEqual(shoulder, 12)

    def test_find_shoulder_index_falls_back_to_nearest_zero_before_minimum(
        self,
    ) -> None:
        dvdt = np.array([-0.1, -0.2, -1.0, -0.3])

        shoulder = find_shoulder_index(dvdt, start_index=5, peak_index=9)

        self.assertEqual(shoulder, 5)


if __name__ == "__main__":
    unittest.main()
