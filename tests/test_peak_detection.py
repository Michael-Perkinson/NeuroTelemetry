import numpy as np
import pandas as pd

import src.core.peak_detection as peak_detection


def test_peak_detection_uses_sampling_rate_from_timestamps(monkeypatch) -> None:
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
