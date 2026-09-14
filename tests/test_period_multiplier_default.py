from __future__ import annotations

import pandas as pd

from src.core.respiratory.period_analysis import (
    DEFAULT_BREAK_MULTIPLIER,
    identify_new_periods,
)


def test_default_break_multiplier_is_fourfold() -> None:
    peak_times = pd.Series([0.0, 1.0, 2.0, 5.0, 6.0, 7.0])

    default_periods = identify_new_periods(
        0.0,
        7.0,
        peak_times,
        min_peaks=3,
        interval_window=3,
    )
    twofold_periods = identify_new_periods(
        0.0,
        7.0,
        peak_times,
        min_peaks=3,
        interval_window=3,
        break_multiplier=2.0,
    )

    assert DEFAULT_BREAK_MULTIPLIER == 4.0
    assert default_periods == [(0.0, 7.0)]
    assert twofold_periods == [(0.0, 2.0), (5.0, 7.0)]
