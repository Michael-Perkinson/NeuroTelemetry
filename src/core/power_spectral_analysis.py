from pathlib import Path

import numpy as np
import pandas as pd


def export_ttot_traces(
    window_periods: dict[str, list[dict]], output_folder: Path, log=print
) -> None:
    """
    Export Ttot breath-by-breath traces for each behavioural window.

    Creates one CSV per window:
        TimeOfBreath_s, Ttot_s, PeriodIndex

    window_periods: dict mapping window_key -> list of period dicts.
                    Each dict contains peak_times, trough_times, etc.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for window_key, periods in window_periods.items():
        rows = []

        for period_idx, period in enumerate(periods, start=1):
            if "peak_times" not in period:
                log(f"Period missing peak_times for window {window_key}, skipping.")
                continue

            peak_times = period["peak_times"]

            if len(peak_times) < 2:
                continue

            # Convert to float numpy array
            p = peak_times.to_numpy(dtype=float)

            # Ttot = peak-to-peak interval
            ttot = np.diff(p)  # length n-1
            # timestamp of first peak in each interval
            t_breath = p[:-1]

            for t, tdur in zip(t_breath, ttot, strict=False):
                rows.append(
                    {
                        "TimeOfBreath_s": float(t),
                        "Ttot_s": float(tdur),
                        "PeriodIndex": period_idx,
                    }
                )

        if not rows:
            log(f"No valid breath data in window {window_key}, skipping CSV.")
            continue

        df = pd.DataFrame(rows)

        safe_key = window_key.replace(":", "_").replace("-", "_").replace(" ", "_")

        out_path = output_folder / f"Ttot_{safe_key}.csv"
        df.to_csv(out_path, index=False)
        log(f"Exported Ttot CSV for window {window_key} -> {out_path}")
