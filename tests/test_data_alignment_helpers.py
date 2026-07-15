from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.data_alignment import (
    _decide_dayfirst_from_probe,
    _to_datetime_with_ms,
    adjust_behaviours,
    align_and_clean_datetime,
    build_output_frames,
    compute_time_offsets,
    parse_numerical_data,
    parse_reference_timestamps,
    prepare_numerical_data,
    prepare_raw_data,
    preprocess_pressure_data,
    safe_interpolate,
    split_data,
)


class TestDataAlignmentHelpers(unittest.TestCase):
    def test_split_data_separates_metadata_from_numerical_rows(self) -> None:
        raw = pd.DataFrame(
            {
                0: [
                    "# Col 2:,Pressure,x,x,Rate: 500",
                    "Time,Pressure",
                    "01/11/2025 05:05:09 PM,1.0",
                ]
            }
        )

        meta, numerical = split_data(raw, skip_rows=1)

        self.assertEqual(meta.iloc[0, 1], "Pressure")
        self.assertEqual(numerical.iloc[0, 0], "Time")

    def test_prepare_numerical_data_keeps_first_duplicate_signal(self) -> None:
        meta = pd.DataFrame(
            [
                ["# Col 2:", "Pressure", None, None, "Rate: 500"],
                ["# Col 3:", "Temp", None, None, "Rate: 10"],
                ["# Col 4:", "Activity", None, None, "Rate: 20"],
                ["# Col 5:", "APR", None, None, "Rate: 1"],
                ["# Col 6:", "Pressure", None, None, "Rate: 250"],
            ]
        )
        numerical = pd.DataFrame(
            [
                ["DateTime", "Pressure", "Temp", "Activity", "APR", "Pressure2"],
                ["01/11/2025 05:05:09 PM", "1", "36", "5", "100", "99"],
            ]
        )

        cleaned, sample_rates = prepare_numerical_data(meta, numerical)

        self.assertEqual(
            cleaned.columns.tolist(),
            ["DateTime", "Pressure", "Temp", "Activity", "AtmPressure"],
        )
        self.assertEqual(
            cleaned.iloc[0].tolist(),
            [
                "01/11/2025 05:05:09 PM",
                "1",
                "36",
                "5",
                "100",
            ],
        )
        self.assertEqual(
            sample_rates,
            {"Pressure": 500.0, "Temp": 10.0, "Activity": 20.0, "AtmPressure": 1.0},
        )

    def test_prepare_numerical_data_rejects_unexpected_rate_format(self) -> None:
        meta = pd.DataFrame([["# Col 2:", "Pressure", None, None, "500 Hz"]])
        numerical = pd.DataFrame(
            [
                ["DateTime", "Pressure"],
                ["01/11/2025 05:05:09 PM", "1"],
            ]
        )

        with self.assertRaisesRegex(ValueError, "Unexpected rate format"):
            prepare_numerical_data(meta, numerical)

    def test_decide_dayfirst_uses_probe_reference_when_ambiguous(self) -> None:
        probe = datetime(2025, 11, 1, 17, 5, 9)

        self.assertFalse(_decide_dayfirst_from_probe("11/01/2025 05:05:09 PM", probe))
        self.assertTrue(_decide_dayfirst_from_probe("01/11/2025 05:05:09 PM", probe))

    def test_decide_dayfirst_uses_greater_than_twelve_rule(self) -> None:
        probe = datetime(2025, 1, 1, 0, 0, 0)

        self.assertTrue(_decide_dayfirst_from_probe("25/04/2025 01:00:00 PM", probe))
        self.assertFalse(_decide_dayfirst_from_probe("04/25/2025 01:00:00 PM", probe))

    def test_to_datetime_with_ms_reconstructs_missing_subseconds(self) -> None:
        series = pd.Series(
            [
                "01/11/2025 05:05:09 PM",
                "01/11/2025 05:05:09 PM",
                "01/11/2025 05:05:09 PM",
            ]
        )

        parsed = _to_datetime_with_ms(series, dayfirst=True, sample_rate_hz=2.0)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2025-11-01 17:05:09"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2025-11-01 17:05:09.500000"))
        self.assertEqual(parsed.iloc[2], pd.Timestamp("2025-11-01 17:05:10"))

    def test_to_datetime_with_ms_parses_ms_before_ampm(self) -> None:
        series = pd.Series(["01/11/2025 05:05:09.123 PM"])

        parsed = _to_datetime_with_ms(series, dayfirst=True, sample_rate_hz=500.0)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2025-11-01 17:05:09.123"))

    def test_to_datetime_with_ms_parses_ms_after_ampm(self) -> None:
        series = pd.Series(["01/11/2025 05:05:09 PM.250"])

        parsed = _to_datetime_with_ms(series, dayfirst=True, sample_rate_hz=500.0)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2025-11-01 17:05:09.250"))

    def test_compute_time_offsets_combines_offsets(self) -> None:
        probe = datetime(2025, 1, 1, 12, 0, 0)
        video = datetime(2025, 1, 1, 12, 1, 30)

        self.assertEqual(
            compute_time_offsets(video, probe, removed_nan=2.5),
            (90.0, 87.5),
        )

    def test_parse_reference_timestamps_uses_day_first_format(self) -> None:
        probe, align = parse_reference_timestamps(
            "01/11/2025 05:05:09 PM",
            "01/11/2025 04:59:59 PM",
        )

        self.assertEqual(probe, datetime(2025, 11, 1, 17, 5, 9))
        self.assertEqual(align, datetime(2025, 11, 1, 16, 59, 59))

    def test_parse_numerical_data_coerces_signal_columns_and_parses_time(self) -> None:
        numerical = pd.DataFrame(
            {
                "DateTime": [
                    "01/11/2025 05:05:09 PM",
                    "01/11/2025 05:05:09 PM",
                ],
                "Pressure": ["1.5", "bad"],
            }
        )

        parsed = parse_numerical_data(
            numerical,
            datetime(2025, 11, 1, 17, 5, 9),
            {"Pressure": 2.0},
        )

        self.assertEqual(parsed.loc[0, "DateTime"], pd.Timestamp("2025-11-01 17:05:09"))
        self.assertTrue(np.isnan(parsed.loc[1, "Pressure"]))

    def test_align_and_clean_datetime_drops_duplicates_and_leading_nan(self) -> None:
        numerical = pd.DataFrame(
            {
                "DateTime": [
                    pd.Timestamp("2025-01-01 12:00:00"),
                    pd.Timestamp("2025-01-01 12:00:00"),
                    pd.Timestamp("2025-01-01 12:00:01"),
                    pd.Timestamp("2025-01-01 12:00:02"),
                ],
                "Pressure": [np.nan, 99.0, 1.0, 2.0],
            }
        )

        cleaned, new_ref, removed = align_and_clean_datetime(
            numerical,
            datetime(2025, 1, 1, 12, 0, 1),
        )

        self.assertEqual(new_ref, datetime(2025, 1, 1, 12, 0, 1))
        self.assertEqual(removed, 0.0)
        self.assertEqual(cleaned["Pressure"].tolist(), [1.0, 2.0])

    def test_align_and_clean_datetime_rejects_missing_or_invalid_signal(self) -> None:
        with self.assertRaisesRegex(ValueError, "No usable signal"):
            align_and_clean_datetime(
                pd.DataFrame({"DateTime": [pd.Timestamp("2025-01-01")]}),
                datetime(2025, 1, 1),
            )

        with self.assertRaisesRegex(ValueError, "No usable signal"):
            align_and_clean_datetime(
                pd.DataFrame(
                    {
                        "DateTime": [pd.Timestamp("2025-01-01")],
                        "Pressure": [np.nan],
                    }
                ),
                datetime(2025, 1, 1),
            )

    def test_align_and_clean_datetime_rejects_out_of_range_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "not within data range"):
            align_and_clean_datetime(
                pd.DataFrame(
                    {
                        "DateTime": [pd.Timestamp("2025-01-01 12:00:00")],
                        "Pressure": [1.0],
                    }
                ),
                datetime(2025, 1, 1, 12, 1, 0),
            )

    def test_build_output_frames_interpolates_optional_channels_and_atm(self) -> None:
        numerical = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 0.5, 1.0, 1.5, 2.0],
                "Pressure": [1.0, 2.0, 1.0, 2.0, 1.0],
                "Temp": [36.0, 36.5, 37.0, 37.5, 38.0],
                "Activity": [0.0, 1.0, 2.0, 3.0, 4.0],
                "AtmPressure": [100.0, 100.0, 101.0, 101.0, 102.0],
            }
        )
        pressure_out = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 1.0, 2.0],
                "Pressure": [1.0, 1.0, 1.0],
            }
        )

        with patch(
            "src.core.data_alignment.preprocess_pressure_data",
            return_value=pressure_out,
        ):
            processed = build_output_frames(
                numerical,
                {"Pressure": 2.0, "AtmPressure": 1.0},
            )

        self.assertEqual(processed["Temp"]["Temp"].tolist(), [36.0, 37.0, 38.0])
        self.assertEqual(processed["Activity"]["Activity"].tolist(), [0.0, 2.0, 4.0])
        self.assertEqual(
            processed["AtmPressure"]["AtmPressure"].tolist(),
            [100.0, 101.0, 101.0],
        )

    def test_build_output_frames_supports_temperature_without_pressure(self) -> None:
        processed = build_output_frames(
            pd.DataFrame({"TimeSinceReference": [0.0], "Temp": [36.0]}),
            {},
        )

        self.assertNotIn("Pressure", processed)
        self.assertEqual(processed["Temp"]["Temp"].tolist(), [36.0])

    def test_preprocess_pressure_data_adds_derived_pressure_columns(self) -> None:
        time = np.arange(0.0, 2.0, 0.002)
        pressure = 1.0 + 0.1 * np.sin(2.0 * np.pi * 2.0 * time)
        pressure[10] = np.nan
        pressure_df = pd.DataFrame(
            {
                "TimeSinceReference": time,
                "Pressure": pressure,
            }
        )

        processed = preprocess_pressure_data(pressure_df, pressure_sample_rate=500.0)

        self.assertIn("PressureHighpass", processed.columns)
        self.assertIn("SmoothedPressure", processed.columns)
        self.assertIn("dvdt", processed.columns)
        self.assertFalse(processed["SmoothedPressure"].isna().any())

    def test_safe_interpolate_maps_signal_onto_requested_time_axis(self) -> None:
        source = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 10.0, 20.0],
                "Temp": [36.0, 38.0, 40.0],
            }
        )
        time_axis = pd.Series([0.0, 5.0, 10.0, 15.0, 20.0])

        interpolated = safe_interpolate(source, time_axis, "Temp")

        self.assertEqual(
            interpolated["TimeSinceReference"].tolist(),
            time_axis.tolist(),
        )
        np.testing.assert_allclose(
            interpolated["Temp"],
            [36.0, 37.0, 38.0, 39.0, 40.0],
        )

    def test_safe_interpolate_returns_schema_for_missing_signal(self) -> None:
        interpolated = safe_interpolate(pd.DataFrame(), pd.Series([0.0, 1.0]), "Temp")

        self.assertEqual(interpolated.columns.tolist(), ["TimeSinceReference", "Temp"])
        self.assertTrue(interpolated.empty)

    def test_adjust_behaviours_offsets_times_and_drops_negative_windows(self) -> None:
        behaviour_data = {
            "Sleep": [
                (1, -5.0, 10.0, 15.0),
                (2, 10.0, 20.0, 10.0),
            ]
        }

        adjusted = adjust_behaviours(behaviour_data, total_time_diff=-8.0)

        self.assertEqual(adjusted["Sleep"], [(2, 2.0, 12.0, 10.0)])
        self.assertEqual(behaviour_data["Sleep"][1], (2, 10.0, 20.0, 10.0))

    def test_prepare_raw_data_aligns_signals_to_injection_time(self) -> None:
        photometry = pd.DataFrame(
            {
                "TimeSinceReference": [100.0, 110.0, 120.0],
                "dFoF_465": [1.0, 2.0, 3.0],
            }
        )
        temp = pd.DataFrame(
            {
                "TimeSinceReference": [100.0, 120.0],
                "Temp": [36.0, 38.0],
            }
        )
        activity = pd.DataFrame(
            {
                "TimeSinceReference": [100.0, 120.0],
                "Activity": [0.0, 10.0],
            }
        )

        raw = prepare_raw_data(photometry, temp, activity, injection_sec=110.0)

        self.assertEqual(raw["TimeRel"].tolist(), [-10.0, 0.0, 10.0])
        np.testing.assert_allclose(raw["Temp"], [36.0, 37.0, 38.0])
        np.testing.assert_allclose(raw["Activity"], [0.0, 5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
