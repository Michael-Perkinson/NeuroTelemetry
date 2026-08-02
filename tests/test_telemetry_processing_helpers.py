from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.telemetry_processing import (
    _decide_dayfirst_from_reference,
    _to_datetime_with_ms,
    adjust_behaviours,
    align_and_clean_datetime,
    build_output_frames,
    continuous_signal_coverage,
    label_numerical_columns_from_metadata,
    parse_numerical_values_with_timestamps,
    parse_reference_timestamp,
    prepare_raw_data,
    preprocess_pressure_data,
    require_signal_frame,
    safe_interpolate,
    split_data,
    trim_coverage_intervals,
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

    def test_label_numerical_columns_from_metadata_keeps_first_duplicate_signal(
        self,
    ) -> None:
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

        cleaned, sample_rates = label_numerical_columns_from_metadata(meta, numerical)

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

    def test_label_numerical_columns_from_metadata_rejects_unexpected_rate_format(
        self,
    ) -> None:
        meta = pd.DataFrame([["# Col 2:", "Pressure", None, None, "500 Hz"]])
        numerical = pd.DataFrame(
            [
                ["DateTime", "Pressure"],
                ["01/11/2025 05:05:09 PM", "1"],
            ]
        )

        with self.assertRaisesRegex(ValueError, "Unexpected rate format"):
            label_numerical_columns_from_metadata(meta, numerical)

    def test_label_numerical_columns_from_metadata_rejects_zero_sample_rate(
        self,
    ) -> None:
        meta = pd.DataFrame([["# Col 2:", "Pressure", None, None, "Rate: 0"]])
        numerical = pd.DataFrame(
            [
                ["DateTime", "Pressure"],
                ["01/11/2025 05:05:09 PM", "1"],
            ]
        )

        with self.assertRaisesRegex(
            ValueError, r"Invalid sample rate for 'Pressure': 0.*must be positive"
        ):
            label_numerical_columns_from_metadata(meta, numerical)

    def test_label_numerical_columns_from_metadata_rejects_negative_sample_rate(
        self,
    ) -> None:
        meta = pd.DataFrame([["# Col 2:", "Activity", None, None, "Rate: -100"]])
        numerical = pd.DataFrame(
            [
                ["DateTime", "Activity"],
                ["01/11/2025 05:05:09 PM", "5"],
            ]
        )

        with self.assertRaisesRegex(
            ValueError, r"Invalid sample rate for 'Activity': -100.*must be positive"
        ):
            label_numerical_columns_from_metadata(meta, numerical)

    def test_label_numerical_columns_from_metadata_warns_on_unrecognized_signal_names(
        self,
    ) -> None:
        """Test that unrecognized signal names in metadata are logged as warnings
        while recognized signals are processed normally."""
        meta = pd.DataFrame(
            [
                ["# Col 2:", "Temp", None, None, "Rate: 10"],
                ["# Col 3:", "Temp (C)", None, None, "Rate: 10"],
                ["# Col 4:", "Activity", None, None, "Rate: 20"],
            ]
        )
        numerical = pd.DataFrame(
            [
                ["DateTime", "Temp", "TempC", "Activity"],
                ["01/11/2025 05:05:09 PM", "36.0", "36.0", "5"],
            ]
        )

        with patch("src.core.telemetry_processing.log_warning") as mock_warn:
            cleaned, sample_rates = label_numerical_columns_from_metadata(
                meta, numerical
            )

            # Verify that recognized signals are still present
            self.assertIn("Temp", cleaned.columns)
            self.assertIn("Activity", cleaned.columns)
            # Unrecognized signal "Temp (C)" should NOT be in output
            self.assertNotIn("TempC", cleaned.columns)

            # Verify that a warning was logged mentioning the unrecognized name
            self.assertTrue(
                any(
                    "temp (c)" in str(call).lower() for call in mock_warn.call_args_list
                ),
                f"Expected 'temp (c)' in log_warning calls, "
                f"got: {mock_warn.call_args_list}",
            )

    def test_decide_dayfirst_uses_alignment_reference_when_ambiguous(self) -> None:
        reference = datetime(2025, 11, 1, 17, 5, 9)

        self.assertFalse(
            _decide_dayfirst_from_reference("11/01/2025 05:05:09 PM", reference)
        )
        self.assertTrue(
            _decide_dayfirst_from_reference("01/11/2025 05:05:09 PM", reference)
        )

    def test_decide_dayfirst_uses_greater_than_twelve_rule(self) -> None:
        reference = datetime(2025, 1, 1, 0, 0, 0)

        self.assertTrue(
            _decide_dayfirst_from_reference("25/04/2025 01:00:00 PM", reference)
        )
        self.assertFalse(
            _decide_dayfirst_from_reference("04/25/2025 01:00:00 PM", reference)
        )

    def test_decide_dayfirst_uses_nearest_date_across_midnight(self) -> None:
        reference = datetime(2025, 1, 31, 23, 59, 50)

        self.assertTrue(
            _decide_dayfirst_from_reference("01/02/2025 12:00:10 AM", reference)
        )

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

    def test_to_datetime_with_ms_preserves_visible_timestamp_gaps(self) -> None:
        series = pd.Series(
            [
                "01/11/2025 05:05:09 PM",
                "01/11/2025 05:05:09 PM",
                "01/11/2025 05:05:12 PM",
                "01/11/2025 05:05:12 PM",
            ]
        )

        parsed = _to_datetime_with_ms(series, dayfirst=True, sample_rate_hz=2.0)

        self.assertEqual(parsed.iloc[1], pd.Timestamp("2025-11-01 17:05:09.500"))
        self.assertEqual(parsed.iloc[2], pd.Timestamp("2025-11-01 17:05:12"))

    def test_to_datetime_with_ms_parses_ms_before_ampm(self) -> None:
        series = pd.Series(["01/11/2025 05:05:09.123 PM"])

        parsed = _to_datetime_with_ms(series, dayfirst=True, sample_rate_hz=500.0)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2025-11-01 17:05:09.123"))

    def test_to_datetime_with_ms_parses_ms_after_ampm(self) -> None:
        series = pd.Series(["01/11/2025 05:05:09 PM.250"])

        parsed = _to_datetime_with_ms(series, dayfirst=True, sample_rate_hz=500.0)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2025-11-01 17:05:09.250"))

    def test_parse_reference_timestamp_uses_day_first_format(self) -> None:
        reference = parse_reference_timestamp("01/11/2025 04:59:59 PM")

        self.assertEqual(reference, datetime(2025, 11, 1, 16, 59, 59))

    def test_parse_numerical_values_with_timestamps_coerces_and_parses_time(
        self,
    ) -> None:
        numerical = pd.DataFrame(
            {
                "DateTime": [
                    "01/11/2025 05:05:09 PM",
                    "01/11/2025 05:05:09 PM",
                ],
                "Pressure": ["1.5", "bad"],
            }
        )

        parsed = parse_numerical_values_with_timestamps(
            numerical,
            datetime(2025, 11, 1, 17, 5, 9),
            {"Pressure": 2.0},
        )

        self.assertEqual(parsed.loc[0, "DateTime"], pd.Timestamp("2025-11-01 17:05:09"))
        self.assertTrue(np.isnan(parsed.loc[1, "Pressure"]))

    def test_align_and_clean_datetime_deduplicates_preserving_valid_values(
        self,
    ) -> None:
        """Test that when DateTime duplicates exist, we preserve valid (non-NaN)
        measurements even if they appear in later rows of the duplicate group."""
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

        cleaned, new_ref = align_and_clean_datetime(numerical)

        # groupby().first() with duplicate DateTime should pick the first non-NaN
        # Pressure value (99.0), not the NaN from row 0
        self.assertEqual(new_ref, datetime(2025, 1, 1, 12, 0, 0))
        self.assertEqual(cleaned["Pressure"].tolist(), [99.0, 1.0, 2.0])

    def test_align_and_clean_datetime_handles_conflicting_values_in_duplicates(
        self,
    ) -> None:
        """Test that when duplicate DateTimes have conflicting non-NaN values
        in the same column, we pick the first and log a warning."""
        numerical = pd.DataFrame(
            {
                "DateTime": [
                    pd.Timestamp("2025-01-01 12:00:00"),
                    pd.Timestamp("2025-01-01 12:00:00"),
                    pd.Timestamp("2025-01-01 12:00:01"),
                ],
                "Pressure": [95.0, 99.0, 1.0],
            }
        )

        cleaned, new_ref = align_and_clean_datetime(numerical)

        # groupby().first() picks the first row in the group (95.0)
        # when both have non-NaN values
        self.assertEqual(new_ref, datetime(2025, 1, 1, 12, 0, 0))
        self.assertEqual(cleaned["Pressure"].tolist(), [95.0, 1.0])

    def test_align_and_clean_datetime_rejects_missing_or_invalid_signal(self) -> None:
        with self.assertRaisesRegex(ValueError, "No usable signal"):
            align_and_clean_datetime(
                pd.DataFrame({"DateTime": [pd.Timestamp("2025-01-01")]}),
            )

        with self.assertRaisesRegex(ValueError, "No usable signal"):
            align_and_clean_datetime(
                pd.DataFrame(
                    {
                        "DateTime": [pd.Timestamp("2025-01-01")],
                        "Pressure": [np.nan],
                    }
                ),
            )

    def test_align_and_clean_datetime_preserves_earlier_signal_without_pressure(
        self,
    ) -> None:
        """Test the concrete bug scenario: Activity has valid data starting at row 50,
        Temp at row 100, no Pressure column. The anchor should be determined by the
        earliest first_valid_index across all signals, not by priority."""
        # Create data with Activity starting at row 50, Temp at row 100, no Pressure
        timestamps = pd.date_range("2025-01-01", periods=150, freq="s")
        activity_values = [np.nan] * 50 + list(range(100))  # valid from row 50
        temp_values = [np.nan] * 100 + [
            36.0 + i * 0.01 for i in range(50)
        ]  # valid from row 100

        numerical = pd.DataFrame(
            {
                "DateTime": timestamps,
                "Temp": temp_values,
                "Activity": activity_values,
            }
        )

        cleaned, new_ref = align_and_clean_datetime(numerical)

        # The trim point should be at row 50 (where Activity becomes valid),
        # not row 100 (where Temp becomes valid)
        # Row 50 has timestamp "2025-01-01 00:00:50" (50 seconds after midnight)
        self.assertEqual(new_ref, pd.Timestamp("2025-01-01 00:00:50").to_pydatetime())
        self.assertEqual(len(cleaned), 100)  # rows 50-149 = 100 rows
        # First Activity value should be 0 (was row 50, now row 0 after trim)
        self.assertEqual(cleaned["Activity"].iloc[0], 0.0)
        # Activity at row 50 should be 50 (originally at row 100)
        self.assertEqual(cleaned["Activity"].iloc[50], 50.0)
        # First Temp should be NaN at row 0, valid from row 50 onward
        self.assertTrue(np.isnan(cleaned["Temp"].iloc[0]))
        self.assertEqual(cleaned["Temp"].iloc[50], 36.0)

    def test_align_and_clean_datetime_detects_and_sorts_backwards_jumps(
        self,
    ) -> None:
        """Test that genuinely out-of-order timestamps (backwards jumps without
        duplicates) are detected, logged, and sorted to ensure monotonicity for
        interpolation and gap detection."""
        # Data with a backwards jump: forward, then backwards, then forward again
        numerical = pd.DataFrame(
            {
                "DateTime": [
                    pd.Timestamp("2025-01-01 12:00:00"),
                    pd.Timestamp("2025-01-01 12:00:02"),
                    pd.Timestamp("2025-01-01 12:00:01"),  # backwards jump
                    pd.Timestamp("2025-01-01 12:00:03"),
                ],
                "Pressure": [1.0, 2.0, 1.5, 3.0],
            }
        )

        with patch("src.core.telemetry_processing.log_warning") as mock_warn:
            cleaned, new_ref = align_and_clean_datetime(numerical)

            # Assert that log_warning was called with a message about backwards jumps
            calls = mock_warn.call_args_list
            self.assertTrue(
                any("backwards" in str(call) for call in calls),
                f"Expected 'backwards' in log_warning calls, got: {calls}",
            )

        # Assert that the output is sorted ascending by DateTime
        self.assertTrue(cleaned["DateTime"].is_monotonic_increasing)
        # Assert that the order is now correct
        self.assertEqual(
            cleaned["DateTime"].tolist(),
            [
                pd.Timestamp("2025-01-01 12:00:00"),
                pd.Timestamp("2025-01-01 12:00:01"),
                pd.Timestamp("2025-01-01 12:00:02"),
                pd.Timestamp("2025-01-01 12:00:03"),
            ],
        )
        # Verify the reference timestamp is the earliest (first after sorting)
        self.assertEqual(new_ref, pd.Timestamp("2025-01-01 12:00:00").to_pydatetime())

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
            "src.core.telemetry_processing.preprocess_pressure_data",
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
        time = np.arange(0.0, 4.0, 0.002)
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
        inner = processed.loc[
            processed["TimeSinceReference"].between(1.0, 1.0), "SmoothedPressure"
        ]
        self.assertFalse(inner.isna().any())
        self.assertTrue(processed["SmoothedPressure"].iloc[0:100].isna().all())

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

    def test_safe_interpolate_does_not_extrapolate_signal(self) -> None:
        source = pd.DataFrame(
            {"TimeSinceReference": [10.0, 20.0], "Temp": [36.0, 38.0]}
        )

        interpolated = safe_interpolate(
            source, pd.Series([0.0, 10.0, 15.0, 20.0, 30.0]), "Temp"
        )

        np.testing.assert_allclose(
            interpolated["Temp"], [np.nan, 36.0, 37.0, 38.0, np.nan], equal_nan=True
        )

    def test_adjust_behaviours_offsets_times_and_retains_negative_windows(self) -> None:
        behaviour_data = {
            "Sleep": [
                (1, -5.0, 10.0, 15.0),
                (2, 10.0, 20.0, 10.0),
            ]
        }

        adjusted = adjust_behaviours(behaviour_data, total_time_diff=-8.0)

        self.assertEqual(
            adjusted["Sleep"],
            [(1, -13.0, 2.0, 15.0), (2, 2.0, 12.0, 10.0)],
        )
        self.assertEqual(behaviour_data["Sleep"][1], (2, 10.0, 20.0, 10.0))

    def test_continuous_signal_coverage_splits_long_pressure_outages(self) -> None:
        numerical = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 0.5, 1.0, 2.5, 3.0, 4.5],
                "Pressure": [1.0, np.nan, 1.1, np.nan, 1.2, 1.3],
            }
        )

        intervals = continuous_signal_coverage(
            numerical, "Pressure", max_gap_seconds=1.0
        )

        self.assertEqual(intervals, [(0.0, 1.0), (3.0, 3.0), (4.5, 4.5)])

    def test_trim_coverage_intervals_reserves_filter_edges(self) -> None:
        self.assertEqual(
            trim_coverage_intervals([(0.0, 10.0), (20.0, 21.0)], margin_seconds=1.0),
            [(1.0, 9.0)],
        )

    def test_preprocess_pressure_data_does_not_filter_across_long_gap(self) -> None:
        first_time = np.arange(0.0, 3.0, 0.002)
        second_time = np.arange(6.0, 9.0, 0.002)
        time = np.concatenate([first_time, second_time])
        pressure = 1.0 + 0.1 * np.sin(2.0 * np.pi * 2.0 * time)

        processed = preprocess_pressure_data(
            pd.DataFrame({"TimeSinceReference": time, "Pressure": pressure}),
            pressure_sample_rate=500.0,
        )

        edge = processed["TimeSinceReference"].between(6.0, 6.9)
        interior = processed["TimeSinceReference"].between(7.0, 7.9)
        self.assertTrue(processed.loc[edge, "SmoothedPressure"].isna().all())
        self.assertFalse(processed.loc[interior, "SmoothedPressure"].isna().any())

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

    def test_prepare_raw_data_does_not_extrapolate_telemetry(self) -> None:
        photometry = pd.DataFrame(
            {
                "TimeSinceReference": [90.0, 100.0, 110.0, 120.0, 130.0],
                "dFoF_465": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        temp = pd.DataFrame(
            {
                "TimeSinceReference": [100.0, 120.0],
                "Temp": [36.0, 38.0],
            }
        )

        raw = prepare_raw_data(photometry, temp, None, injection_sec=110.0)

        np.testing.assert_allclose(
            raw["Temp"], [np.nan, 36.0, 37.0, 38.0, np.nan], equal_nan=True
        )

    def test_require_signal_frame_requires_pressure_and_normalizes_optional_data(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "Pressure data is required"):
            require_signal_frame({}, "Pressure")

        self.assertTrue(require_signal_frame({"Temp": None}, "Temp").empty)
        self.assertTrue(
            require_signal_frame({"Activity": pd.DataFrame()}, "Activity").empty
        )

        pressure_df = pd.DataFrame({"Pressure": [1.0]})
        self.assertTrue(
            require_signal_frame({"Pressure": pressure_df}, "Pressure").equals(
                pressure_df
            )
        )


if __name__ == "__main__":
    unittest.main()
