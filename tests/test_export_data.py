from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.core.export_data import (
    build_export_data,
    create_summary_data,
    export_data_to_excel,
    insert_blank_rows,
)


class TestExportData(unittest.TestCase):
    def test_export_data_to_excel_propagates_write_failure(self) -> None:
        with (
            patch(
                "src.core.export_data.pd.ExcelWriter",
                side_effect=PermissionError("workbook is locked"),
            ),
            self.assertRaisesRegex(PermissionError, "locked"),
        ):
            export_data_to_excel([], {}, Path("unused"))

    def test_create_summary_data_counts_peaks_inside_valid_periods(self) -> None:
        summary = create_summary_data(
            valid_peak_times_all=[1.0, 2.0, 3.0, 12.0],
            updated_valid_periods=[(0.0, 4.0)],
            time_windows=[(0.0, 10.0)],
        )

        self.assertEqual(summary[0]["Description"], "Overall Time Window")
        self.assertEqual(summary[0]["Duration (s)"], 10.0)
        self.assertEqual(summary[1]["Description"], "Valid Period")
        self.assertEqual(summary[1]["Number of Peaks"], 3)
        self.assertEqual(summary[1]["Peaks per Minute"], 45.0)

    def test_insert_blank_rows_separates_group_changes(self) -> None:
        first = pd.DataFrame({"Period": ["A"], "Value": [1]})
        second = pd.DataFrame({"Period": ["B"], "Value": [2]})

        combined = insert_blank_rows([first, second], "Period")

        self.assertEqual(len(combined), 3)
        self.assertEqual(combined.loc[0, "Period"], "A")
        self.assertEqual(combined.loc[1, "Period"], "")
        self.assertEqual(combined.loc[2, "Period"], "B")

    def test_insert_blank_rows_returns_empty_frame_for_empty_inputs(self) -> None:
        self.assertTrue(insert_blank_rows([pd.DataFrame()], "Period").empty)

    def test_build_export_data_shapes_per_bin_period_and_window_tables(self) -> None:
        all_metrics = {
            "0.0-10.0": {
                "WindowSummary": {"T_I_mean": 0.5},
                "Periods": {
                    "Period_1": {
                        "Binned": pd.DataFrame(
                            {
                                "Bin_Start": [0.0],
                                "Bin_End": [5.0],
                                "T_I_mean": [0.4],
                            }
                        ),
                        "Summary": {"T_I_mean": 0.45},
                    }
                },
            },
            "GlobalSummary": {"T_I_mean": 0.5},
        }

        per_bin, per_period, per_window = build_export_data(all_metrics)

        self.assertEqual(per_bin.loc[0, "Window"], "0.0-10.0")
        self.assertEqual(per_bin.loc[0, "Period"], "Period_1")
        self.assertEqual(float(per_bin.loc[0, "T_I_mean"]), 0.4)
        self.assertEqual(per_period.loc[0, "Period"], "Period_1")
        self.assertEqual(float(per_period.loc[0, "T_I_mean"]), 0.45)
        self.assertEqual(per_window.loc[0, "Window"], "0.0-10.0")
        self.assertEqual(float(per_window.loc[0, "T_I_mean"]), 0.5)


if __name__ == "__main__":
    unittest.main()
