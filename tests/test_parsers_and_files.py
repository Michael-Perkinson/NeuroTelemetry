from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.core.data_file_parser import (
    detect_skip_rows,
    read_and_process_photometry_file,
    safe_get_df,
)
from src.core.event_file_parser import (
    read_and_process_event_file,
    select_time_windows,
    structure_behaviour_events,
)
from src.core.file_handling import (
    create_folders_for_graphs,
    detect_file_type,
    list_files,
)


class TestDataFileParser(unittest.TestCase):
    def test_read_and_process_photometry_file_converts_minutes_to_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photometry.csv"
            path.write_text(
                "# t_min,dFoF_465,dFoF_405,Z_465,Ignored\n"
                "0,1.0,10.0,100.0,a\n"
                "0.5,bad,11.0,101.0,b\n"
                "1.0,2.0,12.0,102.0,c\n",
                encoding="utf-8",
            )

            df = read_and_process_photometry_file(path)

        self.assertEqual(
            df.columns.tolist(),
            ["TimeSinceReference", "dFoF_465", "dFoF_405", "Z_465"],
        )
        self.assertEqual(df["TimeSinceReference"].tolist(), [0.0, 60.0])
        self.assertEqual(df["dFoF_465"].tolist(), [1.0, 2.0])

    def test_read_and_process_photometry_file_requires_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photometry.csv"
            path.write_text("time,dFoF_465\n0,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "# t_min"):
                read_and_process_photometry_file(path)

    def test_detect_skip_rows_finds_time_header_case_insensitively(self) -> None:
        data = pd.DataFrame({0: ["metadata", "  Time,Pressure", "0,1"]})

        self.assertEqual(detect_skip_rows(data), 1)

    def test_safe_get_df_requires_pressure(self) -> None:
        with self.assertRaisesRegex(ValueError, "Pressure data is required"):
            safe_get_df({}, "Pressure")

        self.assertTrue(safe_get_df({}, "Temp").empty)
        non_empty = pd.DataFrame({"x": [1]})
        self.assertIs(safe_get_df({"Temp": non_empty}, "Temp"), non_empty)


class TestEventFileParser(unittest.TestCase):
    def test_read_and_process_event_file_instances_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            path.write_text(
                "Behavior,Start (s),Stop (s),Duration (s)\n"
                "Sleep,10,45,35\n"
                "Run,1,2,1\n"
                "Sleep,50,90,40\n",
                encoding="utf-8",
            )

            df = read_and_process_event_file(path)

        self.assertEqual(
            df.columns.tolist(),
            ["event", "instance", "start", "end", "duration"],
        )
        self.assertEqual(df["event"].tolist(), ["Run", "Sleep", "Sleep"])
        self.assertEqual(df["instance"].tolist(), [1, 1, 2])

    def test_read_and_process_event_file_requires_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            path.write_text(
                "Behavior,Start (s),Duration (s)\nSleep,0,60\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Missing column"):
                read_and_process_event_file(path)

    def test_structure_behaviour_events_groups_rows_by_event(self) -> None:
        event_df = pd.DataFrame(
            {
                "event": ["Sleep", "Sleep", "Run"],
                "instance": [1, 2, 1],
                "start": [10.0, 30.0, 50.0],
                "end": [20.0, 40.0, 60.0],
                "duration": [10.0, 10.0, 10.0],
            }
        )

        grouped = structure_behaviour_events(event_df)

        self.assertEqual(
            grouped["Sleep"],
            [(1, 10.0, 20.0, 10.0), (2, 30.0, 40.0, 10.0)],
        )
        self.assertEqual(grouped["Run"], [(1, 50.0, 60.0, 10.0)])

    def test_select_time_windows_filters_duration_cutoff_and_name(self) -> None:
        reference = pd.Timestamp("2025-01-01 00:00:00")
        base = reference.timestamp()
        behaviour_data = {
            "Sleep": [
                (1, base + 10.0, base + 50.0, 40.0),
                (2, base + 100.0, base + 120.0, 20.0),
                (3, base + 91.0 * 60.0, base + 92.0 * 60.0, 60.0),
            ],
            "Run": [(1, base + 5.0, base + 65.0, 60.0)],
        }

        windows = select_time_windows("Sleep", behaviour_data, reference)

        self.assertEqual(windows, [(base + 10.0, base + 50.0)])
        self.assertEqual(select_time_windows("Missing", behaviour_data, reference), [])


class TestFileHandling(unittest.TestCase):
    def test_list_files_reports_matching_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "a.csv").write_text("", encoding="utf-8")
            (path / "b.txt").write_text("", encoding="utf-8")
            (path / "c.xlsx").write_text("", encoding="utf-8")

            output = list_files(path, ["csv", "txt"])

        self.assertIn("a.csv", output)
        self.assertIn("b.txt", output)
        self.assertNotIn("c.xlsx", output)

    def test_list_files_reports_missing_directory(self) -> None:
        output = list_files(Path("does-not-exist-for-unit-test"))

        self.assertIn("Directory does not exist", output)

    def test_create_folders_for_graphs_creates_expected_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "analysis.csv"

            html, svg, full_trace, file_base = create_folders_for_graphs(data_path)

            self.assertEqual(file_base, "analysis")
            self.assertTrue(html.is_dir())
            self.assertTrue(svg.is_dir())
            self.assertTrue(full_trace.is_dir())

    def test_detect_file_type_identifies_behaviour_photometry_and_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            behaviour = root / "behaviour.csv"
            photometry = root / "photometry.csv"
            unknown = root / "unknown.csv"
            behaviour.write_text(
                "Start (s),Stop (s),Behavior\n0,1,Sleep\n",
                encoding="utf-8",
            )
            photometry.write_text("# t_min,dFoF_465\n0,1\n", encoding="utf-8")
            unknown.write_text("A,B\n1,2\n", encoding="utf-8")

            self.assertEqual(detect_file_type(behaviour), "behaviour")
            self.assertEqual(detect_file_type(photometry), "photometry")
            self.assertEqual(detect_file_type(unknown), "unknown")


if __name__ == "__main__":
    unittest.main()
