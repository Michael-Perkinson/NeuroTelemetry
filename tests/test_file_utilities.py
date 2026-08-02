from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.file_utilities import (
    create_folders_for_graphs,
    detect_data_type,
    format_directory_listing,
)


class TestFileHandling(unittest.TestCase):
    def test_format_directory_listing_reports_matching_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "a.csv").write_text("", encoding="utf-8")
            (path / "b.txt").write_text("", encoding="utf-8")
            (path / "c.xlsx").write_text("", encoding="utf-8")

            output = format_directory_listing(path, ["csv", "txt"])

        self.assertIn("a.csv", output)
        self.assertIn("b.txt", output)
        self.assertNotIn("c.xlsx", output)

    def test_format_directory_listing_reports_missing_directory(self) -> None:
        output = format_directory_listing(Path("does-not-exist-for-unit-test"))

        self.assertIn("Directory does not exist", output)

    def test_create_folders_for_graphs_creates_expected_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "analysis.csv"

            html, svg, full_trace, file_base = create_folders_for_graphs(data_path)

            self.assertEqual(file_base, "analysis")
            self.assertTrue(html.is_dir())
            self.assertTrue(svg.is_dir())
            self.assertTrue(full_trace.is_dir())

    def test_detect_data_type_identifies_behaviour_photometry_and_unknown(self) -> None:
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

            self.assertEqual(detect_data_type(behaviour), "behaviour")
            self.assertEqual(detect_data_type(photometry), "photometry")
            self.assertEqual(detect_data_type(unknown), "unknown")


if __name__ == "__main__":
    unittest.main()
