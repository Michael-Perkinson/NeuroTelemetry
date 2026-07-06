from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd


try:
    import plotly.graph_objects  # noqa: F401
except ModuleNotFoundError:
    plotly_module = types.ModuleType("plotly")
    graph_objects_module = types.ModuleType("plotly.graph_objects")

    class _FakeTrace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakeLayout:
        def __init__(self):
            self.shapes = []

    class _FakeFigure:
        def __init__(self):
            self.data = []
            self.layout = _FakeLayout()

        def add_trace(self, trace):
            self.data.append(trace)

        def add_vrect(self, **kwargs):
            self.layout.shapes.append(kwargs)

        def update_layout(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self.layout, key, value)

        def write_html(self, path):
            Path(path).write_text("<html></html>", encoding="utf-8")

    graph_objects_module.Figure = _FakeFigure
    graph_objects_module.Scatter = _FakeTrace
    graph_objects_module.Bar = _FakeTrace
    plotly_module.graph_objects = graph_objects_module
    sys.modules["plotly"] = plotly_module
    sys.modules["plotly.graph_objects"] = graph_objects_module

from src.core.export_graphs import (
    create_interactive_plot,
    create_static_plot,
    export_behavior_images_interactive,
    export_full_time_range_plot,
    filter_df_by_time,
    filter_times_to_range,
    prepare_peaks_for_range,
)


def _main_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TimeSinceReference": [0.0, 60.0, 120.0],
            "SmoothedPressure": [1.0, 3.0, 2.0],
        }
    )


def _temp_df() -> pd.DataFrame:
    return pd.DataFrame({"TimeSinceReference": [0.0, 60.0], "Temp": [36.0, 37.0]})


def _activity_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"TimeSinceReference": [0.0, 30.0, 60.0], "Activity": [1.0, 3.0, 5.0]}
    )


class TestExportGraphHelpers(unittest.TestCase):
    def test_filter_df_by_time_is_inclusive(self) -> None:
        filtered = filter_df_by_time(_main_df(), 60.0, 120.0)

        self.assertEqual(filtered["TimeSinceReference"].tolist(), [60.0, 120.0])

    def test_filter_times_to_range_is_inclusive(self) -> None:
        self.assertEqual(
            filter_times_to_range([-1.0, 0.0, 1.0, 2.0], 0.0, 1.0),
            [0.0, 1.0],
        )

    def test_prepare_peaks_for_range_trims_to_matching_lengths(self) -> None:
        peaks, pre_peaks = prepare_peaks_for_range(
            [0.0, 1.0, 2.0],
            [0.0, 1.0],
            0.0,
            2.0,
        )

        self.assertEqual(peaks, [0.0, 1.0])
        self.assertEqual(pre_peaks, [0.0, 1.0])

    def test_prepare_peaks_for_range_keeps_empty_side_as_is(self) -> None:
        peaks, pre_peaks = prepare_peaks_for_range([0.0, 1.0], [], 0.0, 1.0)

        self.assertEqual(peaks, [0.0, 1.0])
        self.assertEqual(pre_peaks, [])

    def test_create_interactive_plot_adds_expected_traces_and_shapes(self) -> None:
        atm = pd.DataFrame(
            {
                "TimeSinceReference": [0.0, 60.0, 120.0],
                "AtmPressure": [100.0, 101.0, 102.0],
            }
        )

        fig = create_interactive_plot(
            _main_df(),
            _temp_df(),
            _activity_df(),
            peak_times=[60.0],
            pre_peak_times=[120.0],
            title="Title",
            main_signal_col="SmoothedPressure",
            main_signal_label="Pressure",
            atm_pressure_data=atm,
            behavior_windows=[(0.0, 60.0)],
        )

        trace_names = [trace.name for trace in fig.data]
        self.assertIn("Pressure", trace_names)
        self.assertIn("Temperature", trace_names)
        self.assertIn("Activity (1-min bins)", trace_names)
        self.assertIn("Peaks", trace_names)
        self.assertIn("Pre-Peaks", trace_names)
        self.assertIn("Atm. Pressure", trace_names)
        self.assertEqual(len(fig.layout.shapes), 1)

    def test_create_static_plot_writes_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "plot.svg"

            create_static_plot(
                _main_df(),
                _temp_df(),
                _activity_df(),
                peak_times=[60.0],
                pre_peak_times=[120.0],
                save_path=str(save_path),
                title="Title",
                main_signal_col="SmoothedPressure",
                main_signal_label="Pressure",
                atm_pressure_data=pd.DataFrame(
                    {
                        "TimeSinceReference": [0.0, 60.0],
                        "AtmPressure": [100.0, 101.0],
                    }
                ),
                behavior_windows=[(0.0, 60.0)],
            )

            self.assertTrue(save_path.exists())
            self.assertIn("<svg", save_path.read_text(encoding="utf-8"))


class TestExportGraphEntrypoints(unittest.TestCase):
    def test_export_full_time_range_plot_writes_html_and_svg(self) -> None:
        fake_fig = Mock()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "src.core.export_graphs.create_interactive_plot",
                return_value=fake_fig,
            ),
            patch("src.core.export_graphs.create_static_plot") as static_plot,
        ):
            export_full_time_range_plot(
                _main_df(),
                _temp_df(),
                _activity_df(),
                valid_peak_times_all=[60.0],
                valid_pre_peak_times_all=[120.0],
                min_time=0.0,
                max_time=120.0,
                behaviour_to_plot="Sleep",
                full_trace_folder=tmp,
                file_base="file",
                main_signal_col="SmoothedPressure",
                main_signal_label="Pressure",
            )

        fake_fig.write_html.assert_called_once()
        static_plot.assert_called_once()

    def test_export_behavior_images_interactive_skips_empty_segments(self) -> None:
        fake_fig = Mock()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "src.core.export_graphs.create_interactive_plot",
                return_value=fake_fig,
            ),
            patch("src.core.export_graphs.create_static_plot") as static_plot,
        ):
            export_behavior_images_interactive(
                time_windows=[(0.0, 120.0), (500.0, 600.0)],
                pressure_data=_main_df(),
                temp_data=_temp_df(),
                activity_data=_activity_df(),
                valid_peak_times_all=[60.0],
                valid_pre_peak_times_all=[120.0],
                behaviour_to_plot="Sleep",
                html_save_folder=tmp,
                svg_save_folder=tmp,
                file_base="file",
                main_signal_col="SmoothedPressure",
                main_signal_label="Pressure",
            )

        fake_fig.write_html.assert_called_once()
        static_plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
