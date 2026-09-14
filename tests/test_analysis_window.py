from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

import src.gui.analysis_window as analysis_window_module  # noqa: E402
import src.gui.analysis_worker as analysis_worker_module  # noqa: E402
from src.gui.analysis_window import AnalysisWindow  # noqa: E402


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def window(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[AnalysisWindow]:
    class MemorySettings:
        values: dict[str, object] = {}

        def __init__(self, *_args: object) -> None:
            pass

        def value(self, key: str, default: object = None, **_kwargs: object) -> object:
            return self.values.get(key, default)

        def setValue(self, key: str, value: object) -> None:
            self.values[key] = value

        def clear(self) -> None:
            self.values.clear()

    monkeypatch.setattr(analysis_window_module, "QSettings", MemorySettings)
    monkeypatch.setattr(AnalysisWindow, "restore_settings", lambda self: None)
    monkeypatch.setattr(AnalysisWindow, "save_settings", lambda self: None)
    widget = AnalysisWindow()
    yield widget
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


def test_run_button_tracks_both_paths_and_recognized_secondary_type(
    window: AnalysisWindow,
    local_tmpdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = local_tmpdir / "telemetry.csv"
    behaviour = local_tmpdir / "behaviour.csv"
    unknown = local_tmpdir / "unknown.csv"
    telemetry.write_text("time,value\n0,1\n", encoding="utf-8")
    behaviour.write_text("start,stop\n0,1\n", encoding="utf-8")
    unknown.write_text("other,value\n0,1\n", encoding="utf-8")

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message),
    )

    assert not window.run_button.isEnabled()

    window.secondary_path.setText(str(behaviour))
    assert window.file_mode == "behaviour"
    assert not window.stack.isHidden()
    assert not window.run_button.isEnabled()

    window.telemetry_path.setText(str(telemetry))
    assert window.run_button.isEnabled()

    window.telemetry_path.setText(str(local_tmpdir / "missing.csv"))
    assert not window.run_button.isEnabled()

    window.telemetry_path.setText(str(telemetry))
    window.secondary_path.setText(str(unknown))
    assert window.file_mode == "unknown"
    assert window.stack.isHidden()
    assert not window.run_button.isEnabled()
    assert len(warnings) == 1

    window.telemetry_path.setText(str(local_tmpdir / "also-missing.csv"))
    assert not window.run_button.isEnabled()
    assert len(warnings) == 1

    window.secondary_path.setText(str(local_tmpdir / "missing-secondary.csv"))
    assert window.file_mode == "unknown"
    assert not window.run_button.isEnabled()


def test_selecting_secondary_file_handles_change_once(
    window: AnalysisWindow,
    local_tmpdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = local_tmpdir / "telemetry.csv"
    secondary = local_tmpdir / "behaviour.csv"
    telemetry.touch()
    secondary.touch()
    window.telemetry_path.setText(str(telemetry))

    detected_paths: list[Path] = []

    def detect_once(path: Path) -> str:
        detected_paths.append(path)
        return "behaviour"

    monkeypatch.setattr(analysis_window_module, "detect_data_type", detect_once)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(secondary), "CSV Files (*.csv)"),
    )

    window.select_file(window.secondary_path)

    assert detected_paths == [secondary]
    assert window.run_button.isEnabled()


def test_cancelled_file_dialog_preserves_existing_selection(
    window: AnalysisWindow,
    local_tmpdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secondary = local_tmpdir / "behaviour.csv"
    secondary.touch()
    monkeypatch.setattr(
        analysis_window_module, "detect_data_type", lambda _path: "behaviour"
    )
    window.secondary_path.setText(str(secondary))
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    window.select_file(window.secondary_path)

    assert window.secondary_path.text() == str(secondary)
    assert window.file_mode == "behaviour"


def test_analysis_runs_on_worker_thread_and_revalidates_button(
    window: AnalysisWindow,
    qt_app: QApplication,
    local_tmpdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = local_tmpdir / "telemetry.csv"
    behaviour = local_tmpdir / "behaviour.csv"
    telemetry.touch()
    behaviour.write_text("start,stop\n0,1\n", encoding="utf-8")
    ran_off_gui_thread: list[bool] = []
    gui_thread = qt_app.thread()

    def execute(request, log_callback):
        ran_off_gui_thread.append(gui_thread is not QThread.currentThread())
        log_callback("worker log")
        return {"analysis_folder": local_tmpdir}

    monkeypatch.setattr(analysis_worker_module, "execute_analysis", execute)
    window.telemetry_path.setText(str(telemetry))
    window.secondary_path.setText(str(behaviour))

    window.run_analysis()

    assert not window.run_button.isEnabled()
    deadline = time.monotonic() + 3.0
    while window._analysis_thread is not None and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)

    assert window._analysis_thread is None
    assert ran_off_gui_thread == [True]
    assert window.run_button.isEnabled()
    assert "worker log" in window.log_box.toPlainText()
    assert "Analysis complete." in window.log_box.toPlainText()
