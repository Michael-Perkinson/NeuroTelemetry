from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.gui.analysis_worker as worker_module
from src.controllers.photometry_controller import PhotometryPipelineResult
from src.gui.analysis_worker import (
    AnalysisFailure,
    AnalysisSuccess,
    AnalysisWorker,
    BehaviourAnalysisRequest,
    PhotometryAnalysisRequest,
)


def test_photometry_worker_emits_logs_and_success(
    monkeypatch,
    local_tmpdir: Path,
) -> None:
    telemetry_path = local_tmpdir / "telemetry.csv"
    photometry_path = local_tmpdir / "photometry.csv"
    photometry_df = pd.DataFrame({"dFoF_465": [1.0]})
    telemetry_df = pd.DataFrame({"Pressure": [1.0]})
    expected_result: PhotometryPipelineResult = {
        "summary": {},
        "per_peak_df": pd.DataFrame(),
        "signal_binned": pd.DataFrame(),
        "peaks_binned": pd.DataFrame(),
        "analysis_folder": local_tmpdir,
        "excel_path": local_tmpdir / "analysis.xlsx",
        "config_path": local_tmpdir / "config.json",
    }
    pipeline_arguments: dict[str, object] = {}

    monkeypatch.setattr(
        worker_module,
        "load_photometry_data",
        lambda _photometry_path, _telemetry_path: (photometry_df, telemetry_df),
    )

    def run_pipeline(**kwargs):
        pipeline_arguments.update(kwargs)
        return expected_result

    monkeypatch.setattr(worker_module, "run_photometry_pipeline", run_pipeline)
    request = PhotometryAnalysisRequest(
        mode="photometry",
        telemetry_path=telemetry_path,
        secondary_path=photometry_path,
        photometry_start_time="01/01/2025 12:00:00 PM",
        injection_sec=10,
        pre_min=20,
        post_min=30,
        bin_min=5,
    )
    worker = AnalysisWorker(request)
    logs: list[str] = []
    successes: list[AnalysisSuccess] = []
    failures: list[AnalysisFailure] = []
    finished: list[bool] = []
    worker.log_message.connect(logs.append)
    worker.succeeded.connect(successes.append)
    worker.failed.connect(failures.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert logs[:2] == [
        "Loading telemetry and photometry...",
        "Running photometry analysis...",
    ]
    assert pipeline_arguments["telemetry_path"] == telemetry_path
    assert pipeline_arguments["photometry_start_time"] == request.photometry_start_time
    assert successes == [AnalysisSuccess(result=expected_result)]
    assert failures == []
    assert finished == [True]


def test_behaviour_worker_emits_typed_failure_and_finishes(
    monkeypatch,
    local_tmpdir: Path,
) -> None:
    def fail_loading(_telemetry_path: Path, _secondary_path: Path):
        raise RuntimeError("load failed")

    monkeypatch.setattr(worker_module, "load_pressure_data", fail_loading)
    worker = AnalysisWorker(
        BehaviourAnalysisRequest(
            mode="behaviour",
            telemetry_path=local_tmpdir / "telemetry.csv",
            secondary_path=local_tmpdir / "behaviour.csv",
            behaviour_to_plot="sleep",
            video_time="01/01/2025 12:00:00 PM",
            bin_size_sec=10,
            atm_bin_size_sec=300,
            export_atm_summary=True,
        )
    )
    failures: list[AnalysisFailure] = []
    successes: list[AnalysisSuccess] = []
    finished: list[bool] = []
    worker.failed.connect(failures.append)
    worker.succeeded.connect(successes.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert successes == []
    assert len(failures) == 1
    assert failures[0].message == "load failed"
    assert "RuntimeError: load failed" in failures[0].traceback_text
    assert finished == [True]
