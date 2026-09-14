from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from PySide6.QtCore import QObject, Signal, Slot

from src.controllers.photometry_controller import (
    PhotometryPipelineResult,
    load_photometry_data,
    run_photometry_pipeline,
)
from src.controllers.pressure_controller import (
    PressurePipelineResult,
    load_pressure_data,
    run_pressure_pipeline,
)


@dataclass(frozen=True)
class PhotometryAnalysisRequest:
    mode: Literal["photometry"]
    telemetry_path: Path
    secondary_path: Path
    photometry_start_time: str
    injection_sec: int
    pre_min: int
    post_min: int
    bin_min: int


@dataclass(frozen=True)
class BehaviourAnalysisRequest:
    mode: Literal["behaviour"]
    telemetry_path: Path
    secondary_path: Path
    behaviour_to_plot: str
    video_time: str
    bin_size_sec: int
    atm_bin_size_sec: int
    export_atm_summary: bool


AnalysisRequest: TypeAlias = PhotometryAnalysisRequest | BehaviourAnalysisRequest
PipelineResult: TypeAlias = PhotometryPipelineResult | PressurePipelineResult


@dataclass(frozen=True)
class AnalysisSuccess:
    result: PipelineResult | None


@dataclass(frozen=True)
class AnalysisFailure:
    message: str
    traceback_text: str


def execute_analysis(
    request: AnalysisRequest,
    log_callback: Callable[[str], None],
) -> PipelineResult | None:
    """Load inputs and run the requested pipeline outside the GUI layer."""
    if request.mode == "photometry":
        log_callback("Loading telemetry and photometry...")
        photometry_df, telemetry_df = load_photometry_data(
            request.secondary_path,
            request.telemetry_path,
        )
        log_callback("Running photometry analysis...")
        return run_photometry_pipeline(
            telemetry_df=telemetry_df,
            photometry_df=photometry_df,
            photometry_start_time=request.photometry_start_time,
            injection_sec=request.injection_sec,
            pre_min=request.pre_min,
            post_min=request.post_min,
            bin_min=request.bin_min,
            telemetry_path=request.telemetry_path,
            log_callback=log_callback,
        )

    log_callback("Loading telemetry and behaviour data...")
    telemetry_df, behaviour_df = load_pressure_data(
        request.telemetry_path,
        request.secondary_path,
    )
    log_callback("Running behaviour analysis...")
    return run_pressure_pipeline(
        telemetry_df=telemetry_df,
        event_df=behaviour_df,
        behaviour_to_plot=request.behaviour_to_plot,
        video_time=request.video_time,
        bin_size_sec=request.bin_size_sec,
        telemetry_path=request.telemetry_path,
        atm_bin_size_sec=request.atm_bin_size_sec,
        export_atm_summary=request.export_atm_summary,
        log_callback=log_callback,
        event_path=request.secondary_path,
    )


class AnalysisWorker(QObject):
    log_message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self, request: AnalysisRequest) -> None:
        super().__init__()
        self._request = request

    @Slot()
    def run(self) -> None:
        try:
            result = execute_analysis(self._request, self.log_message.emit)
        except Exception as exc:
            self.failed.emit(
                AnalysisFailure(
                    message=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            )
        else:
            self.succeeded.emit(AnalysisSuccess(result=result))
        finally:
            self.finished.emit()
