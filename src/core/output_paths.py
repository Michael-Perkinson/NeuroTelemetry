"""Helpers for checking platform-sensitive analysis output paths."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

WINDOWS_PATH_BUDGET = 240
OUTPUT_FILENAME_BUDGET = 120
MAX_TRACE_TIME_TOKEN = "999999999999.99"


@dataclass(frozen=True)
class AnalysisOutputLocation:
    folder: Path
    file_stem: str


def _windows_path_budget() -> int | None:
    return WINDOWS_PATH_BUDGET if os.name == "nt" else None


def analysis_output_location(
    recording_path: Path,
    suffix: str,
    analysis_root: Path | None = None,
) -> AnalysisOutputLocation:
    """Return the normal adjacent output folder or one under an explicit root."""
    file_stem = recording_path.stem
    output_root = analysis_root or (recording_path.parent / "extracted_data")
    return AnalysisOutputLocation(output_root / f"{file_stem}_{suffix}", file_stem)


def output_path_exceeds_platform_budget(location: AnalysisOutputLocation) -> bool:
    """Conservatively detect whether generated files may exceed Windows MAX_PATH."""
    budget = _windows_path_budget()
    if budget is None:
        return False

    absolute_folder = location.folder.absolute()
    full_trace_path = (
        absolute_folder
        / "full_trace"
        / f"{location.file_stem}_full_trace_-{MAX_TRACE_TIME_TOKEN}_to_"
        f"{MAX_TRACE_TIME_TOKEN}.html"
    )
    other_output_path = absolute_folder / ("x" * OUTPUT_FILENAME_BUDGET)
    return max(len(str(full_trace_path)), len(str(other_output_path))) > budget


def choose_analysis_root_if_needed(
    recording_path: Path,
    suffix: str,
    choose_folder: Callable[[], str | None],
    warn_too_long: Callable[[AnalysisOutputLocation], None],
) -> tuple[Path | None, bool]:
    """Ask for a parent folder only when default paths exceed the budget.

    Returns ``(None, False)`` when the default location is safe, otherwise
    ``(selected_root, False)`` or ``(None, True)`` when the user cancels.
    Invalid selections are reported and retried before any output is written.
    """
    default_location = analysis_output_location(recording_path, suffix)
    if not output_path_exceeds_platform_budget(default_location):
        return None, False

    while True:
        selected_folder = choose_folder()
        if not selected_folder:
            return None, True

        analysis_root = Path(selected_folder)
        location = analysis_output_location(recording_path, suffix, analysis_root)
        if not output_path_exceeds_platform_budget(location):
            return analysis_root, False
        warn_too_long(location)
