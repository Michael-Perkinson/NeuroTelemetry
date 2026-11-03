import os
from pathlib import Path

from src.core import output_paths
from src.core.output_paths import (
    analysis_output_location,
    choose_analysis_root_if_needed,
    output_path_exceeds_platform_budget,
)


def _simulate_windows_path_budget(monkeypatch) -> None:
    """Simulate Windows drive-rooted paths even when tests run on POSIX."""
    monkeypatch.setattr(output_paths, "_windows_path_budget", lambda: 240)
    if os.name != "nt":
        absolute = Path.absolute

        def absolute_with_windows_drive(path: Path) -> Path:
            if path.parts and path.parts[0].endswith(":"):
                return path
            return absolute(path)

        monkeypatch.setattr(Path, "absolute", absolute_with_windows_drive)


def test_short_paths_keep_outputs_beside_recording(
    monkeypatch,
) -> None:
    _simulate_windows_path_budget(monkeypatch)
    recording = Path("C:/input/recording.ascii")

    location = analysis_output_location(recording, "PhotometryAnalysis")

    assert location.folder == Path(
        "C:/input/extracted_data/recording_PhotometryAnalysis"
    )
    assert location.file_stem == "recording"
    assert not output_path_exceeds_platform_budget(location)


def test_reported_windows_path_requires_a_shorter_user_selected_root(
    monkeypatch,
) -> None:
    _simulate_windows_path_budget(monkeypatch)
    recording = Path(
        "X:/Teo/Fever/Telemetry/Ponemah Recordings/export"
        "/2026-10-22 Vgat photom/2026-10-22 Vgat Photom"
        "/2026-10-22 Vgat Photom_2026-10-28_09-00-34 vp1.ascii"
    )

    default_location = analysis_output_location(recording, "PhotometryAnalysis")
    safe_location = analysis_output_location(
        recording, "PhotometryAnalysis", Path("C:/u/Analysis")
    )
    long_selected_location = analysis_output_location(
        recording,
        "PhotometryAnalysis",
        Path("C:/") / ("long" * 70),
    )

    assert output_path_exceeds_platform_budget(default_location)
    assert not output_path_exceeds_platform_budget(safe_location)
    assert output_path_exceeds_platform_budget(long_selected_location)
    generated_html = (
        safe_location.folder
        / "full_trace"
        / f"{safe_location.file_stem}_full_trace_9480.00_to_32570.80.html"
    )
    assert len(str(generated_html)) <= output_paths.WINDOWS_PATH_BUDGET


def test_long_stem_is_included_in_full_trace_path_budget(monkeypatch) -> None:
    _simulate_windows_path_budget(monkeypatch)
    recording = Path("C:/data") / f"{'r' * 90}.ascii"
    location = analysis_output_location(recording, "PhotometryAnalysis", Path("C:/u"))

    # The fixed-size reserve alone would accept this folder, but the full-trace
    # filename repeats the recording stem and would exceed the Windows budget.
    assert len(str(location.folder)) + 1 + output_paths.OUTPUT_FILENAME_BUDGET <= 240
    assert output_path_exceeds_platform_budget(location)


def test_relative_output_root_uses_absolute_path_length(
    monkeypatch, tmp_path: Path
) -> None:
    _simulate_windows_path_budget(monkeypatch)
    monkeypatch.chdir(tmp_path)
    relative_folder_name = "a_A"
    root_length = (
        output_paths.WINDOWS_PATH_BUDGET
        - len(relative_folder_name)
        - output_paths.OUTPUT_FILENAME_BUDGET
        - 2
    )
    location = analysis_output_location(Path("a.ascii"), "A", Path("r" * root_length))

    assert (
        len(str(location.folder)) + 1 + output_paths.OUTPUT_FILENAME_BUDGET
        == output_paths.WINDOWS_PATH_BUDGET
    )
    assert output_path_exceeds_platform_budget(location)


def test_gui_reprompts_when_selected_folder_is_still_too_long(
    monkeypatch,
) -> None:
    _simulate_windows_path_budget(monkeypatch)
    choices = iter([str(Path("C:/") / ("long" * 70)), "C:/u/Results"])
    choice_calls = 0
    warning_calls = []
    recording = Path(
        "X:/Teo/Fever/Telemetry/Ponemah Recordings/export"
        "/2026-10-22 Vgat photom/2026-10-22 Vgat Photom"
        "/2026-10-22 Vgat Photom_2026-10-28_09-00-34 vp1.ascii"
    )

    def choose_folder() -> str:
        nonlocal choice_calls
        choice_calls += 1
        return next(choices, "")

    selected_root, cancelled = choose_analysis_root_if_needed(
        recording,
        "PhotometryAnalysis",
        choose_folder,
        lambda location: warning_calls.append(location),
    )

    assert selected_root == Path("C:/u/Results")
    assert not cancelled
    assert choice_calls == 2
    assert len(warning_calls) == 1


def test_long_path_can_be_cancelled_before_analysis(monkeypatch) -> None:
    _simulate_windows_path_budget(monkeypatch)
    recording = Path(
        "X:/Teo/Fever/Telemetry/Ponemah Recordings/export"
        "/2026-10-22 Vgat photom/2026-10-22 Vgat Photom"
        "/2026-10-22 Vgat Photom_2026-10-28_09-00-34 vp1.ascii"
    )

    selected_root, cancelled = choose_analysis_root_if_needed(
        recording, "PhotometryAnalysis", lambda: "", lambda location: None
    )

    assert selected_root is None
    assert cancelled
