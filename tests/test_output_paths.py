from pathlib import Path

from src.core import output_paths
from src.core.output_paths import (
    analysis_output_location,
    choose_analysis_root_if_needed,
    output_path_exceeds_platform_budget,
)


def test_short_paths_keep_outputs_beside_recording(
    monkeypatch,
) -> None:
    monkeypatch.setattr(output_paths, "_windows_path_budget", lambda: 240)
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
    monkeypatch.setattr(output_paths, "_windows_path_budget", lambda: 240)
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
    monkeypatch.setattr(output_paths, "_windows_path_budget", lambda: 240)
    recording = Path("C:/data") / f"{'r' * 90}.ascii"
    location = analysis_output_location(recording, "PhotometryAnalysis", Path("C:/u"))

    # The fixed-size reserve alone would accept this folder, but the full-trace
    # filename repeats the recording stem and would exceed the Windows budget.
    assert len(str(location.folder)) + 1 + output_paths.OUTPUT_FILENAME_BUDGET <= 240
    assert output_path_exceeds_platform_budget(location)


def test_relative_output_root_uses_absolute_path_length(monkeypatch) -> None:
    monkeypatch.setattr(output_paths, "_windows_path_budget", lambda: 240)
    location = analysis_output_location(Path("recording.ascii"), "A", Path("r" * 70))

    assert len(str(location.folder)) + 1 + output_paths.OUTPUT_FILENAME_BUDGET <= 240
    assert output_path_exceeds_platform_budget(location)


def test_gui_reprompts_when_selected_folder_is_still_too_long(
    monkeypatch,
) -> None:
    monkeypatch.setattr(output_paths, "_windows_path_budget", lambda: 240)
    choices = iter([str(Path("C:/") / ("long" * 70)), "C:/u/Results"])
    warning_calls = []
    recording = Path(
        "X:/Teo/Fever/Telemetry/Ponemah Recordings/export"
        "/2026-10-22 Vgat photom/2026-10-22 Vgat Photom"
        "/2026-10-22 Vgat Photom_2026-10-28_09-00-34 vp1.ascii"
    )

    selected_root, cancelled = choose_analysis_root_if_needed(
        recording,
        "PhotometryAnalysis",
        lambda: next(choices),
        lambda location: warning_calls.append(location),
    )

    assert selected_root == Path("C:/u/Results")
    assert not cancelled
    assert len(warning_calls) == 1


def test_long_path_can_be_cancelled_before_analysis(monkeypatch) -> None:
    monkeypatch.setattr(output_paths, "_windows_path_budget", lambda: 240)
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
