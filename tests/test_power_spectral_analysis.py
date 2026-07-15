from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal.windows import hamming

from src.core.power_spectral_analysis import (
    DEFAULT_PSD_NFFT,
    DEFAULT_PSD_OVERLAP_FRACTION,
    DEFAULT_PSD_RESAMPLE_HZ,
    DEFAULT_PSD_SEGMENT_SECONDS,
    DEFAULT_PSD_SMOOTH_NPERSEG,
    _build_ttot_trace,
    _compute_welch_psd,
    _iter_segment_bounds,
    _metrics_table,
    _resample_segment,
    _trim_period_ends,
    analyze_ttot_psd,
    plot_psd_results,
)


def _make_trace_df(duration_s: float, interval_s: float = 1.0) -> pd.DataFrame:
    times: np.ndarray = np.arange(0.0, duration_s + interval_s, interval_s, dtype=float)
    ttot = 1.0 + 0.1 * np.sin(2.0 * np.pi * 0.2 * times)
    return pd.DataFrame({"TimeOfBreath_s": times, "Ttot_s": ttot})


def _make_period(duration_s: float, interval_s: float = 1.0) -> dict[str, object]:
    peak_times = pd.Series(np.arange(0.0, duration_s + interval_s, interval_s))
    return {
        "name": f"Period_0_{duration_s}",
        "start_time": 0.0,
        "end_time": duration_s,
        "peak_times": peak_times,
        "trough_times": peak_times.iloc[:-1],
        "peak_indices": peak_times,
        "trough_indices": peak_times.iloc[:-1],
    }


def test_segments_shorter_than_configured_duration_are_discarded() -> None:
    assert (
        _iter_segment_bounds(
            0.0,
            DEFAULT_PSD_SEGMENT_SECONDS - 0.1,
            DEFAULT_PSD_SEGMENT_SECONDS,
        )
        == []
    )


def test_long_periods_split_into_contiguous_configured_segments() -> None:
    bounds = _iter_segment_bounds(
        0.0,
        3 * DEFAULT_PSD_SEGMENT_SECONDS,
        DEFAULT_PSD_SEGMENT_SECONDS,
    )

    assert bounds == [
        (0.0, DEFAULT_PSD_SEGMENT_SECONDS),
        (DEFAULT_PSD_SEGMENT_SECONDS, 2 * DEFAULT_PSD_SEGMENT_SECONDS),
        (2 * DEFAULT_PSD_SEGMENT_SECONDS, 3 * DEFAULT_PSD_SEGMENT_SECONDS),
    ]


def test_build_ttot_trace_drops_invalid_and_nonmonotonic_peaks() -> None:
    period = {
        "peak_times": pd.Series([0.0, 1.0, 1.0, np.inf, 3.0, 2.0, 4.0]),
    }

    trace_df = _build_ttot_trace(period)

    assert trace_df.to_dict("list") == {
        "TimeOfBreath_s": [0.0, 1.0, 2.0, 3.0],
        "Ttot_s": [1.0, 1.0, 1.0, 1.0],
    }


def test_trim_period_ends_removes_configured_edge_samples() -> None:
    trace_df = _make_trace_df(duration_s=20.0, interval_s=1.0)

    trimmed = _trim_period_ends(trace_df, n_samples=3)

    assert trimmed["TimeOfBreath_s"].tolist() == list(np.arange(3.0, 18.0))


def test_resampled_segments_have_expected_length_and_columns() -> None:
    trace_df = _make_trace_df(duration_s=180.0, interval_s=1.0)

    resampled = _resample_segment(
        trace_df=trace_df,
        segment_start=0.0,
        segment_seconds=DEFAULT_PSD_SEGMENT_SECONDS,
        resample_hz=DEFAULT_PSD_RESAMPLE_HZ,
    )

    assert resampled is not None
    assert resampled.columns.tolist() == [
        "SegmentTime_s",
        "AbsoluteTime_s",
        "Ttot_s",
    ]
    assert len(resampled) == int(DEFAULT_PSD_SEGMENT_SECONDS * DEFAULT_PSD_RESAMPLE_HZ)
    assert resampled["Ttot_s"].isna().sum() == 0
    assert resampled["SegmentTime_s"].iloc[0] == 0.0
    assert np.isclose(
        resampled["SegmentTime_s"].iloc[-1],
        DEFAULT_PSD_SEGMENT_SECONDS - (1.0 / DEFAULT_PSD_RESAMPLE_HZ),
    )


def test_resample_segment_rejects_insufficient_boundary_coverage() -> None:
    trace_df = _make_trace_df(duration_s=40.0, interval_s=1.0)

    assert (
        _resample_segment(
            trace_df=trace_df,
            segment_start=0.0,
            segment_seconds=DEFAULT_PSD_SEGMENT_SECONDS,
            resample_hz=DEFAULT_PSD_RESAMPLE_HZ,
        )
        is None
    )


def test_resample_segment_rejects_nonphysical_cubic_spline_output() -> None:
    trace_df = pd.DataFrame(
        {
            "TimeOfBreath_s": [0.0, 1.0, 1.1, 2.1],
            "Ttot_s": [1.0, 0.1, 1.0, 0.1],
        }
    )

    assert (
        _resample_segment(
            trace_df=trace_df,
            segment_start=0.0,
            segment_seconds=1.1,
            resample_hz=10.0,
        )
        is None
    )


def test_welch_psd_recovers_expected_modulation_frequency() -> None:
    time_s: np.ndarray = np.arange(
        0.0,
        DEFAULT_PSD_SEGMENT_SECONDS,
        1.0 / DEFAULT_PSD_RESAMPLE_HZ,
        dtype=float,
    )
    modulation_hz = 0.2
    signal = np.sin(2.0 * np.pi * modulation_hz * time_s)

    freq_hz, psd = _compute_welch_psd(
        signal=signal,
        resample_hz=DEFAULT_PSD_RESAMPLE_HZ,
        smooth_nperseg=DEFAULT_PSD_SMOOTH_NPERSEG,
        overlap_fraction=DEFAULT_PSD_OVERLAP_FRACTION,
        nfft=DEFAULT_PSD_NFFT,
    )

    peak_freq = float(freq_hz[int(np.argmax(psd))])
    assert np.isclose(peak_freq, modulation_hz, atol=0.03)


def test_welch_psd_uses_symmetric_hamming_configuration() -> None:
    signal = np.linspace(0.0, 1.0, 500) + np.sin(
        2.0 * np.pi * 0.2 * np.arange(500) / DEFAULT_PSD_RESAMPLE_HZ
    )

    freq_hz, psd = _compute_welch_psd(
        signal=signal,
        resample_hz=DEFAULT_PSD_RESAMPLE_HZ,
        smooth_nperseg=100,
        overlap_fraction=0.5,
        nfft=2048,
    )
    centered = signal - signal.mean()
    expected_freq, expected_psd = welch(
        centered,
        fs=DEFAULT_PSD_RESAMPLE_HZ,
        window=hamming(100, sym=True),
        nperseg=100,
        noverlap=50,
        nfft=2048,
        detrend=False,
        scaling="density",
    )

    assert np.array_equal(freq_hz, expected_freq)
    assert np.allclose(psd, expected_psd)

    _, comparison_psd = _compute_welch_psd(
        signal=signal,
        resample_hz=DEFAULT_PSD_RESAMPLE_HZ,
        smooth_nperseg=167,
        overlap_fraction=0.5,
        nfft=2048,
    )
    assert not np.allclose(psd, comparison_psd)


def test_window_metrics_warn_when_fmax_is_below_nominal_resolution() -> None:
    metrics = _metrics_table(
        {"window": {"fmax": 0.02, "PSDmax": 1.0, "AUC": 0.5, "n_segments": 1}},
        resample_hz=10.0,
        smooth_nperseg=100,
    )

    assert metrics["NominalWelchResolution_Hz"].tolist() == [0.1]
    assert "below nominal Welch resolution" in metrics["QC_Warning"].iloc[0]


def test_analyze_ttot_psd_exports_expected_outputs(local_tmpdir: Path) -> None:
    window_periods = {
        "0.0-180.0": [_make_period(duration_s=180.0, interval_s=1.0)],
    }
    output_folder = local_tmpdir / "PSD"

    results = analyze_ttot_psd(
        window_periods=window_periods,
        output_folder=output_folder,
        log=lambda _msg: None,
    )

    assert len(results["segment_summary"]) == 3
    assert not results["per_segment_psd"].empty
    assert not results["comparison_per_segment_psd"].empty
    assert not results["per_window_psd"].empty
    assert not results["comparison_per_window_psd"].empty
    assert not results["pooled_psd"].empty
    assert not results["resampled_traces"].empty
    assert results["window_metrics"]["n_segments"].tolist() == [3]
    assert results["config"]["WelchComparisonNperseg"] == 167
    assert results["segment_summary"].columns[0] == "AnalysisID"
    assert results["rejection_log"].columns[0] == "AnalysisID"

    expected_files = {
        "ttot_psd_rejected.csv",
        "ttot_psd_segment_summary.csv",
        "ttot_psd_per_segment.csv",
        "ttot_psd_comparison_per_segment.csv",
        "ttot_psd_per_window.csv",
        "ttot_psd_comparison_per_window.csv",
        "ttot_psd_pooled.csv",
        "ttot_resampled_segments.csv",
        "ttot_psd_window_metrics.csv",
        "ttot_psd_summary.xlsx",
    }
    assert expected_files.issubset({p.name for p in output_folder.iterdir()})
    with pd.ExcelFile(output_folder / "ttot_psd_summary.xlsx") as workbook:
        assert "Comparison PSD" in workbook.sheet_names


def test_analyze_ttot_psd_records_rejections_for_short_periods(
    local_tmpdir: Path,
) -> None:
    window_periods = {
        "0.0-40.0": [_make_period(duration_s=40.0, interval_s=1.0)],
    }

    results = analyze_ttot_psd(
        window_periods=window_periods,
        output_folder=local_tmpdir / "PSD",
        log=lambda _msg: None,
    )

    assert results["segment_summary"].empty
    assert results["per_segment_psd"].empty
    assert results["per_window_psd"].empty
    assert results["pooled_psd"].empty
    assert len(results["rejection_log"]) == 1
    assert "period too short" in results["rejection_log"]["RejectionReason"].iloc[0]
    for filename in [
        "ttot_psd_segment_summary.csv",
        "ttot_psd_per_segment.csv",
        "ttot_psd_comparison_per_segment.csv",
        "ttot_psd_per_window.csv",
        "ttot_psd_comparison_per_window.csv",
        "ttot_psd_pooled.csv",
        "ttot_resampled_segments.csv",
    ]:
        assert (local_tmpdir / "PSD" / filename).exists()
    assert results["segment_summary"].columns[0] == "AnalysisID"
    assert results["rejection_log"].columns[0] == "AnalysisID"


def test_empty_plot_overwrites_stale_summary_images(local_tmpdir: Path) -> None:
    output_folder = local_tmpdir / "PSD"
    output_folder.mkdir()
    png_path = output_folder / "ttot_psd_summary.png"
    svg_path = output_folder / "ttot_psd_summary.svg"
    png_path.write_text("stale")
    svg_path.write_text("stale")

    plot_psd_results(
        {"per_window_psd": pd.DataFrame(), "per_window_metrics": {}},
        output_folder=output_folder,
        log=lambda _msg: None,
    )

    assert png_path.read_bytes() != b"stale"
    assert svg_path.read_text(encoding="utf-8") != "stale"
