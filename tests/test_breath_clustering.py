from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.respiratory.breath_clustering import (
    BreathClusteringConfig,
    BreathClusteringResult,
    BreathDataset,
    BreathMorphologyModel,
    cluster_breath_dataset,
    cluster_breaths,
    export_breath_clustering,
    extract_breath_dataset,
    fit_reference_breath_model,
    load_breath_morphology_model,
    plot_breath_cluster_waveforms,
    predict_breath_dataset,
)


def _pressure_data(
    *,
    duration: float = 12.0,
    sample_rate: float = 20.0,
    gap: tuple[float, float] | None = None,
) -> pd.DataFrame:
    time = np.arange(0.0, duration + 1.0 / sample_rate, 1.0 / sample_rate)
    phase = np.mod(time, 2.0 * np.pi)
    pressure = 10.0 + np.sin(phase)
    data = pd.DataFrame({"TimeSinceReference": time, "SmoothedPressure": pressure})
    if gap is not None:
        data = data[~data["TimeSinceReference"].between(gap[0], gap[1])].reset_index(
            drop=True
        )
    return data


def _periods(peak_times: list[float]) -> dict[str, list[dict[str, object]]]:
    return {
        "sleep": [
            {
                "name": "Period_1",
                "start_time": peak_times[0],
                "end_time": peak_times[-1],
                "peak_times": pd.Series(peak_times),
                "trough_times": pd.Series(peak_times[:-1]),
            }
        ]
    }


def test_extracts_time_normalised_trough_to_trough_breaths() -> None:
    dataset = extract_breath_dataset(
        _periods([1.0, 3.0, 5.0]),
        _pressure_data(),
        waveform_points=16,
    )

    assert len(dataset.records) == 2
    assert dataset.clusterable_indices == (0, 1)
    assert dataset.waveforms.shape == (2, 16)
    assert np.isfinite(dataset.waveforms).all()
    assert np.all((dataset.waveforms >= 0.0) & (dataset.waveforms <= 1.0))
    assert dataset.records["duration_s"].tolist() == [2.0, 2.0]
    assert (dataset.records["amplitude"] > 0).all()


def test_rejects_breaths_that_cross_a_signal_gap() -> None:
    dataset = extract_breath_dataset(
        _periods([1.0, 3.0, 5.0]),
        _pressure_data(gap=(2.0, 2.5)),
        max_gap_seconds=0.2,
    )

    assert len(dataset.records) == 2
    assert dataset.records["drop_reason"].iloc[0] == "signal_gap"
    assert pd.isna(dataset.records["drop_reason"].iloc[1])
    assert dataset.clusterable_indices == (1,)


def test_retains_but_excludes_duration_outliers_from_shape_fitting() -> None:
    dataset = extract_breath_dataset(
        _periods([1.0, 3.0, 5.0, 9.0]),
        _pressure_data(duration=10.0),
        waveform_points=16,
    )

    assert len(dataset.records) == 3
    assert dataset.records["drop_reason"].isna().tolist() == [True, True, False]
    assert dataset.records["drop_reason"].iloc[2] == "segmentation_outlier"
    assert dataset.clusterable_indices == (0, 1)


def test_small_recording_returns_insufficient_status_without_fitting() -> None:
    result = cluster_breaths(
        _periods([1.0, 3.0, 5.0]),
        _pressure_data(),
        config=BreathClusteringConfig(waveform_points=16),
    )

    assert result.metadata["status"] == "insufficient_data"
    assert (result.breaths["cluster_label_shape"] == -2).all()
    assert result.period_summary.loc[0, "status"] == "no_clusterable_breaths"


def test_marks_only_excluded_periods_in_exported_flag() -> None:
    records = pd.DataFrame(
        {
            "window_key": ["sleep"] * 41,
            "period_name": ["Clean"] * 20 + ["Noisy"] * 21,
            "drop_reason": [None] * 40 + ["segmentation_outlier"],
        }
    )
    clean_waveform = np.linspace(0.0, 1.0, 16)
    noisy_waveform = np.vstack(
        [clean_waveform + (0.5 if index % 2 else 0.0) for index in range(20)]
    )
    dataset = BreathDataset(
        records=records,
        waveforms=np.vstack([np.tile(clean_waveform, (20, 1)), noisy_waveform]),
        clusterable_indices=tuple(range(40)),
    )

    result = cluster_breath_dataset(
        dataset,
        config=BreathClusteringConfig(
            waveform_points=16,
            max_clusters=2,
            preferred_clusters=2,
            min_cluster_sizes=(5,),
            min_samples=(3,),
            stability_repeats=0,
        ),
    )

    excluded = result.breaths["period_excluded_from_fit"]
    assert not excluded.iloc[:20].any()
    assert excluded.iloc[20:].all()
    assert result.metadata["excluded_periods_for_fit"] == ["Noisy"]


def test_reports_periods_with_no_clusterable_breaths() -> None:
    records = pd.DataFrame(
        {
            "window_key": ["sleep"] * 2,
            "period_name": ["Dropped"] * 2,
            "drop_reason": ["signal_gap"] * 2,
        }
    )
    dataset = BreathDataset(
        records=records,
        waveforms=np.empty((0, 16)),
        clusterable_indices=(),
    )

    result = cluster_breath_dataset(
        dataset,
        config=BreathClusteringConfig(waveform_points=16),
    )

    assert result.period_summary.loc[0, "status"] == "no_clusterable_breaths"
    distribution = result.period_distribution
    assert set(distribution["method"]) == {"shape", "shape_amplitude"}
    assert set(distribution["proportion_status"]) == {"no_clusterable_breaths"}
    assert (distribution["count"] == 0).all()
    assert distribution["fraction"].isna().all()


def test_period_summary_marks_small_periods_even_when_recording_fit_is_possible() -> (
    None
):
    peak_times = [float(value) for value in range(1, 43, 2)]
    result = cluster_breaths(
        _periods(peak_times),
        _pressure_data(duration=45.0),
        config=BreathClusteringConfig(waveform_points=16),
    )

    assert len(result.period_summary) == 1
    shape_distribution = result.period_distribution.query("method == 'shape'")
    assert set(shape_distribution["cluster_label"]) <= set(
        result.breaths["cluster_label_shape"]
    )
    assert result.period_summary.loc[0, "n_breaths"] == 20
    assert result.period_summary.loc[0, "status"] in {
        "ok",
        "flagged_noise",
        "unusable_noise",
        "no_clusterable_breaths",
    }


def test_small_period_proportions_are_marked_insufficient() -> None:
    large_peaks = [float(value) for value in range(10, 90, 2)]
    periods = {
        "sleep": [
            {
                "name": "Short_Period",
                "start_time": 1.0,
                "end_time": 5.0,
                "peak_times": pd.Series([1.0, 3.0, 5.0]),
            },
            {
                "name": "Long_Period",
                "start_time": large_peaks[0],
                "end_time": large_peaks[-1],
                "peak_times": pd.Series(large_peaks),
            },
        ]
    }
    result = cluster_breaths(
        periods,
        _pressure_data(duration=90.0),
        config=BreathClusteringConfig(
            waveform_points=16,
            amplitude_sensitivity=False,
            min_cluster_sizes=(10, 20),
            min_samples=(5, 10),
        ),
    )

    short_distribution = result.period_distribution.query(
        "period_name == 'Short_Period'"
    )
    assert not short_distribution.empty
    assert set(short_distribution["proportion_status"]) == {"insufficient_breaths"}
    assert short_distribution["fraction"].isna().all()


def test_finds_stable_morphology_groups_in_synthetic_breaths() -> None:
    sample_rate = 20.0
    time = np.arange(0.0, 180.0, 1.0 / sample_rate)
    phase = np.mod(time, 2.0) / 2.0
    breath_number = np.floor(time / 2.0).astype(int)
    pattern = breath_number % 3
    shapes = np.stack(
        [
            -np.cos(2.0 * np.pi * phase),
            -np.cos(2.0 * np.pi * phase) + 0.35 * np.sin(2.0 * np.pi * phase),
            -np.cos(2.0 * np.pi * phase) + 0.35 * np.sin(4.0 * np.pi * phase),
        ]
    )
    pressure = shapes[pattern, np.arange(len(time))]
    pressure_data = pd.DataFrame(
        {"TimeSinceReference": time, "SmoothedPressure": pressure}
    )

    result = cluster_breaths(
        _periods([float(value) for value in range(0, 180, 2)]),
        pressure_data,
        config=BreathClusteringConfig(
            waveform_points=32,
            amplitude_sensitivity=False,
            min_cluster_sizes=(10, 20),
            min_samples=(5, 10),
        ),
    )

    labels = result.breaths["cluster_label_shape"]
    real_labels = labels[labels >= 0]
    assert result.metadata["status"] in {"ok", "flagged_noise", "unusable_noise"}
    assert 2 <= result.metadata["cluster_count"] <= 8
    assert real_labels.nunique() >= 2
    assert "method" in result.period_distribution.columns
    assert result.metadata["stability_validation"]["requested_repeats"] == 10
    assert result.metadata["stability_validation"]["failed_repeats"] == 0
    assert result.metadata["stability_validation"]["sample_size"] < len(real_labels)


def test_reference_model_round_trip_and_assignment(tmp_path) -> None:
    records = pd.DataFrame(
        {
            "breath_id": [0, 1, 2, 3],
            "window_key": ["sleep"] * 4,
            "period_name": ["Period_1"] * 4,
            "start_time": [0.0, 1.0, 2.0, 3.0],
            "end_time": [1.0, 2.0, 3.0, 4.0],
            "duration_s": [1.0] * 4,
            "amplitude": [1.0] * 4,
            "drop_reason": [None] * 4,
            "cluster_label_shape": [10, 20, 10, 20],
            "cluster_label_shape_amplitude": [-2] * 4,
        }
    )
    dataset = BreathDataset(
        records=records,
        waveforms=np.asarray(
            [
                [0.0, 0.2, 0.0, 0.2],
                [1.0, 0.8, 1.0, 0.8],
                [0.0, 0.1, 0.0, 0.1],
                [1.0, 0.9, 1.0, 0.9],
            ]
        ),
        clusterable_indices=(0, 1, 2, 3),
    )
    model = BreathMorphologyModel(
        waveform_points=4,
        labels=(10, 20),
        prototypes=np.asarray([[0.0, 0.15, 0.0, 0.15], [1.0, 0.85, 1.0, 0.85]]),
        distance_threshold=0.3,
    )
    result = BreathClusteringResult(
        breaths=records,
        period_summary=pd.DataFrame(),
        period_distribution=pd.DataFrame(),
        metadata={"status": "ok"},
        model=model,
        dataset=dataset,
    )

    paths = export_breath_clustering(result, tmp_path)
    loaded = load_breath_morphology_model(paths["model_npz"])
    assigned = predict_breath_dataset(dataset, loaded)

    assert paths["model_npz"].is_file()
    assert paths["model_json"].is_file()
    assert paths["metadata"].is_file()
    assert assigned["cluster_label_shape_reference"].tolist() == [10, 20, 10, 20]

    plot_path = plot_breath_cluster_waveforms(result, tmp_path)
    assert plot_path is not None
    assert plot_path.is_file()


def test_reference_fit_can_be_applied_to_multiple_recordings() -> None:
    config = BreathClusteringConfig(
        waveform_points=16,
        min_period_breaths=20,
        amplitude_sensitivity=False,
        min_cluster_sizes=(5, 10),
        min_samples=(3, 5),
    )
    shape_a = np.linspace(0.0, 1.0, 16)
    shape_b = np.linspace(1.0, 0.0, 16)

    def dataset() -> BreathDataset:
        waveforms = np.vstack(
            [shape_a + index * 0.001 for index in range(20)]
            + [shape_b - index * 0.001 for index in range(20)]
        )
        records = pd.DataFrame(
            {
                "breath_id": range(40),
                "window_key": ["sleep"] * 40,
                "period_name": ["Period_1"] * 40,
                "start_time": np.arange(40, dtype=float),
                "end_time": np.arange(1, 41, dtype=float),
                "duration_s": [1.0] * 40,
                "amplitude": [1.0] * 40,
                "drop_reason": [None] * 40,
            }
        )
        return BreathDataset(records, waveforms, tuple(range(40)))

    first = dataset()
    second = dataset()

    model = fit_reference_breath_model([first, second], config=config)

    assert model is not None
    assigned = predict_breath_dataset(second, model)
    assert (assigned["cluster_label_shape_reference"] >= -1).all()
