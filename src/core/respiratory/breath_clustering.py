"""Breath waveform morphology clustering.

This module is deliberately downstream of peak and period detection. It treats
each consecutive pair of detected inspiration minima as one breath, clusters
time-normalised waveform shape, and retains amplitude and duration as
descriptive metadata rather than letting them silently dominate the primary
shape labels.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreathClusteringConfig:
    """Configuration for one recording-level morphology fit."""

    waveform_points: int = 128
    min_period_breaths: int = 20
    preferred_clusters: int = 3
    max_clusters: int = 8
    min_cluster_fraction: float = 0.05
    duration_ratio_bounds: tuple[float, float] = (0.6, 1.6)
    noise_flag_fraction: float = 0.30
    noise_unusable_fraction: float = 0.50
    max_gap_seconds: float = 1.0
    amplitude_sensitivity: bool = True
    amplitude_weight: float = 2.0
    min_cluster_sizes: tuple[int, ...] = (10, 20, 30, 50)
    min_samples: tuple[int, ...] = (5, 10, 15)
    stability_repeats: int = 10
    stability_sample_fraction: float = 0.8
    stability_max_samples: int = 1500
    fit_max_samples: int = 2500
    selection_stability_weight: float = 0.5
    exclude_noisy_periods: bool = True
    noisy_period_sd_ratio: float = 1.5

    def __post_init__(self) -> None:
        if self.waveform_points < 4:
            raise ValueError("waveform_points must be at least 4")
        if self.min_period_breaths < 1:
            raise ValueError("min_period_breaths must be positive")
        if self.preferred_clusters < 2 or self.max_clusters < 2:
            raise ValueError("cluster counts must be at least 2")
        if self.preferred_clusters > self.max_clusters:
            raise ValueError("preferred_clusters cannot exceed max_clusters")
        if not 0.0 < self.min_cluster_fraction <= 0.5:
            raise ValueError("min_cluster_fraction must be in (0, 0.5]")
        if (
            len(self.duration_ratio_bounds) != 2
            or not 0.0 < self.duration_ratio_bounds[0] < 1.0
            or self.duration_ratio_bounds[1] <= 1.0
        ):
            raise ValueError("duration_ratio_bounds must contain sensible limits")
        if not 0.0 <= self.noise_flag_fraction <= 1.0:
            raise ValueError("noise_flag_fraction must be between 0 and 1")
        if not 0.0 <= self.noise_unusable_fraction <= 1.0:
            raise ValueError("noise_unusable_fraction must be between 0 and 1")
        if self.noise_unusable_fraction < self.noise_flag_fraction:
            raise ValueError("unusable noise threshold must not be lower than flag")
        if self.max_gap_seconds <= 0 or self.amplitude_weight < 0:
            raise ValueError("max_gap_seconds and amplitude_weight must be valid")
        if self.stability_repeats < 0 or self.stability_max_samples < 3:
            raise ValueError("stability settings must be valid")
        if not 0.0 < self.stability_sample_fraction <= 1.0:
            raise ValueError("stability_sample_fraction must be in (0, 1]")
        if self.fit_max_samples < 20:
            raise ValueError("fit_max_samples must be at least 20")
        if self.selection_stability_weight < 0:
            raise ValueError("selection_stability_weight must be non-negative")
        if self.noisy_period_sd_ratio <= 1.0:
            raise ValueError("noisy_period_sd_ratio must be greater than 1.0")


@dataclass(frozen=True)
class BreathDataset:
    """Extracted breaths and the waveform matrix for clusterable rows."""

    records: pd.DataFrame
    waveforms: np.ndarray
    clusterable_indices: tuple[int, ...]


@dataclass(frozen=True)
class BreathClusteringResult:
    """Labels, per-period summaries, and fit diagnostics."""

    breaths: pd.DataFrame
    period_summary: pd.DataFrame
    period_distribution: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    model: BreathMorphologyModel | None = None
    dataset: BreathDataset | None = None


@dataclass(frozen=True)
class BreathMorphologyModel:
    """Safe, prediction-ready representation of recording-level prototypes."""

    waveform_points: int
    labels: tuple[int, ...]
    prototypes: np.ndarray
    distance_threshold: float
    schema_version: int = 1
    algorithm: str = "unknown"
    distance_thresholds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        prototypes = np.asarray(self.prototypes, dtype=float)
        if prototypes.ndim != 2:
            raise ValueError("prototypes must be a two-dimensional array")
        if prototypes.shape != (len(self.labels), self.waveform_points):
            raise ValueError("prototype shape does not match labels and waveform size")
        if not np.isfinite(prototypes).all():
            raise ValueError("prototypes must contain only finite values")
        if self.distance_threshold <= 0 or not np.isfinite(self.distance_threshold):
            raise ValueError("distance_threshold must be finite and positive")
        thresholds = tuple(float(value) for value in self.distance_thresholds)
        if not thresholds:
            thresholds = (float(self.distance_threshold),) * len(self.labels)
            object.__setattr__(self, "distance_thresholds", thresholds)
        if len(thresholds) != len(self.labels):
            raise ValueError("distance_thresholds must match labels")
        if not all(np.isfinite(value) and value > 0 for value in thresholds):
            raise ValueError("distance_thresholds must be finite and positive")


def _finite_sorted_unique(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return np.unique(array)


def _prepare_signal(pressure_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"TimeSinceReference", "SmoothedPressure"}
    missing = required.difference(pressure_data.columns)
    if missing:
        raise ValueError(f"Pressure data is missing columns: {sorted(missing)}")

    time = np.asarray(pressure_data["TimeSinceReference"], dtype=float)
    pressure = np.asarray(pressure_data["SmoothedPressure"], dtype=float)
    finite = np.isfinite(time) & np.isfinite(pressure)
    time = time[finite]
    pressure = pressure[finite]
    if time.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(time, kind="stable")
    time = time[order]
    pressure = pressure[order]
    keep = np.concatenate(([True], np.diff(time) > 0))
    return time[keep], pressure[keep]


def _normalise_waveform(
    segment_time: np.ndarray,
    segment_pressure: np.ndarray,
    waveform_points: int,
) -> tuple[np.ndarray, float] | None:
    if segment_time.size < 4 or not np.isfinite(segment_pressure).all():
        return None
    duration = float(segment_time[-1] - segment_time[0])
    if duration <= 0:
        return None

    quantiles = np.asarray(np.quantile(segment_pressure, [0.05, 0.95]))
    lower = float(quantiles[0])
    upper = float(quantiles[1])
    amplitude = upper - lower
    if not np.isfinite(amplitude) or amplitude <= 1e-9:
        return None

    phase = (segment_time - segment_time[0]) / duration
    target_phase = np.linspace(0.0, 1.0, waveform_points)
    waveform = np.asarray(np.interp(target_phase, phase, segment_pressure))
    waveform = np.clip((waveform - lower) / amplitude, 0.0, 1.0)
    if not np.isfinite(waveform).all():
        return None
    return waveform, amplitude


def extract_breath_dataset(
    window_periods: dict[str, list[dict[str, Any]]],
    pressure_data: pd.DataFrame,
    *,
    waveform_points: int = 128,
    max_gap_seconds: float = 1.0,
    duration_ratio_bounds: tuple[float, float] = (0.6, 1.6),
) -> BreathDataset:
    """Extract finite, time-normalised trough-to-trough breath waveforms.

    ``peak_times`` are the pressure minima produced by the current detector.
    The name is retained at this seam for compatibility with
    ``find_valid_periods``. Each consecutive pair defines one breath. Every
    detected interval is retained in ``records``; rows that cannot be safely
    represented have a non-null ``drop_reason`` and label ``-2`` later.
    """
    if waveform_points < 4:
        raise ValueError("waveform_points must be at least 4")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    if (
        len(duration_ratio_bounds) != 2
        or not 0.0 < duration_ratio_bounds[0] < 1.0
        or duration_ratio_bounds[1] <= 1.0
    ):
        raise ValueError("duration_ratio_bounds must contain sensible limits")

    time, pressure = _prepare_signal(pressure_data)
    records: list[dict[str, Any]] = []
    waveforms: list[np.ndarray] = []
    clusterable_indices: list[int] = []

    for window_key, periods in window_periods.items():
        for period in periods:
            peak_times = _finite_sorted_unique(period.get("peak_times", []))
            period_durations = np.diff(peak_times)
            period_median_duration = (
                float(np.median(period_durations)) if period_durations.size else np.nan
            )
            for breath_index in range(max(0, peak_times.size - 1)):
                start = float(peak_times[breath_index])
                end = float(peak_times[breath_index + 1])
                record: dict[str, Any] = {
                    "breath_id": len(records),
                    "window_key": str(window_key),
                    "period_name": str(period.get("name", "")),
                    "start_time": start,
                    "end_time": end,
                    "duration_s": end - start,
                    "amplitude": np.nan,
                    "drop_reason": None,
                }

                if time.size == 0 or end <= start:
                    record["drop_reason"] = "invalid_interval"
                    records.append(record)
                    continue

                if np.isfinite(period_median_duration) and (
                    end - start < period_median_duration * duration_ratio_bounds[0]
                    or end - start > period_median_duration * duration_ratio_bounds[1]
                ):
                    record["drop_reason"] = "segmentation_outlier"
                    records.append(record)
                    continue

                left = int(np.searchsorted(time, start, side="left"))
                right = int(np.searchsorted(time, end, side="right"))
                segment_time = time[left:right]
                segment_pressure = pressure[left:right]
                if segment_time.size < 4:
                    record["drop_reason"] = "too_few_samples"
                    records.append(record)
                    continue

                gaps = np.diff(segment_time)
                typical_step = float(np.median(gaps)) if gaps.size else 0.0
                if (
                    not np.isfinite(typical_step)
                    or typical_step <= 0
                    or np.any(gaps > max_gap_seconds)
                ):
                    record["drop_reason"] = "signal_gap"
                    records.append(record)
                    continue

                normalised = _normalise_waveform(
                    segment_time, segment_pressure, waveform_points
                )
                if normalised is None:
                    record["drop_reason"] = "degenerate_or_nonfinite"
                    records.append(record)
                    continue

                waveform, amplitude = normalised
                record["amplitude"] = amplitude
                records.append(record)
                clusterable_indices.append(len(records) - 1)
                waveforms.append(waveform)

    return BreathDataset(
        records=pd.DataFrame(records),
        waveforms=(
            np.asarray(waveforms, dtype=float)
            if waveforms
            else np.empty((0, waveform_points), dtype=float)
        ),
        clusterable_indices=tuple(clusterable_indices),
    )


@dataclass(frozen=True)
class _Candidate:
    algorithm: str
    labels: np.ndarray
    cluster_count: int
    noise_fraction: float
    separation: float
    persistence: float
    minimum_cluster_fraction: float = 1.0
    forced_partition: bool = False
    min_cluster_size: int | None = None
    min_samples: int | None = None
    backend: str = "unknown"
    stability: float = 0.0
    score: float = float("-inf")


def _fit_hdbscan(
    matrix: np.ndarray, min_cluster_size: int, min_samples: int
) -> tuple[np.ndarray, float, str]:
    """Fit HDBSCAN, preferring the project dependency with a sklearn fallback."""
    try:
        import hdbscan  # type: ignore[import-not-found]

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=False,
        )
        labels = np.asarray(model.fit_predict(matrix), dtype=int)
        persistence_values = np.asarray(model.cluster_persistence_, dtype=float)
        persistence = (
            float(np.mean(persistence_values)) if persistence_values.size else 0.0
        )
        return labels, persistence, "hdbscan"
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Breath clustering requires hdbscan or scikit-learn HDBSCAN."
            ) from exc
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=False,
        )
        labels = np.asarray(model.fit_predict(matrix), dtype=int)
        return labels, 0.0, "sklearn"


def _fit_kmeans(matrix: np.ndarray, cluster_count: int) -> np.ndarray:
    from sklearn.cluster import KMeans  # type: ignore[import-not-found]

    model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
    raw_labels = np.asarray(model.fit_predict(matrix), dtype=int)
    labels = raw_labels.copy()
    for label in np.unique(raw_labels):
        distances = np.linalg.norm(
            matrix[raw_labels == label] - model.cluster_centers_[label], axis=1
        )
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        threshold = max(float(np.quantile(distances, 0.99)), median + 3.0 * mad)
        cluster_indices = np.flatnonzero(raw_labels == label)
        labels[cluster_indices[distances > threshold]] = -1
    return labels


def _candidate_separation(matrix: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score  # type: ignore[import-not-found]

    mask = labels >= 0
    kept_labels = labels[mask]
    if mask.sum() < 3 or np.unique(kept_labels).size < 2:
        return 0.0
    try:
        sample_size = min(1000, int(mask.sum()))
        return max(
            0.0,
            float(
                silhouette_score(
                    matrix[mask],
                    kept_labels,
                    sample_size=sample_size,
                    random_state=42,
                )
            ),
        )
    except ValueError:
        return 0.0


def _adjusted_rand(left: np.ndarray, right: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score  # type: ignore[import-not-found]

    return float(adjusted_rand_score(left, right))


def _candidate_score(candidate: _Candidate, config: BreathClusteringConfig) -> float:
    excess_noise = max(0.0, candidate.noise_fraction - config.noise_flag_fraction)
    return float(
        0.75 * candidate.separation
        + 0.25 * (1.0 - candidate.noise_fraction)
        - 2.0 * excess_noise
    )


def _select_candidate(
    candidates: Sequence[_Candidate], config: BreathClusteringConfig
) -> _Candidate:
    """Select by fit quality, using the preferred count only as a tie-break."""
    return max(
        candidates,
        key=lambda candidate: (
            candidate.score,
            -abs(candidate.cluster_count - config.preferred_clusters),
            candidate.minimum_cluster_fraction,
        ),
    )


def _select_candidate_stability_aware(
    candidates: Sequence[_Candidate],
    dataset: BreathDataset,
    config: BreathClusteringConfig,
) -> tuple[_Candidate, dict[str, Any]]:
    """Select a candidate, folding out-of-block stability into the score.

    Many HDBSCAN parameterisations produce identical label vectors, so the
    candidates are deduplicated by labels before out-of-block stability is
    evaluated. The selected candidate and its stability report are returned
    together so the caller can reuse the report.
    """
    seen: dict[bytes, _Candidate] = {}
    for candidate in candidates:
        seen.setdefault(candidate.labels.tobytes(), candidate)

    scored: list[tuple[float, _Candidate, dict[str, Any]]] = []
    for candidate in seen.values():
        stability = assess_clustering_stability(
            dataset,
            candidate.labels,
            config=config,
            candidate=candidate,
            algorithm=candidate.algorithm,
        )
        status = stability.get("status")
        mean_ari = stability.get("mean_ari")
        validated = (
            float(mean_ari)
            if status in {"stable", "unstable"} and mean_ari is not None
            else 0.0
        )
        total = float(candidate.score) + config.selection_stability_weight * validated
        scored.append((total, candidate, stability))

    scored.sort(
        key=lambda item: (
            item[0],
            -abs(item[1].cluster_count - config.preferred_clusters),
            item[1].minimum_cluster_fraction,
        ),
        reverse=True,
    )
    _total, best, stability = scored[0]
    return best, stability


def _fit_candidates(
    matrix: np.ndarray,
    config: BreathClusteringConfig,
    *,
    algorithms: tuple[str, ...] = ("hdbscan", "kmeans"),
) -> list[_Candidate]:
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        return []
    if not np.isfinite(matrix).all():
        raise ValueError("Clustering matrix contains non-finite values")

    sizes = sorted(
        {
            min(int(size), matrix.shape[0] - 1)
            for size in config.min_cluster_sizes
            if 2 <= int(size) < matrix.shape[0]
        }
    )
    samples = sorted(
        {
            min(int(value), matrix.shape[0] - 1)
            for value in config.min_samples
            if 1 <= int(value) < matrix.shape[0]
        }
    )
    candidates: list[_Candidate] = []
    if "hdbscan" in algorithms:
        for min_cluster_size in sizes:
            for min_samples in samples:
                try:
                    labels, persistence, backend = _fit_hdbscan(
                        matrix, min_cluster_size, min_samples
                    )
                except (RuntimeError, ValueError):
                    continue
                real_labels = labels[labels >= 0]
                cluster_count = int(np.unique(real_labels).size)
                if cluster_count < 2 or cluster_count > config.max_clusters:
                    continue
                _, cluster_sizes = np.unique(real_labels, return_counts=True)
                minimum_fraction = (
                    float(np.min(cluster_sizes) / len(labels))
                    if real_labels.size
                    else 0.0
                )
                if minimum_fraction < config.min_cluster_fraction:
                    continue
                candidates.append(
                    _Candidate(
                        algorithm="hdbscan",
                        labels=labels,
                        cluster_count=cluster_count,
                        noise_fraction=float(np.mean(labels == -1)),
                        separation=_candidate_separation(matrix, labels),
                        persistence=max(0.0, min(1.0, persistence)),
                        minimum_cluster_fraction=minimum_fraction,
                        forced_partition=False,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        backend=backend,
                    )
                )

    if "kmeans" in algorithms and not any(
        candidate.algorithm == "hdbscan" for candidate in candidates
    ):
        distinct_rows = np.unique(matrix, axis=0).shape[0]
        max_cluster_count = min(config.max_clusters, matrix.shape[0] - 1, distinct_rows)
        for cluster_count in range(2, max_cluster_count + 1):
            try:
                labels = _fit_kmeans(matrix, cluster_count)
            except (ImportError, ValueError):
                continue
            real_labels = labels[labels >= 0]
            cluster_count_found = int(np.unique(real_labels).size)
            if cluster_count_found < 2 or cluster_count_found > config.max_clusters:
                continue
            _, cluster_sizes = np.unique(real_labels, return_counts=True)
            minimum_fraction = (
                float(np.min(cluster_sizes) / len(labels)) if real_labels.size else 0.0
            )
            if minimum_fraction < config.min_cluster_fraction:
                continue
            candidates.append(
                _Candidate(
                    algorithm="kmeans",
                    labels=labels,
                    cluster_count=cluster_count_found,
                    noise_fraction=float(np.mean(labels == -1)),
                    separation=_candidate_separation(matrix, labels),
                    persistence=0.0,
                    minimum_cluster_fraction=minimum_fraction,
                    forced_partition=True,
                    min_cluster_size=None,
                    min_samples=None,
                    backend="sklearn-kmeans",
                )
            )

    return _finalise_candidates(candidates, config)


def _finalise_candidates(
    candidates: Sequence[_Candidate], config: BreathClusteringConfig
) -> list[_Candidate]:
    """Add same-algorithm parameter agreement and the base fit score."""
    finalised: list[_Candidate] = []
    for index, candidate in enumerate(candidates):
        peers = [
            _adjusted_rand(candidate.labels, other.labels)
            for other_index, other in enumerate(candidates)
            if other_index != index and other.algorithm == candidate.algorithm
        ]
        stability = float(np.mean(peers)) if peers else 0.5
        finalised.append(
            _Candidate(
                algorithm=candidate.algorithm,
                labels=candidate.labels,
                cluster_count=candidate.cluster_count,
                noise_fraction=candidate.noise_fraction,
                separation=candidate.separation,
                persistence=candidate.persistence,
                minimum_cluster_fraction=candidate.minimum_cluster_fraction,
                forced_partition=candidate.forced_partition,
                min_cluster_size=candidate.min_cluster_size,
                min_samples=candidate.min_samples,
                backend=candidate.backend,
                stability=stability,
                score=_candidate_score(
                    _Candidate(
                        algorithm=candidate.algorithm,
                        labels=candidate.labels,
                        cluster_count=candidate.cluster_count,
                        noise_fraction=candidate.noise_fraction,
                        separation=candidate.separation,
                        persistence=candidate.persistence,
                        minimum_cluster_fraction=candidate.minimum_cluster_fraction,
                        forced_partition=candidate.forced_partition,
                        min_cluster_size=candidate.min_cluster_size,
                        min_samples=candidate.min_samples,
                        backend=candidate.backend,
                        stability=stability,
                    ),
                    config,
                ),
            )
        )
    return finalised


def _fit_candidates_scalable(
    matrix: np.ndarray,
    config: BreathClusteringConfig,
    *,
    algorithms: tuple[str, ...] = ("hdbscan", "kmeans"),
    fit_indices: np.ndarray | None = None,
) -> list[_Candidate]:
    """Fit candidates on a bounded subsample, then predict the full matrix.

    HDBSCAN is roughly quadratic in the number of rows, so fitting on tens of
    thousands of breaths is slow. For large recordings the candidate fit runs
    on a bounded row set (a seeded random subsample, or an explicit quality
    filtered subset) and the resulting prototypes are applied to every breath,
    keeping label counts and cluster fractions on the full dataset.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        return []
    n_rows = matrix.shape[0]
    if fit_indices is not None:
        fit_indices = np.unique(
            np.clip(np.asarray(fit_indices, dtype=int), 0, n_rows - 1)
        )
        if fit_indices.size < 3:
            return []
        if fit_indices.size > config.fit_max_samples:
            rng = np.random.default_rng(42)
            fit_indices = np.sort(
                rng.choice(fit_indices, size=config.fit_max_samples, replace=False)
            )
    elif n_rows <= config.fit_max_samples:
        return _fit_candidates(matrix, config, algorithms=algorithms)
    else:
        rng = np.random.default_rng(42)
        fit_indices = np.sort(
            rng.choice(n_rows, size=config.fit_max_samples, replace=False)
        )
    subsample = matrix[fit_indices]
    raw_candidates = _fit_candidates(subsample, config, algorithms=algorithms)

    expanded: list[_Candidate] = []
    for candidate in raw_candidates:
        model = _model_from_labels(
            subsample,
            candidate.labels,
            subsample.shape[1],
            algorithm=candidate.algorithm,
        )
        if model is None:
            continue
        full_labels = _predict_waveforms(matrix, model)
        real_labels = full_labels[full_labels >= 0]
        cluster_count = int(np.unique(real_labels).size)
        if cluster_count < 2 or cluster_count > config.max_clusters:
            continue
        _, cluster_sizes = np.unique(real_labels, return_counts=True)
        minimum_fraction = (
            float(np.min(cluster_sizes) / n_rows) if real_labels.size else 0.0
        )
        if minimum_fraction < config.min_cluster_fraction:
            continue
        expanded.append(
            _Candidate(
                algorithm=candidate.algorithm,
                labels=full_labels,
                cluster_count=cluster_count,
                noise_fraction=float(np.mean(full_labels == -1)),
                separation=_candidate_separation(matrix, full_labels),
                persistence=candidate.persistence,
                minimum_cluster_fraction=minimum_fraction,
                forced_partition=candidate.forced_partition,
                min_cluster_size=candidate.min_cluster_size,
                min_samples=candidate.min_samples,
                backend=f"{candidate.backend}-subsample",
            )
        )

    return _finalise_candidates(expanded, config)


def _model_from_labels(
    waveforms: np.ndarray,
    labels: np.ndarray,
    waveform_points: int,
    *,
    algorithm: str = "unknown",
) -> BreathMorphologyModel | None:
    real_labels = sorted(int(label) for label in np.unique(labels) if label >= 0)
    if not real_labels:
        return None

    prototypes = np.vstack(
        [waveforms[labels == label].mean(axis=0) for label in real_labels]
    )
    thresholds: list[float] = []
    for label, prototype in zip(real_labels, prototypes, strict=True):
        distances = np.linalg.norm(waveforms[labels == label] - prototype, axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        thresholds.append(
            max(float(np.quantile(distances, 0.99) * 1.25), median + 3.0 * mad, 1e-6)
        )
    threshold = max(thresholds)
    return BreathMorphologyModel(
        waveform_points=waveform_points,
        labels=tuple(real_labels),
        prototypes=prototypes,
        distance_threshold=threshold,
        algorithm=algorithm,
        distance_thresholds=tuple(thresholds),
    )


def _predict_waveforms(
    waveforms: np.ndarray, model: BreathMorphologyModel
) -> np.ndarray:
    if waveforms.ndim != 2 or waveforms.shape[1] != model.waveform_points:
        raise ValueError("waveform matrix is incompatible with the morphology model")
    predictions = np.full(waveforms.shape[0], -1, dtype=int)
    finite = np.isfinite(waveforms).all(axis=1)
    if not finite.any():
        return predictions
    distances = np.linalg.norm(
        waveforms[finite, :, None] - model.prototypes.T[None, :, :], axis=1
    )
    nearest = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(nearest)), nearest]
    thresholds = np.asarray(model.distance_thresholds, dtype=float)
    accepted = nearest_distances <= thresholds[nearest]
    predictions[finite] = np.where(
        accepted, np.asarray(model.labels, dtype=int)[nearest], -1
    )
    return predictions


def predict_breath_dataset(
    dataset: BreathDataset,
    model: BreathMorphologyModel,
    *,
    label_column: str = "cluster_label_shape_reference",
) -> pd.DataFrame:
    """Assign saved prototype labels to a new breath dataset.

    Rows that failed extraction retain ``-2``. Finite breaths farther from every
    prototype than the model threshold receive ``-1`` as explicit noise.
    """
    result = dataset.records.copy()
    result[label_column] = -2
    if not dataset.clusterable_indices:
        return result
    labels = _predict_waveforms(dataset.waveforms, model)
    for row_index, label in zip(dataset.clusterable_indices, labels, strict=True):
        result.at[row_index, label_column] = int(label)
    return result


def save_breath_morphology_model(
    model: BreathMorphologyModel, model_path: Path
) -> dict[str, Path]:
    """Persist prototypes without pickle or executable object deserialisation."""
    npz_path = (
        model_path if model_path.suffix == ".npz" else model_path.with_suffix(".npz")
    )
    json_path = npz_path.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        prototypes=np.asarray(model.prototypes, dtype=float),
        labels=np.asarray(model.labels, dtype=int),
    )
    metadata = {
        "schema_version": model.schema_version,
        "waveform_points": model.waveform_points,
        "distance_threshold": model.distance_threshold,
        "distance_thresholds": list(model.distance_thresholds),
        "n_clusters": len(model.labels),
        "algorithm": model.algorithm,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model": npz_path, "metadata": json_path}


def load_breath_morphology_model(model_path: Path) -> BreathMorphologyModel:
    """Load a validated prototype model from the safe NPZ/JSON pair."""
    npz_path = (
        model_path if model_path.suffix == ".npz" else model_path.with_suffix(".npz")
    )
    json_path = npz_path.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as raw:
        prototypes = np.asarray(raw["prototypes"], dtype=float)
        labels = tuple(int(label) for label in np.asarray(raw["labels"], dtype=int))
    if metadata.get("n_clusters") != len(labels):
        raise ValueError("morphology model metadata does not match stored labels")
    return BreathMorphologyModel(
        waveform_points=int(metadata["waveform_points"]),
        labels=labels,
        prototypes=prototypes,
        distance_threshold=float(metadata["distance_threshold"]),
        schema_version=int(metadata["schema_version"]),
        algorithm=str(metadata.get("algorithm", "unknown")),
        distance_thresholds=tuple(
            float(value) for value in metadata.get("distance_thresholds", [])
        ),
    )


def assess_clustering_stability(
    dataset: BreathDataset,
    reference_labels: np.ndarray,
    *,
    config: BreathClusteringConfig | None = None,
    candidate: _Candidate | None = None,
    algorithm: str | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Estimate temporal out-of-block stability of a recording-level fit.

    A selected candidate configuration is refit on contiguous temporal blocks,
    then evaluated only on breaths outside each training block. This prevents
    the optimistic leakage produced by fitting and scoring on the same breaths.
    It remains a diagnostic, not validation across independent recordings.
    """
    config = config or BreathClusteringConfig()
    n_samples = len(dataset.clusterable_indices)
    if n_samples < max(3, config.min_period_breaths):
        return {"status": "insufficient_data", "valid_repeats": 0}
    if reference_labels.shape != (n_samples,):
        raise ValueError("reference_labels must align with dataset.waveforms")

    if candidate is None:
        candidates = _fit_candidates(
            dataset.waveforms,
            config,
            algorithms=(algorithm,) if algorithm is not None else ("hdbscan", "kmeans"),
        )
        matching = [
            item
            for item in candidates
            if algorithm is None or item.algorithm == algorithm
        ]
        if not matching:
            return {
                "status": "not_available",
                "valid_repeats": 0,
                "requested_repeats": config.stability_repeats,
            }
        candidate = _select_candidate(
            matching,
            config,
        )

    sample_size = min(
        n_samples,
        max(3, int(round(n_samples * config.stability_sample_fraction))),
        config.stability_max_samples,
    )
    if sample_size >= n_samples:
        return {
            "status": "not_available",
            "valid_repeats": 0,
            "failed_repeats": config.stability_repeats,
            "requested_repeats": config.stability_repeats,
            "sample_size": sample_size,
            "sample_fraction": sample_size / n_samples,
        }

    requested_repeats = config.stability_repeats
    if requested_repeats <= 0:
        return {
            "status": "not_available",
            "valid_repeats": 0,
            "failed_repeats": 0,
            "requested_repeats": requested_repeats,
            "sample_size": sample_size,
            "sample_fraction": sample_size / n_samples,
        }

    max_start = n_samples - sample_size
    starts = np.rint(
        np.linspace(0, max_start, requested_repeats, endpoint=True)
    ).astype(int)
    if max_start:
        starts = (starts + int(random_seed)) % (max_start + 1)
    scores: list[float] = []
    failed_repeats = 0
    for start in starts:
        indices: np.ndarray = np.arange(start, start + sample_size, dtype=int)
        try:
            if candidate.algorithm == "kmeans":
                labels = _fit_kmeans(
                    dataset.waveforms[indices], candidate.cluster_count
                )
            elif candidate.algorithm == "hdbscan":
                if candidate.min_cluster_size is None or candidate.min_samples is None:
                    raise ValueError("HDBSCAN candidate is missing its fit parameters")
                labels, _persistence, _backend = _fit_hdbscan(
                    dataset.waveforms[indices],
                    candidate.min_cluster_size,
                    candidate.min_samples,
                )
            else:
                raise ValueError(
                    f"Unsupported clustering algorithm: {candidate.algorithm}"
                )
        except (ImportError, RuntimeError, ValueError):
            failed_repeats += 1
            continue
        model = _model_from_labels(
            dataset.waveforms[indices],
            labels,
            config.waveform_points,
            algorithm=candidate.algorithm,
        )
        if model is None:
            failed_repeats += 1
            continue
        predicted = _predict_waveforms(dataset.waveforms, model)
        out_of_block: np.ndarray = np.ones(n_samples, dtype=bool)
        out_of_block[indices] = False
        comparable = out_of_block & (reference_labels >= -1) & (predicted >= -1)
        if comparable.sum() < 3 or np.unique(reference_labels[comparable]).size < 2:
            failed_repeats += 1
            continue
        scores.append(
            _adjusted_rand(reference_labels[comparable], predicted[comparable])
        )

    if not scores:
        return {
            "status": "not_available",
            "valid_repeats": 0,
            "failed_repeats": failed_repeats,
            "requested_repeats": requested_repeats,
            "sample_size": sample_size,
            "sample_fraction": sample_size / n_samples,
        }
    mean_ari = float(np.mean(scores))
    return {
        "status": "stable" if mean_ari >= 0.80 and failed_repeats == 0 else "unstable",
        "valid_repeats": len(scores),
        "failed_repeats": failed_repeats,
        "requested_repeats": requested_repeats,
        "sample_size": sample_size,
        "sample_fraction": sample_size / n_samples,
        "mean_ari": mean_ari,
        "min_ari": float(np.min(scores)),
        "max_ari": float(np.max(scores)),
    }


def fit_reference_breath_model(
    datasets: Sequence[BreathDataset],
    *,
    config: BreathClusteringConfig | None = None,
) -> BreathMorphologyModel | None:
    """Fit one reusable morphology model from one or more recordings.

    The prototypes use the same extraction and time normalisation as the
    recording-level fit. Applying this model makes numeric labels comparable
    across recordings without allowing the model to alter peak or period
    validity. Period summaries remain local to each recording.
    """
    config = config or BreathClusteringConfig()
    matrices = [
        np.asarray(dataset.waveforms, dtype=float)
        for dataset in datasets
        if dataset.waveforms.ndim == 2 and len(dataset.waveforms)
    ]
    if not matrices:
        return None
    waveform_points = matrices[0].shape[1]
    if any(matrix.shape[1] != waveform_points for matrix in matrices):
        raise ValueError("reference datasets use different waveform sizes")
    matrix = np.vstack(matrices)
    if len(matrix) < max(3, config.min_period_breaths):
        return None
    if not np.isfinite(matrix).all():
        raise ValueError("reference clustering matrix contains non-finite values")
    candidates = _fit_candidates_scalable(matrix, config)
    if not candidates:
        return None
    best = _select_candidate(candidates, config)
    return _model_from_labels(
        matrix, best.labels, waveform_points, algorithm=best.algorithm
    )


def _status_for_noise(noise_fraction: float, config: BreathClusteringConfig) -> str:
    if noise_fraction > config.noise_unusable_fraction:
        return "unusable_noise"
    if noise_fraction > config.noise_flag_fraction:
        return "flagged_noise"
    return "ok"


def _period_quality_mask(
    dataset: BreathDataset, config: BreathClusteringConfig
) -> tuple[np.ndarray, list[str]]:
    """Return a mask over clusterable breaths, excluding high-variability periods.

    Periods whose mean within-period waveform SD exceeds the recording median by
    ``noisy_period_sd_ratio`` are excluded from the morphology fit. Their
    breaths stay in the output and are labelled by the nearest fitted
    prototype. The gate is clustering-scoped and does not alter peak or period
    validity.
    """
    if not dataset.clusterable_indices:
        return np.ones(0, dtype=bool), []
    rows = dataset.records.iloc[list(dataset.clusterable_indices)]
    periods = rows["period_name"].astype(str).to_numpy()
    unique_periods = np.unique(periods)
    if unique_periods.size <= 1:
        return np.ones(len(periods), dtype=bool), []
    per_period_sd = {
        period: float(dataset.waveforms[periods == period].std(axis=0).mean())
        for period in unique_periods
    }
    median_sd = float(np.median(list(per_period_sd.values())))
    if not np.isfinite(median_sd) or median_sd <= 0.0:
        return np.ones(len(periods), dtype=bool), []
    threshold = median_sd * config.noisy_period_sd_ratio
    excluded = sorted(period for period, sd in per_period_sd.items() if sd > threshold)
    mask = np.array(
        [per_period_sd[period] <= threshold for period in periods], dtype=bool
    )
    return mask, excluded


def _period_summary(
    breaths: pd.DataFrame, config: BreathClusteringConfig
) -> pd.DataFrame:
    columns = [
        "window_key",
        "period_name",
        "n_breaths",
        "n_clusterable",
        "n_noise",
        "noise_fraction",
        "n_clusters",
        "status",
    ]
    if breaths.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (window_key, period_name), group in breaths.groupby(
        ["window_key", "period_name"], dropna=False
    ):
        clusterable = group[group["cluster_label_shape"] != -2]
        labels = clusterable["cluster_label_shape"]
        noise = int((labels == -1).sum())
        denominator = len(labels)
        noise_fraction = noise / denominator if denominator else np.nan
        if denominator == 0:
            status = "no_clusterable_breaths"
        elif denominator < config.min_period_breaths:
            status = "insufficient_breaths"
        elif noise_fraction > config.noise_unusable_fraction:
            status = "unusable_noise"
        elif noise_fraction > config.noise_flag_fraction:
            status = "flagged_noise"
        else:
            status = "ok"
        rows.append(
            {
                "window_key": window_key,
                "period_name": period_name,
                "n_breaths": len(group),
                "n_clusterable": denominator,
                "n_noise": noise,
                "noise_fraction": noise_fraction,
                "n_clusters": int((labels[labels >= 0].value_counts() >= 2).sum()),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _period_distribution(
    breaths: pd.DataFrame, config: BreathClusteringConfig
) -> pd.DataFrame:
    """Return cluster counts and fractions for each label view and period."""
    columns = [
        "window_key",
        "period_name",
        "method",
        "cluster_label",
        "count",
        "fraction",
        "proportion_status",
    ]
    if breaths.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (window_key, period_name), group in breaths.groupby(
        ["window_key", "period_name"], dropna=False
    ):
        for method, column in (
            ("shape", "cluster_label_shape"),
            ("shape_amplitude", "cluster_label_shape_amplitude"),
        ):
            if column not in group or (
                method == "shape_amplitude" and not config.amplitude_sensitivity
            ):
                continue
            labels = group.loc[group[column] != -2, column]
            if labels.empty:
                rows.append(
                    {
                        "window_key": window_key,
                        "period_name": period_name,
                        "method": method,
                        "cluster_label": -2,
                        "count": 0,
                        "fraction": np.nan,
                        "proportion_status": "no_clusterable_breaths",
                    }
                )
                continue
            counts = labels.value_counts().sort_index()
            noise_fraction = float((labels == -1).mean())
            if len(labels) < config.min_period_breaths:
                proportion_status = "insufficient_breaths"
                fraction = np.nan
            else:
                proportion_status = _status_for_noise(noise_fraction, config)
                fraction = None
            for label, count in counts.items():
                rows.append(
                    {
                        "window_key": window_key,
                        "period_name": period_name,
                        "method": method,
                        "cluster_label": int(label),
                        "count": int(count),
                        "fraction": (
                            float(count / len(labels)) if fraction is None else fraction
                        ),
                        "proportion_status": proportion_status,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def cluster_breath_dataset(
    dataset: BreathDataset,
    *,
    config: BreathClusteringConfig | None = None,
) -> BreathClusteringResult:
    """Cluster one recording and return labels plus behaviour-period summaries."""
    config = config or BreathClusteringConfig()
    if config.max_clusters < 2:
        raise ValueError("max_clusters must be at least 2")
    if config.preferred_clusters < 2:
        raise ValueError("preferred_clusters must be at least 2")

    breaths = dataset.records.copy()
    for column in ("cluster_label_shape", "cluster_label_shape_amplitude"):
        breaths[column] = -2
    breaths["period_excluded_from_fit"] = False

    n_clusterable = len(dataset.clusterable_indices)
    metadata: dict[str, Any] = {
        "status": "insufficient_data",
        "n_breaths": len(breaths),
        "n_clusterable": n_clusterable,
        "config": asdict(config),
        "preferred_clusters": config.preferred_clusters,
        "max_clusters": config.max_clusters,
    }
    if n_clusterable < max(3, config.min_period_breaths):
        return BreathClusteringResult(
            breaths=breaths,
            period_summary=_period_summary(breaths, config),
            period_distribution=_period_distribution(breaths, config),
            metadata=metadata,
            dataset=dataset,
        )

    fit_indices: np.ndarray | None = None
    if config.exclude_noisy_periods:
        period_mask, excluded_periods = _period_quality_mask(dataset, config)
        if not period_mask.all():
            fit_indices = np.flatnonzero(period_mask)
            metadata["excluded_periods_for_fit"] = excluded_periods
            metadata["n_fit_breaths"] = int(period_mask.sum())
            metadata["fit_quality_gate"] = True
            # This describes the period-level fit decision, so include retained
            # rows that were individually dropped from the morphology fit too.
            excluded_period_set = set(excluded_periods)
            breaths["period_excluded_from_fit"] = (
                breaths["period_name"].astype(str).isin(excluded_period_set)
            )

    try:
        candidates = _fit_candidates_scalable(
            dataset.waveforms, config, fit_indices=fit_indices
        )
    except RuntimeError as exc:
        metadata["status"] = "dependency_unavailable"
        metadata["error"] = str(exc)
        return BreathClusteringResult(
            breaths=breaths,
            period_summary=_period_summary(breaths, config),
            period_distribution=_period_distribution(breaths, config),
            metadata=metadata,
            dataset=dataset,
        )

    if not candidates:
        metadata["status"] = "no_valid_fit"
        return BreathClusteringResult(
            breaths=breaths,
            period_summary=_period_summary(breaths, config),
            period_distribution=_period_distribution(breaths, config),
            metadata=metadata,
            dataset=dataset,
        )

    best, stability = _select_candidate_stability_aware(candidates, dataset, config)
    for row_index, label in zip(dataset.clusterable_indices, best.labels, strict=True):
        breaths.at[row_index, "cluster_label_shape"] = int(label)

    noise_fraction = float(np.mean(best.labels == -1))
    model = _model_from_labels(
        dataset.waveforms,
        best.labels,
        config.waveform_points,
        algorithm=best.algorithm,
    )
    noise_status = _status_for_noise(noise_fraction, config)
    fit_status = (
        "unusable_noise"
        if noise_status == "unusable_noise"
        else "forced_partition"
        if best.forced_partition
        else noise_status
    )
    metadata.update(
        {
            "status": fit_status,
            "algorithm": best.algorithm,
            "backend": best.backend,
            "forced_partition": best.forced_partition,
            "cluster_count": best.cluster_count,
            "noise_fraction": noise_fraction,
            "noise_status": noise_status,
            "manual_review_required": (
                best.forced_partition
                or noise_status == "unusable_noise"
                or stability["status"] != "stable"
            ),
            "separation": best.separation,
            "parameter_agreement": best.stability,
            "score": best.score,
            "candidate_count": len(candidates),
            "candidate_diagnostics": [
                {
                    "algorithm": candidate.algorithm,
                    "cluster_count": candidate.cluster_count,
                    "noise_fraction": candidate.noise_fraction,
                    "minimum_cluster_fraction": candidate.minimum_cluster_fraction,
                    "separation": candidate.separation,
                    "parameter_agreement": candidate.stability,
                    "score": candidate.score,
                    "forced_partition": candidate.forced_partition,
                    "backend": candidate.backend,
                    "min_cluster_size": candidate.min_cluster_size,
                    "min_samples": candidate.min_samples,
                }
                for candidate in candidates
            ],
            "stability_validation": stability,
        }
    )

    if config.amplitude_sensitivity:
        amplitude = np.log1p(
            np.asarray(
                breaths.iloc[list(dataset.clusterable_indices)]["amplitude"],
                dtype=float,
            )
        )
        amplitude_std = float(np.std(amplitude))
        if amplitude_std > 1e-9 and np.isfinite(amplitude).all():
            amplitude_z = (amplitude - float(np.mean(amplitude))) / amplitude_std
            sensitivity_matrix = np.column_stack(
                [dataset.waveforms, config.amplitude_weight * amplitude_z]
            )
            sensitivity_candidates = _fit_candidates_scalable(
                sensitivity_matrix, config, fit_indices=fit_indices
            )
            if sensitivity_candidates:
                sensitivity_best = _select_candidate(sensitivity_candidates, config)
                for row_index, label in zip(
                    dataset.clusterable_indices,
                    sensitivity_best.labels,
                    strict=True,
                ):
                    breaths.at[row_index, "cluster_label_shape_amplitude"] = int(label)
                metadata["amplitude_sensitivity"] = {
                    "algorithm": sensitivity_best.algorithm,
                    "backend": sensitivity_best.backend,
                    "forced_partition": sensitivity_best.forced_partition,
                    "cluster_count": sensitivity_best.cluster_count,
                    "noise_fraction": float(np.mean(sensitivity_best.labels == -1)),
                    "noise_status": _status_for_noise(
                        float(np.mean(sensitivity_best.labels == -1)), config
                    ),
                    "separation": sensitivity_best.separation,
                    "parameter_agreement": sensitivity_best.stability,
                    "score": sensitivity_best.score,
                }

    return BreathClusteringResult(
        breaths=breaths,
        period_summary=_period_summary(breaths, config),
        period_distribution=_period_distribution(breaths, config),
        metadata=metadata,
        model=model,
        dataset=dataset,
    )


def cluster_breaths(
    window_periods: dict[str, list[dict[str, Any]]],
    pressure_data: pd.DataFrame,
    *,
    config: BreathClusteringConfig | None = None,
) -> BreathClusteringResult:
    """Extract and cluster breath waveforms for one recording."""
    config = config or BreathClusteringConfig()
    dataset = extract_breath_dataset(
        window_periods,
        pressure_data,
        waveform_points=config.waveform_points,
        max_gap_seconds=config.max_gap_seconds,
        duration_ratio_bounds=config.duration_ratio_bounds,
    )
    return cluster_breath_dataset(dataset, config=config)


def export_breath_clustering(
    result: BreathClusteringResult, output_folder: Path
) -> dict[str, Path]:
    """Write machine-readable breath labels, period summaries, and metadata."""
    output_folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "breaths": output_folder / "breath_cluster_labels.csv",
        "periods": output_folder / "period_cluster_summary.csv",
        "distribution": output_folder / "period_cluster_distribution.csv",
        "metadata": output_folder / "clustering_summary.json",
    }
    result.breaths.to_csv(paths["breaths"], index=False)
    result.period_summary.to_csv(paths["periods"], index=False)
    result.period_distribution.to_csv(paths["distribution"], index=False)
    if result.model is not None:
        model_paths = save_breath_morphology_model(
            result.model, output_folder / "breath_morphology_model.npz"
        )
        paths.update(
            {"model_npz": model_paths["model"], "model_json": model_paths["metadata"]}
        )
    metadata = dict(result.metadata)
    if result.model is not None:
        metadata["model_files"] = {
            "npz": paths["model_npz"].name,
            "json": paths["model_json"].name,
        }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return paths


def plot_breath_cluster_waveforms(
    result: BreathClusteringResult, output_folder: Path
) -> Path | None:
    """Write a deterministic mean-and-spread plot for shape clusters."""
    if result.dataset is None or not result.dataset.clusterable_indices:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_folder.mkdir(parents=True, exist_ok=True)
    clusterable = result.breaths.iloc[list(result.dataset.clusterable_indices)]
    labels = np.asarray(clusterable["cluster_label_shape"], dtype=int)
    included = labels >= -1
    if not included.any():
        return None
    labels = labels[included]
    waveforms = result.dataset.waveforms[included]
    x = np.linspace(0.0, 1.0, waveforms.shape[1])

    figure, axis = plt.subplots(figsize=(10, 6))
    unique_labels = sorted(int(label) for label in np.unique(labels))
    colour_map = plt.get_cmap("tab10")
    for label in unique_labels:
        group = waveforms[labels == label]
        if not len(group):
            continue
        mean = group.mean(axis=0)
        spread = group.std(axis=0)
        name = "Noise" if label == -1 else f"Cluster {label}"
        colour = "#808080" if label == -1 else colour_map(label % 10)
        axis.plot(x, mean, label=f"{name} (n={len(group)})", color=colour)
        axis.fill_between(x, mean - spread, mean + spread, alpha=0.15, color=colour)

    axis.set(
        title="Breath morphology clusters (mean ± SD)",
        xlabel="Normalised trough-to-trough time",
        ylabel="Normalised pressure",
        ylim=(-0.05, 1.05),
    )
    axis.legend()
    figure.tight_layout()
    path = output_folder / "breath_cluster_waveforms.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
