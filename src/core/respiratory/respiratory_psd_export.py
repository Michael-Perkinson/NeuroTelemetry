from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.respiratory.respiratory_psd_analysis import (
    DEFAULT_PSD_AUC_BAND_HZ,
    DEFAULT_PSD_SEGMENT_SECONDS,
    _build_ttot_trace,
)


def _export_psd_workbook(
    output_folder: Path,
    config: dict[str, Any],
    metrics_df: pd.DataFrame,
    rejection_df: pd.DataFrame,
    segment_summary_df: pd.DataFrame,
    per_window_psd_df: pd.DataFrame,
    comparison_per_window_psd_df: pd.DataFrame,
    log=print,
) -> None:
    """Write the PSD workbook students can send back for easy collation."""
    workbook_path = output_folder / "ttot_psd_summary.xlsx"

    accepted_segments = int(len(segment_summary_df))
    rejected_rows = int(len(rejection_df))
    windows_with_psd = (
        int(metrics_df["Window"].nunique()) if not metrics_df.empty else 0
    )
    windows_with_warnings = (
        int((metrics_df["QC_Warning"].fillna("") != "").sum())
        if not metrics_df.empty
        else 0
    )

    summary_rows = [
        {"Field": "Metric", "Value": config.get("Metric", "Ttot")},
        {"Field": "AcceptedSegments", "Value": accepted_segments},
        {"Field": "RejectedPeriodsOrSegments", "Value": rejected_rows},
        {"Field": "WindowsWithPSD", "Value": windows_with_psd},
        {"Field": "WindowsWithQCWarnings", "Value": windows_with_warnings},
        {"Field": "SummaryPlotPNG", "Value": "ttot_psd_summary.png"},
        {"Field": "SummaryPlotSVG", "Value": "ttot_psd_summary.svg"},
    ]
    summary_rows.extend(
        {"Field": key, "Value": value}
        for key, value in config.items()
        if key != "Metric"
    )
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Run Summary", index=False)
        metrics_df.to_excel(writer, sheet_name="Window Metrics", index=False)
        rejection_df.to_excel(writer, sheet_name="Rejected", index=False)
        segment_summary_df.to_excel(writer, sheet_name="Segment Summary", index=False)
        per_window_psd_df.to_excel(writer, sheet_name="Mean PSD Curves", index=False)
        comparison_per_window_psd_df.to_excel(
            writer, sheet_name="Comparison PSD", index=False
        )

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                adjusted_width = min(max(max_length + 2, 12), 48)
                sheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = adjusted_width

    log(f"Ttot PSD workbook saved: {workbook_path}")


def export_ttot_traces(
    window_periods: dict[str, list[dict[str, Any]]],
    output_folder: Path,
    log=print,
) -> None:
    """Export raw breath-by-breath Ttot traces for each behavioural window."""
    output_folder.mkdir(parents=True, exist_ok=True)

    for window_key, periods in window_periods.items():
        period_frames: list[pd.DataFrame] = []
        for period_idx, period in enumerate(periods, start=1):
            trace_df = _build_ttot_trace(period)
            if trace_df.empty:
                continue
            period_frames.append(trace_df.assign(PeriodIndex=period_idx))
        if not period_frames:
            continue
        df = pd.concat(period_frames, ignore_index=True)
        safe_key = window_key.replace(":", "_").replace("-", "_").replace(" ", "_")
        out_path = output_folder / f"Ttot_{safe_key}.csv"
        df.to_csv(out_path, index=False)

    n = len(window_periods)
    log(f"Ttot traces exported ({n} windows): {output_folder}")


def export_ttot_psd_results(
    psd_results: dict[str, Any],
    output_folder: Path,
    *,
    suspicious_fmax: dict[str, float],
    total_candidate_segments: int,
    log=print,
) -> None:
    """Write computed Ttot PSD tables, QC workbook, and status messages."""
    output_folder.mkdir(parents=True, exist_ok=True)

    if suspicious_fmax:
        lines = ", ".join(
            f"{key}: {value:.4f} Hz" for key, value in suspicious_fmax.items()
        )
        log(
            f"WARNING — fmax at DC or near Nyquist for: {lines}. "
            "Check PSD curve manually."
        )

    rejection_df: pd.DataFrame = psd_results["rejection_log"]
    segment_summary_df: pd.DataFrame = psd_results["segment_summary"]
    per_segment_psd_df: pd.DataFrame = psd_results["per_segment_psd"]
    comparison_per_segment_psd_df: pd.DataFrame = psd_results[
        "comparison_per_segment_psd"
    ]
    per_window_psd_df: pd.DataFrame = psd_results["per_window_psd"]
    comparison_per_window_psd_df: pd.DataFrame = psd_results[
        "comparison_per_window_psd"
    ]
    pooled_psd_df: pd.DataFrame = psd_results["pooled_psd"]
    resampled_traces_df: pd.DataFrame = psd_results["resampled_traces"]
    window_metrics_df: pd.DataFrame = psd_results["window_metrics"]
    config: dict[str, Any] = psd_results["config"]

    rejection_df.to_csv(output_folder / "ttot_psd_rejected.csv", index=False)
    log(
        f"Ttot PSD: {len(rejection_df)} periods/segments rejected — "
        "see ttot_psd_rejected.csv"
    )
    segment_summary_df.to_csv(
        output_folder / "ttot_psd_segment_summary.csv", index=False
    )
    per_segment_psd_df.to_csv(output_folder / "ttot_psd_per_segment.csv", index=False)
    comparison_per_segment_psd_df.to_csv(
        output_folder / "ttot_psd_comparison_per_segment.csv", index=False
    )
    per_window_psd_df.to_csv(output_folder / "ttot_psd_per_window.csv", index=False)
    comparison_per_window_psd_df.to_csv(
        output_folder / "ttot_psd_comparison_per_window.csv", index=False
    )
    pooled_psd_df.to_csv(output_folder / "ttot_psd_pooled.csv", index=False)
    resampled_traces_df.to_csv(
        output_folder / "ttot_resampled_segments.csv", index=False
    )

    kept_segments = len(segment_summary_df)
    segment_seconds = float(config["SegmentDuration_s"])
    log(
        f"Ttot PSD: {kept_segments} / {total_candidate_segments} "
        f"candidate {segment_seconds:.0f}s segments used."
    )
    window_metrics_df.to_csv(output_folder / "ttot_psd_window_metrics.csv", index=False)
    _export_psd_workbook(
        output_folder=output_folder,
        config=config,
        metrics_df=window_metrics_df,
        rejection_df=rejection_df,
        segment_summary_df=segment_summary_df,
        per_window_psd_df=per_window_psd_df,
        comparison_per_window_psd_df=comparison_per_window_psd_df,
        log=log,
    )


def plot_psd_results(
    psd_results: dict[str, Any],
    output_folder: Path | None = None,
    *,
    show: bool = False,
    log=print,
) -> None:
    """Save a 4-panel PSD QC figure; optionally show it interactively."""
    per_window_psd_df: pd.DataFrame = psd_results.get("per_window_psd", pd.DataFrame())
    per_window_metrics: dict[str, dict[str, float]] = psd_results.get(
        "per_window_metrics", {}
    )
    auc_band = psd_results.get("config", {}).get("AUCBandHz", DEFAULT_PSD_AUC_BAND_HZ)
    segment_seconds = psd_results.get("config", {}).get(
        "SegmentDuration_s", DEFAULT_PSD_SEGMENT_SECONDS
    )

    if per_window_psd_df.empty or not per_window_metrics:
        log("Ttot PSD plot contains no valid per-window PSD results.")
        if output_folder is not None:
            output_folder.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No valid PSD results for this run",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            fig.savefig(
                output_folder / "ttot_psd_summary.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                output_folder / "ttot_psd_summary.svg",
                bbox_inches="tight",
            )
            plt.close(fig)
        return

    window_keys = list(per_window_metrics.keys())
    fmax_vals = [per_window_metrics[k]["fmax"] for k in window_keys]
    psdmax_vals = [per_window_metrics[k]["PSDmax"] for k in window_keys]
    auc_vals = [per_window_metrics[k]["AUC"] for k in window_keys]
    x = np.arange(len(window_keys))

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2)
    ax_psd = fig.add_subplot(grid[0, 0])
    ax_fmax = fig.add_subplot(grid[0, 1])
    ax_psdmax = fig.add_subplot(grid[1, 0])
    ax_auc = fig.add_subplot(grid[1, 1])

    colors = plt.cm.tab10(np.linspace(0, 1, len(window_keys)))

    for color, window_key in zip(colors, window_keys, strict=False):
        subset = per_window_psd_df[per_window_psd_df["Window"] == window_key]
        if subset.empty:
            continue
        n_seg = int(per_window_metrics[window_key].get("n_segments", 0))
        fmax = float(per_window_metrics[window_key].get("fmax", np.nan))
        psdmax = float(per_window_metrics[window_key].get("PSDmax", np.nan))
        ax_psd.plot(
            subset["Frequency_Hz"],
            subset["PSD"],
            color=color,
            alpha=0.8,
            linewidth=1.0,
            label=f"{window_key} (n={n_seg})",
        )
        if np.isfinite(fmax) and np.isfinite(psdmax):
            ax_psd.scatter([fmax], [psdmax], color=color, s=24, zorder=3)

    ax_psd.set_title(
        f"Mean PSD per window - Ttot variability ({segment_seconds:.0f}s segments)"
    )
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (s^2/Hz)")
    ax_psd.legend(fontsize=6)

    ax_fmax.bar(x, fmax_vals)
    ax_fmax.set_title("fmax per window")
    ax_fmax.set_ylabel("Frequency (Hz)")
    ax_fmax.set_xticks(x, window_keys, rotation=45, ha="right")

    ax_psdmax.bar(x, psdmax_vals)
    ax_psdmax.set_title("PSDmax per window")
    ax_psdmax.set_ylabel("Power (s^2/Hz)")
    ax_psdmax.set_xticks(x, window_keys, rotation=45, ha="right")

    ax_auc.bar(x, auc_vals)
    ax_auc.set_title(f"AUC +/-{auc_band} Hz around fmax")
    ax_auc.set_ylabel("Area (s^2)")
    ax_auc.set_xticks(x, window_keys, rotation=45, ha="right")

    plt.tight_layout()
    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)
        png_path = output_folder / "ttot_psd_summary.png"
        svg_path = output_folder / "ttot_psd_summary.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        log(f"Ttot PSD summary plot saved: {png_path}")
        log(f"Ttot PSD summary plot saved: {svg_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
