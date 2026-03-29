import os
from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# --- Main Full Trace Export ---
def export_full_time_range_plot(
    main_data: pd.DataFrame,
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame,
    valid_peak_times_all: list[float],
    valid_pre_peak_times_all: list[float],
    min_time: float,
    max_time: float,
    behaviour_to_plot: str,
    full_trace_folder: str,
    file_base: str,
    main_signal_col: str,  # e.g. "SmoothedPressure" or "dfof_465"
    main_signal_label: str,  # e.g. "Pressure" or "Photometry"
    atm_pressure_data: pd.DataFrame | None = None,
    behavior_windows: list[tuple[float, float]] | None = None,
):
    # --- Filter dataframes to range ---
    main_filtered = filter_df_by_time(main_data, min_time, max_time)
    temp_filtered = filter_df_by_time(temp_data, min_time, max_time)
    activity_filtered = filter_df_by_time(activity_data, min_time, max_time)

    # --- Handle peaks ---
    valid_peak_times, valid_pre_peak_times = prepare_peaks_for_range(
        valid_peak_times_all, valid_pre_peak_times_all, min_time, max_time
    )

    title = f"{behaviour_to_plot}: Full Time Range {min_time:.2f} to {max_time:.2f}"

    # --- Interactive plot ---
    fig_html = create_interactive_plot(
        main_filtered,
        temp_filtered,
        activity_filtered,
        valid_peak_times,
        valid_pre_peak_times,
        title,
        main_signal_col,
        main_signal_label,
        atm_pressure_data=atm_pressure_data,
        behavior_windows=behavior_windows,
    )
    save_path_html = os.path.join(
        full_trace_folder,
        f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.html",
    )
    fig_html.write_html(save_path_html)
    print(f"Saved full trace HTML: {save_path_html}")

    # --- Static plot ---
    save_path_svg = os.path.join(
        full_trace_folder,
        f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.svg",
    )
    create_static_plot(
        main_filtered,
        temp_filtered,
        activity_filtered,
        valid_peak_times,
        valid_pre_peak_times,
        save_path_svg,
        title,
        main_signal_col,
        main_signal_label,
        atm_pressure_data=atm_pressure_data,
        behavior_windows=behavior_windows,
    )
    print(f"Saved full trace SVG: {save_path_svg}")


# --- Extracted helper for peaks ---
def prepare_peaks_for_range(
    valid_peak_times_all: list[float],
    valid_pre_peak_times_all: list[float],
    min_time: float,
    max_time: float,
) -> tuple[list[float], list[float]]:
    valid_peak_times = filter_times_to_range(valid_peak_times_all, min_time, max_time)
    valid_pre_peak_times = filter_times_to_range(
        valid_pre_peak_times_all, min_time, max_time
    )

    if not valid_peak_times or not valid_pre_peak_times:
        # If one is empty, don’t trim — return as is
        return valid_peak_times, valid_pre_peak_times

    # Otherwise, trim to same length
    min_len = min(len(valid_peak_times), len(valid_pre_peak_times))
    return valid_peak_times[:min_len], valid_pre_peak_times[:min_len]


# --- Main Behavior-by-Window Export ---
def export_behavior_images_interactive(
    time_windows: list[tuple[float, float]],
    pressure_data: pd.DataFrame,
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame,
    valid_peak_times_all: list[float],
    valid_pre_peak_times_all: list[float],
    behaviour_to_plot: str,
    html_save_folder: str,
    svg_save_folder: str,
    file_base: str,
    main_signal_col: str,
    main_signal_label: str,
):
    for i, (start_time, end_time) in enumerate(time_windows):
        # Downsample pressure data for faster plotting
        pressure_segment = filter_df_by_time(pressure_data, start_time, end_time)
        temp_segment = filter_df_by_time(temp_data, start_time, end_time)
        activity_segment = filter_df_by_time(activity_data, start_time, end_time)

        peak_times = filter_times_to_range(valid_peak_times_all, start_time, end_time)
        pre_peak_times = filter_times_to_range(
            valid_pre_peak_times_all, start_time, end_time
        )

        if pressure_segment.empty or temp_segment.empty or activity_segment.empty:
            print(f"Skipping plot for window {start_time}-{end_time}: no data.")
            continue

        title = f"{behaviour_to_plot} from {start_time:.2f} to {end_time:.2f}"

        html_path = os.path.join(
            html_save_folder,
            f"{behaviour_to_plot}_behavior_{i}_from_{start_time:.2f}_to_{end_time:.2f}.html",
        )
        fig_html = create_interactive_plot(
            pressure_segment,
            temp_segment,
            activity_segment,
            peak_times,
            pre_peak_times,
            title,
            main_signal_col,
            main_signal_label,
        )
        fig_html.write_html(html_path)
        print(f"Saved behavior HTML: {html_path}")

        svg_path = os.path.join(
            svg_save_folder,
            f"{behaviour_to_plot}_behavior_{i}_from_{start_time:.2f}_to_{end_time:.2f}.svg",
        )
        create_static_plot(
            pressure_segment,
            temp_segment,
            activity_segment,
            peak_times,
            pre_peak_times,
            svg_path,
            title,
            main_signal_col,
            main_signal_label,
        )
        print(f"Saved behavior SVG: {svg_path}")

    print("Finished exporting all interactive and static plots to folders.")


# --- Plot Helpers ---
def create_interactive_plot(
    main_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    peak_times: list[float],
    pre_peak_times: list[float],
    title: str,
    main_signal_col: str,
    main_signal_label: str,
    atm_pressure_data: pd.DataFrame | None = None,
    behavior_windows: list[tuple[float, float]] | None = None,
) -> go.Figure:
    fig = go.Figure()

    ymin, ymax = main_df[main_signal_col].min(), main_df[main_signal_col].max()
    yrange = ymax - ymin

    # --- Main signal ---
    main_color = "black" if main_signal_label == "Pressure" else "green"
    fig.add_trace(
        go.Scatter(
            x=main_df["TimeSinceReference"] / 60,
            y=main_df[main_signal_col],
            mode="lines",
            name=main_signal_label,
            line=dict(color=main_color, width=1),
            yaxis="y",
        )
    )

    # --- Temperature ---
    if not temp_df.empty:
        fig.add_trace(
            go.Scatter(
                x=temp_df["TimeSinceReference"] / 60,
                y=temp_df["Temp"],
                mode="lines",
                name="Temperature",
                line=dict(color="red", width=1),
                yaxis="y2",
            )
        )

    # --- Activity (1-min bins) ---
    if not activity_df.empty:
        act_binned = activity_df.copy()
        act_binned["Bin"] = (act_binned["TimeSinceReference"] // 60).astype(int)
        activity_binned = act_binned.groupby("Bin")["Activity"].mean().reset_index()

        scale = yrange * 0.3
        fig.add_trace(
            go.Bar(
                x=activity_binned["Bin"],  # minutes
                y=(activity_binned["Activity"] / activity_binned["Activity"].max())
                * scale,
                base=ymin,
                width=1,  # 1 min wide
                name="Activity (1-min bins)",
                marker_color="purple",
                opacity=0.5,
                yaxis="y",
            )
        )
        fig.update_layout(bargap=0)

    # --- Peaks ---
    if peak_times:
        peak_series = main_df.set_index("TimeSinceReference")[main_signal_col]
        peak_vals = peak_series.reindex(pd.Index(peak_times)).dropna()
        fig.add_trace(
            go.Scatter(
                x=peak_vals.index / 60,
                y=peak_vals.values,
                mode="markers",
                name="Peaks",
                marker=dict(color="magenta", size=5, symbol="cross"),
                yaxis="y",
            )
        )

    # --- Pre-peaks ---
    if pre_peak_times:
        pre_series = main_df.set_index("TimeSinceReference")[main_signal_col]
        pre_vals = pre_series.reindex(pd.Index(pre_peak_times)).dropna()
        fig.add_trace(
            go.Scatter(
                x=pre_vals.index / 60,
                y=pre_vals.values,
                mode="markers",
                name="Pre-Peaks",
                marker=dict(color="gold", size=5, symbol="triangle-up"),
                yaxis="y",
            )
        )

    # --- Atmospheric Pressure ---
    if atm_pressure_data is not None and not atm_pressure_data.empty and "AtmPressure" in atm_pressure_data.columns:
        atm_filtered = atm_pressure_data[
            (atm_pressure_data["TimeSinceReference"] >= main_df["TimeSinceReference"].min()) &
            (atm_pressure_data["TimeSinceReference"] <= main_df["TimeSinceReference"].max())
        ]
        fig.add_trace(
            go.Scatter(
                x=atm_filtered["TimeSinceReference"] / 60,
                y=atm_filtered["AtmPressure"],
                mode="lines",
                name="Atm. Pressure",
                line=dict(color="steelblue", width=1, dash="dot"),
                yaxis="y3",
            )
        )

    # --- Behavior window shading ---
    if behavior_windows:
        for bw_start, bw_end in behavior_windows:
            fig.add_vrect(
                x0=bw_start / 60,
                x1=bw_end / 60,
                fillcolor="rgba(0,200,0,0.08)",
                line_width=0,
                layer="below",
            )

    # --- Layout ---
    fig.update_layout(
        title=title,
        xaxis_title="Time (minutes)",
        yaxis=dict(title=main_signal_label, color=main_color, range=[ymin, ymax]),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right", color="red"),
        yaxis3=dict(title="Atm. Pressure (mmHg)", overlaying="y", side="right", anchor="free", position=1.0, color="steelblue"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        hovermode="x unified",
    )

    return fig


def create_static_plot(
    main_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    peak_times: list[float],
    pre_peak_times: list[float],
    save_path: str,
    title: str,
    main_signal_col: str,
    main_signal_label: str,
    atm_pressure_data: pd.DataFrame | None = None,
    behavior_windows: list[tuple[float, float]] | None = None,
):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ymin, ymax = main_df[main_signal_col].min(), main_df[main_signal_col].max()
    yrange = ymax - ymin

    # --- Main signal ---
    main_color = "black" if main_signal_label == "Pressure" else "green"
    ax1.plot(
        main_df["TimeSinceReference"] / 60,
        main_df[main_signal_col],
        color=main_color,
        label=main_signal_label,
    )
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel(main_signal_label, color=main_color)
    ax1.set_ylim(ymin, ymax)

    # --- Activity (1-min bins) ---
    if not activity_df.empty:
        act_binned = activity_df.copy()
        act_binned["Bin"] = (act_binned["TimeSinceReference"] // 60).astype(int)
        activity_bars = act_binned.groupby("Bin")["Activity"].mean().reset_index()

        scale = yrange * 0.3
        ax1.bar(
            activity_bars["Bin"],  # minutes
            (activity_bars["Activity"] / activity_bars["Activity"].max()) * scale,
            bottom=ymin,
            color="purple",
            alpha=0.5,
            width=1.0,
            label="Activity (1-min bins)",
            zorder=0,
        )

    # --- Temperature ---
    if not temp_df.empty:
        ax2 = ax1.twinx()
        ax2.plot(
            temp_df["TimeSinceReference"] / 60,
            temp_df["Temp"],
            color="red",
            label="Temperature",
        )
        ax2.set_ylabel("Temperature", color="red")

    # --- Peaks ---
    if peak_times:
        peak_series = main_df.set_index("TimeSinceReference")[main_signal_col]
        peak_vals = peak_series.reindex(pd.Index(peak_times)).dropna()
        ax1.scatter(
            peak_vals.index / 60,
            peak_vals.values,
            color="magenta",
            marker="x",
            s=30,
            label="Peaks",
        )

    # --- Pre-peaks ---
    if pre_peak_times:
        pre_series = main_df.set_index("TimeSinceReference")[main_signal_col]
        pre_vals = pre_series.reindex(pd.Index(pre_peak_times)).dropna()
        ax1.scatter(
            pre_vals.index / 60,
            pre_vals.values,
            color="gold",
            marker="^",
            s=30,
            label="Pre-Peaks",
        )

    # --- Atmospheric Pressure ---
    if atm_pressure_data is not None and not atm_pressure_data.empty and "AtmPressure" in atm_pressure_data.columns:
        atm_filtered = atm_pressure_data[
            (atm_pressure_data["TimeSinceReference"] >= main_df["TimeSinceReference"].min()) &
            (atm_pressure_data["TimeSinceReference"] <= main_df["TimeSinceReference"].max())
        ]
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(
            atm_filtered["TimeSinceReference"] / 60,
            atm_filtered["AtmPressure"],
            color="steelblue",
            linestyle="dotted",
            label="Atm. Pressure",
            linewidth=1,
        )
        ax3.set_ylabel("Atm. Pressure (mmHg)", color="steelblue")

    # --- Behavior window shading ---
    if behavior_windows:
        for bw_start, bw_end in behavior_windows:
            ax1.axvspan(bw_start / 60, bw_end / 60, alpha=0.08, color="green")

    # --- Final touches ---
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def filter_df_by_time(
    df: pd.DataFrame,
    start: float,
    end: float,
) -> pd.DataFrame:
    """Filter a dataframe to rows where TimeSinceReference lies within [start, end]."""
    return df[(df["TimeSinceReference"] >= start) & (df["TimeSinceReference"] <= end)]


def filter_times_to_range(
    times: Sequence[float],
    start: float,
    end: float,
) -> list[float]:
    """Return only time values within [start, end]."""
    times_array = np.asarray(times, dtype=np.float64)
    filtered = times_array[(times_array >= start) & (times_array <= end)]
    return cast(list[float], filtered.tolist())
