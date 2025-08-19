import plotly.graph_objects as go
from typing import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
from typing import List, Tuple

from src.core.adaptive_algorithms import get_nearest_points


# --- Main Full Trace Export ---
def export_full_time_range_plot(
    main_data: pd.DataFrame,             # pressure OR photometry
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame,
    valid_peak_times_all: List[float],
    valid_pre_peak_times_all: List[float],
    min_time: float,
    max_time: float,
    behaviour_to_plot: str,
    full_trace_folder: str,
    file_base: str,
    main_signal_col: str,                # e.g. "SmoothedPressure" or "dFF"
    main_signal_label: str               # e.g. "Pressure" or "Photometry"
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
        main_filtered, temp_filtered, activity_filtered,
        valid_peak_times, valid_pre_peak_times,
        title, main_signal_col, main_signal_label
    )
    save_path_html = os.path.join(
        full_trace_folder,
        f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.html"
    )
    fig_html.write_html(save_path_html)
    print(f"Saved full trace HTML: {save_path_html}")

    # --- Static plot ---
    save_path_svg = os.path.join(
        full_trace_folder,
        f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.svg"
    )
    create_static_plot(
        main_filtered, temp_filtered, activity_filtered,
        valid_peak_times, valid_pre_peak_times,
        save_path_svg, title, main_signal_col, main_signal_label
    )
    print(f"Saved full trace SVG: {save_path_svg}")


# --- Extracted helper for peaks ---
def prepare_peaks_for_range(
    valid_peak_times_all: List[float],
    valid_pre_peak_times_all: List[float],
    min_time: float,
    max_time: float
) -> Tuple[List[float], List[float]]:
    valid_peak_times = filter_times_to_range(
        valid_peak_times_all, min_time, max_time)
    valid_pre_peak_times = filter_times_to_range(
        valid_pre_peak_times_all, min_time, max_time)

    # Store original lengths
    original_peak_len = len(valid_peak_times)
    original_pre_peak_len = len(valid_pre_peak_times)

    # Trim both to match
    min_len = min(original_peak_len, original_pre_peak_len)
    trimmed_peaks = original_peak_len - min_len
    trimmed_pre_peaks = original_pre_peak_len - min_len

    if trimmed_peaks > 0 or trimmed_pre_peaks > 0:
        print(
            f"Trimmed {trimmed_peaks} peaks and {trimmed_pre_peaks} pre-peaks to match lengths.")

    return valid_peak_times[:min_len], valid_pre_peak_times[:min_len]



# --- Main Behavior-by-Window Export ---
def export_behavior_images_interactive(
    time_windows: List[Tuple[float, float]],
    pressure_data: pd.DataFrame,
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame,
    valid_peak_times_all: List[float],
    valid_pre_peak_times_all: List[float],
    behaviour_to_plot: str,
    html_save_folder: str,
    svg_save_folder: str,
    file_base: str,
    main_signal_col: str, 
    main_signal_label: str
):

    for i, (start_time, end_time) in enumerate(time_windows):
        # Downsample pressure data for faster plotting
        pressure_segment = filter_df_by_time(
            pressure_data, start_time, end_time)
        temp_segment = filter_df_by_time(temp_data, start_time, end_time)
        activity_segment = filter_df_by_time(
            activity_data, start_time, end_time)

        peak_times = filter_times_to_range(
            valid_peak_times_all, start_time, end_time)
        pre_peak_times = filter_times_to_range(
            valid_pre_peak_times_all, start_time, end_time)

        if pressure_segment.empty or temp_segment.empty or activity_segment.empty:
            print(
                f"Skipping plot for window {start_time}-{end_time}: no data.")
            continue

        title = f"{behaviour_to_plot} from {start_time:.2f} to {end_time:.2f}"

        html_path = os.path.join(
            html_save_folder,
            f"{behaviour_to_plot}_behavior_{i}_from_{start_time:.2f}_to_{end_time:.2f}.html"
        )
        fig_html = create_interactive_plot(
            pressure_segment, temp_segment, activity_segment,
            peak_times, pre_peak_times, title, main_signal_col, main_signal_label
        )
        fig_html.write_html(html_path)
        print(f"Saved behavior HTML: {html_path}")

        svg_path = os.path.join(
            svg_save_folder,
            f"{behaviour_to_plot}_behavior_{i}_from_{start_time:.2f}_to_{end_time:.2f}.svg"
        )
        create_static_plot(
            pressure_segment, temp_segment, activity_segment,
            peak_times, pre_peak_times, svg_path, title, main_signal_col, main_signal_label
        )
        print(f"Saved behavior SVG: {svg_path}")

    print("Finished exporting all interactive and static plots to folders.")


# --- Plot Helpers ---
def create_interactive_plot(
    main_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    peak_times: List[float],
    pre_peak_times: List[float],
    title: str,
    main_signal_col: str,
    main_signal_label: str     
) -> go.Figure:
    fig = go.Figure()

    # Main signal (pressure/photometry)
    fig.add_trace(go.Scatter(
        x=main_df['TimeSinceReference'],
        y=main_df[main_signal_col],
        mode='lines',
        name=main_signal_label,
        line=dict(color='black', width=1)
    ))

    # Temperature (right y-axis)
    fig.add_trace(go.Scatter(
        x=temp_df['TimeSinceReference'],
        y=temp_df['Temp'],
        mode='lines',
        name='Temperature',
        line=dict(color='red', width=1),
        yaxis='y2'
    ))

    # Activity (overlay on main axis)
    fig.add_trace(go.Scatter(
        x=activity_df['TimeSinceReference'],
        y=activity_df['Activity'],
        mode='lines',
        name='Activity',
        line=dict(color='green', width=1, dash='dot')
    ))

    # Peaks
    if len(peak_times) > 0:
        peak_series = main_df.set_index('TimeSinceReference')[main_signal_col]
        peak_series_unique = peak_series[~peak_series.index.duplicated(
            keep='first')]
        peak_times_unique = pd.Index(peak_times).drop_duplicates()
        peak_vals = peak_series_unique.reindex(peak_times_unique).dropna()

        peak_times_filtered = [t for t, v in zip(
            peak_times, peak_vals) if not pd.isna(v)]
        fig.add_trace(go.Scatter(
            x=peak_times_filtered,
            y=peak_vals.values,
            mode='markers',
            name=f"{main_signal_label} Peaks",
            marker=dict(color='magenta', size=6, symbol='cross')
        ))

    # Pre-peaks
    if len(pre_peak_times) > 0:
        pre_series = main_df.set_index('TimeSinceReference')[main_signal_col]
        pre_vals = pre_series.reindex(pre_peak_times).dropna()

        pre_times_filtered = [t for t, v in zip(
            pre_peak_times, pre_vals) if not pd.isna(v)]
        fig.add_trace(go.Scatter(
            x=pre_times_filtered,
            y=pre_vals.values,
            mode='markers',
            name=f"{main_signal_label} Pre-Peaks",
            marker=dict(color='gold', size=6, symbol='triangle-up')
        ))

    # Layout (with secondary y-axis)
    fig.update_layout(
        title=title,
        xaxis_title='Time Since Reference (seconds)',
        yaxis_title=main_signal_label,
        yaxis2=dict(
            title='Temperature',
            overlaying='y',
            side='right',
            color='red'
        ),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        height=600,
        hovermode='x unified',
        xaxis=dict(range=[
            main_df['TimeSinceReference'].min(),
            main_df['TimeSinceReference'].max()
        ])
    )

    return fig

def create_static_plot(
    main_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    peak_times: List[float],
    pre_peak_times: List[float],
    save_path: str,
    title: str,
    main_signal_col: str,     # e.g. "SmoothedPressure" or "dFF"
    main_signal_label: str    # e.g. "Pressure" or "Photometry"
):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Main signal ---
    ax1.plot(
        main_df['TimeSinceReference'],
        main_df[main_signal_col],
        color='black',
        label=f"{main_signal_label}"
    )
    ax1.set_xlabel("Time Since Reference (seconds)")
    ax1.set_ylabel(main_signal_label, color='black')

    # --- Temperature ---
    if not temp_df.empty:
        ax2 = ax1.twinx()
        ax2.plot(temp_df['TimeSinceReference'], temp_df['Temp'],
                 color='red', label="Temperature")
        ax2.set_ylabel("Temperature", color='red')

    # --- Activity ---
    if not activity_df.empty:
        ax1.plot(activity_df['TimeSinceReference'], activity_df['Activity'],
                 color='green', linestyle='dotted', label="Activity")

    # --- Peaks ---
    if len(peak_times) > 0:
        peak_series = main_df.set_index('TimeSinceReference')[main_signal_col]
        peak_series_unique = peak_series[~peak_series.index.duplicated(
            keep='first')]
        peak_times_unique = pd.Index(peak_times).drop_duplicates()
        peak_vals = peak_series_unique.reindex(peak_times_unique).dropna()

        peak_times_filtered = [t for t, v in zip(
            peak_times, peak_vals) if not pd.isna(v)]
        ax1.scatter(peak_times_filtered, peak_vals.values,
                    color='magenta', marker='x', label="Peaks")

    # --- Pre-Peaks ---
    if len(pre_peak_times) > 0:
        pre_series = main_df.set_index('TimeSinceReference')[main_signal_col]
        pre_vals = pre_series.reindex(pre_peak_times).dropna()
        pre_times_filtered = [t for t, v in zip(
            pre_peak_times, pre_vals) if not pd.isna(v)]
        ax1.scatter(pre_times_filtered, pre_vals.values,
                    color='gold', marker='^', label="Pre-Peaks")

    # --- Finalise plot ---
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True)
    fig.savefig(save_path, format="svg", bbox_inches='tight')
    plt.close(fig)


def filter_df_by_time(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return df[(df['TimeSinceReference'] >= start) & (df['TimeSinceReference'] <= end)]


def filter_times_to_range(times: Sequence[float], start: float, end: float) -> List[float]:
    times_array = np.asarray(times, dtype=np.float64)
    filtered = times_array[(times_array >= start) & (times_array <= end)]
    return filtered.tolist()
