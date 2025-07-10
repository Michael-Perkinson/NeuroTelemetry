import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
from typing import List, Tuple


# --- Main Full Trace Export ---
def export_full_time_range_plot(
    pressure_data: pd.DataFrame,
    temp_data: pd.DataFrame,
    activity_data: pd.DataFrame,
    valid_peak_times_all: List[float],
    valid_pre_peak_times_all: List[float],
    min_time: float,
    max_time: float,
    behaviour_to_plot: str,
    full_trace_folder: str,
    file_base: str
):
    pressure_filtered = filter_df_by_time(pressure_data, min_time, max_time)
    temp_filtered = filter_df_by_time(temp_data, min_time, max_time)
    activity_filtered = filter_df_by_time(activity_data, min_time, max_time)

    valid_peak_times = filter_times_to_range(
        valid_peak_times_all, min_time, max_time)
    valid_pre_peak_times = filter_times_to_range(
        valid_pre_peak_times_all, min_time, max_time)

    title = f"{behaviour_to_plot}: Full Time Range {min_time:.2f} to {max_time:.2f}"

    fig_html = create_interactive_plot(
        pressure_filtered, temp_filtered, activity_filtered,
        valid_peak_times, valid_pre_peak_times, title
    )
    save_path_html = os.path.join(
        full_trace_folder, f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.html"
    )
    fig_html.write_html(save_path_html)
    print(f"Saved full trace HTML: {save_path_html}")

    save_path_svg = os.path.join(
        full_trace_folder, f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.svg"
    )
    create_static_plot(
        pressure_filtered, temp_filtered, activity_filtered,
        valid_peak_times, valid_pre_peak_times, save_path_svg, title
    )
    print(f"Saved full trace SVG: {save_path_svg}")


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
    file_base: str
):
    valid_peak_times_all = np.array(valid_peak_times_all)
    valid_pre_peak_times_all = np.array(valid_pre_peak_times_all)

    for i, (start_time, end_time) in enumerate(time_windows):
        pressure_segment = filter_df_by_time(
            pressure_data, start_time, end_time).iloc[::10].copy()
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
            peak_times, pre_peak_times, title
        )
        fig_html.write_html(html_path)
        print(f"Saved behavior HTML: {html_path}")

        svg_path = os.path.join(
            svg_save_folder,
            f"{behaviour_to_plot}_behavior_{i}_from_{start_time:.2f}_to_{end_time:.2f}.svg"
        )
        create_static_plot(
            pressure_segment, temp_segment, activity_segment,
            peak_times, pre_peak_times, svg_path, title
        )
        print(f"Saved behavior SVG: {svg_path}")

    print("Finished exporting all interactive and static plots to folders.")


# --- Plot Helpers ---
def create_interactive_plot(
    pressure_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    peak_times: List[float],
    pre_peak_times: List[float],
    title: str
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pressure_df['TimeSinceReference'], y=pressure_df['SmoothedPressure'],
        mode='lines', name='Smoothed Pressure', line=dict(color='black', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=temp_df['TimeSinceReference'], y=temp_df['Temp'],
        mode='lines', name='Temperature', line=dict(color='red', width=1),
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=activity_df['TimeSinceReference'], y=activity_df['Activity'],
        mode='lines', name='Activity', line=dict(color='green', width=1, dash='dot')
    ))

    if peak_times:
        fig.add_trace(go.Scatter(
            x=peak_times,
            y=pressure_df.loc[pressure_df['TimeSinceReference'].isin(
                peak_times), 'SmoothedPressure'],
            mode='markers', name='Peaks', marker=dict(color='magenta', size=6, symbol='cross')
        ))
    if pre_peak_times:
        fig.add_trace(go.Scatter(
            x=pre_peak_times,
            y=pressure_df.loc[pressure_df['TimeSinceReference'].isin(
                pre_peak_times), 'SmoothedPressure'],
            mode='markers', name='Pre-Peaks', marker=dict(color='gold', size=6, symbol='triangle-up')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time Since Reference (seconds)',
        yaxis_title='Pressure',
        yaxis2=dict(title='Temperature', overlaying='y',
                    side='right', color='red'),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        height=600,
        hovermode='x unified',
        xaxis=dict(range=[
            pressure_df['TimeSinceReference'].min(),
            pressure_df['TimeSinceReference'].max()
        ])
    )

    return fig


def create_static_plot(
    pressure_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    peak_times: List[float],
    pre_peak_times: List[float],
    save_path: str,
    title: str
):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(pressure_df['TimeSinceReference'], pressure_df['SmoothedPressure'],
             color='black', label="Smoothed Pressure")
    ax1.set_xlabel("Time Since Reference (seconds)")
    ax1.set_ylabel("Pressure", color='black')

    ax2 = ax1.twinx()
    ax2.plot(temp_df['TimeSinceReference'], temp_df['Temp'],
             color='red', label="Temperature")
    ax2.set_ylabel("Temperature", color='red')

    ax1.plot(activity_df['TimeSinceReference'], activity_df['Activity'],
             color='green', linestyle='dotted', label="Activity")

    if peak_times:
        ax1.scatter(peak_times,
                    pressure_df.loc[pressure_df['TimeSinceReference'].isin(
                        peak_times), 'SmoothedPressure'],
                    color='magenta', marker='x', label="Peaks")
    if pre_peak_times:
        ax1.scatter(pre_peak_times,
                    pressure_df.loc[pressure_df['TimeSinceReference'].isin(
                        pre_peak_times), 'SmoothedPressure'],
                    color='gold', marker='^', label="Pre-Peaks")

    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True)
    fig.savefig(save_path, format="svg", bbox_inches='tight')
    plt.close(fig)


# --- Utilities ---
def filter_df_by_time(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return df[(df['TimeSinceReference'] >= start) & (df['TimeSinceReference'] <= end)]


def filter_times_to_range(times: List[float], start: float, end: float) -> np.ndarray:
    times_array = np.array(times, dtype=np.float64)
    return times_array[(times_array >= start) & (times_array <= end)]
