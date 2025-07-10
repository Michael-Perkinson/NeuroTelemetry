import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
from typing import List, Tuple


def export_full_time_range_plot(pressure_data, temp_data, activity_data,
                                valid_peak_times_all, valid_pre_peak_times_all,
                                min_time, max_time, behaviour_to_plot, full_trace_folder, file_base):
    """
    Export full time range interactive (HTML) and static (SVG) plots.
    """

    # --- Filter Data ---
    pressure_filtered = filter_df_by_time(pressure_data, min_time, max_time)
    temp_filtered = filter_df_by_time(temp_data, min_time, max_time)
    activity_filtered = filter_df_by_time(activity_data, min_time, max_time)

    valid_peak_times = filter_times_to_range(
        valid_peak_times_all, min_time, max_time)
    valid_pre_peak_times = filter_times_to_range(
        valid_pre_peak_times_all, min_time, max_time)

    # --- Plotly HTML Plot ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pressure_filtered['TimeSinceReference'],
                             y=pressure_filtered['Pressure'],
                             mode='lines', name='Pressure', line=dict(color='black', width=1)))

    fig.add_trace(go.Scatter(x=temp_filtered['TimeSinceReference'],
                             y=temp_filtered['Temp'],
                             mode='lines', name='Temperature', line=dict(color='red', width=1),
                             yaxis='y2'))

    fig.add_trace(go.Scatter(x=activity_filtered['TimeSinceReference'],
                             y=activity_filtered['Activity'],
                             mode='lines', name='Activity', line=dict(color='green', width=1, dash='dot')))

    fig.add_trace(go.Scatter(x=valid_peak_times,
                             y=pressure_data.loc[pressure_data['TimeSinceReference'].isin(
                                 valid_peak_times), 'Pressure'],
                             mode='markers', name='Peaks', marker=dict(color='magenta', size=6, symbol='cross')))

    fig.add_trace(go.Scatter(x=valid_pre_peak_times,
                             y=pressure_data.loc[pressure_data['TimeSinceReference'].isin(
                                 valid_pre_peak_times), 'Pressure'],
                             mode='markers', name='Pre-Peaks', marker=dict(color='gold', size=6, symbol='triangle-up')))

    fig.update_layout(
        title=f"{behaviour_to_plot}: Full Time Range {min_time:.2f} to {max_time:.2f}",
        xaxis_title='Time Since Reference (seconds)',
        yaxis_title='Pressure',
        yaxis2=dict(title='Temperature', overlaying='y',
                    side='right', color='red'),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        height=600,
        hovermode='x unified',
        xaxis=dict(range=[min_time, max_time])
    )

    save_path_html = os.path.join(
        full_trace_folder, f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.html")
    fig.write_html(save_path_html)
    print(f"Saved full trace HTML: {save_path_html}")

    # --- Matplotlib SVG Plot ---
    fig_mpl, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(pressure_filtered['TimeSinceReference'],
             pressure_filtered['Pressure'], color='black', label="Pressure")
    ax1.set_xlabel("Time Since Reference (seconds)")
    ax1.set_ylabel("Pressure", color='black')

    ax2 = ax1.twinx()
    ax2.plot(temp_filtered['TimeSinceReference'],
             temp_filtered['Temp'], color='red', label="Temperature")
    ax2.set_ylabel("Temperature", color='red')

    ax1.plot(activity_filtered['TimeSinceReference'], activity_filtered['Activity'],
             color='green', linestyle='dotted', label="Activity")

    if len(valid_peak_times) > 0:
        ax1.scatter(valid_peak_times, pressure_data.loc[pressure_data['TimeSinceReference'].isin(valid_peak_times), 'Pressure'],
                    color='magenta', marker='x', label="Peaks")
    if len(valid_pre_peak_times) > 0:
        ax1.scatter(valid_pre_peak_times, pressure_data.loc[pressure_data['TimeSinceReference'].isin(valid_pre_peak_times), 'Pressure'],
                    color='gold', marker='^', label="Pre-Peaks")

    ax1.legend(loc="upper left")
    ax1.grid(True)

    save_path_svg = os.path.join(
        full_trace_folder, f"{file_base}_full_trace_{min_time:.2f}_to_{max_time:.2f}.svg")
    fig_mpl.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.close(fig_mpl)

    print(f"Saved full trace SVG: {save_path_svg}")


def export_behavior_images_interactive(time_windows: List[Tuple[float, float]],
                                       pressure_data: pd.DataFrame,
                                       temp_data: pd.DataFrame,
                                       activity_data: pd.DataFrame,
                                       valid_peak_times_all: List[float],
                                       valid_pre_peak_times_all: List[float],
                                       behaviour_to_plot: str,
                                       html_save_folder: str,
                                       svg_save_folder: str,
                                       file_base: str):
    """
    Export interactive and static plots for each behavioral time window.
    """

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

        # Plotly Interactive HTML
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pressure_segment['TimeSinceReference'], y=pressure_segment['Pressure'],
                                 mode='lines', name='Pressure', line=dict(color='black', width=1)))
        fig.add_trace(go.Scatter(x=temp_segment['TimeSinceReference'], y=temp_segment['Temp'],
                                 mode='lines', name='Temperature', line=dict(color='red', width=1), yaxis='y2'))
        fig.add_trace(go.Scatter(x=activity_segment['TimeSinceReference'], y=activity_segment['Activity'],
                                 mode='lines', name='Activity', line=dict(color='green', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=peak_times, y=pressure_data.loc[pressure_data['TimeSinceReference'].isin(peak_times), 'Pressure'],
                                 mode='markers', name='Peaks', marker=dict(color='magenta', size=6, symbol='cross')))
        fig.add_trace(go.Scatter(x=pre_peak_times, y=pressure_data.loc[pressure_data['TimeSinceReference'].isin(pre_peak_times), 'Pressure'],
                                 mode='markers', name='Pre-Peaks', marker=dict(color='gold', size=6, symbol='triangle-up')))

        fig.update_layout(
            title=f"{behaviour_to_plot} from {start_time:.2f} to {end_time:.2f}",
            xaxis_title='Time Since Reference (seconds)',
            yaxis_title='Pressure',
            yaxis2=dict(title='Temperature', overlaying='y',
                        side='right', color='red'),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1),
            height=600,
            hovermode='x unified',
        )

        html_path = os.path.join(
            html_save_folder, f"{behaviour_to_plot}_behavior_{i}_from_{start_time:.2f}_to_{end_time:.2f}.html")
        fig.write_html(html_path)
        print(f"Saved behavior HTML: {html_path}")

        # Static Matplotlib SVG
        fig_mpl, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(pressure_segment['TimeSinceReference'],
                 pressure_segment['Pressure'], color='black', label="Pressure")
        ax1.set_xlabel("Time Since Reference (seconds)")
        ax1.set_ylabel("Pressure", color='black')

        ax2 = ax1.twinx()
        ax2.plot(temp_segment['TimeSinceReference'],
                 temp_segment['Temp'], color='red', label="Temperature")
        ax2.set_ylabel("Temperature", color='red')

        ax1.plot(activity_segment['TimeSinceReference'], activity_segment['Activity'],
                 color='green', linestyle='dotted', label="Activity")

        if len(peak_times) > 0:
            ax1.scatter(peak_times, pressure_data.loc[pressure_data['TimeSinceReference'].isin(peak_times), 'Pressure'],
                        color='magenta', marker='x', label="Peaks")
        if len(pre_peak_times) > 0:
            ax1.scatter(pre_peak_times, pressure_data.loc[pressure_data['TimeSinceReference'].isin(pre_peak_times), 'Pressure'],
                        color='gold', marker='^', label="Pre-Peaks")

        ax1.legend(loc="upper left")
        ax1.grid(True)

        svg_path = os.path.join(
            svg_save_folder, f"{behaviour_to_plot}_behavior_{i}_from_{start_time:.2f}_to_{end_time:.2f}.svg")
        fig_mpl.savefig(svg_path, format="svg", bbox_inches='tight')
        plt.close(fig_mpl)

        print(f"Saved behavior SVG: {svg_path}")

    print("Finished exporting all interactive and static plots to folders.")


# --- Shared Filtering Utilities ---

def filter_df_by_time(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return df[(df['TimeSinceReference'] >= start) & (df['TimeSinceReference'] <= end)]


def filter_times_to_range(times: List[float], start: float, end: float) -> np.ndarray:
    times_array = np.array(times, dtype=np.float64)
    return times_array[(times_array >= start) & (times_array <= end)]
