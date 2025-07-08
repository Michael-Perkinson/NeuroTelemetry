# # Analysis and Plotting Process
#
# This script follows a sequence of steps to analyze and plot behavioral data:
#
# 1. **Select Time Windows**: Identifies relevant time windows based on the behavior to analyze.
# 2. **Analyze Sleeping Peaks**: Detects peaks within the selected time windows.
# 3. **Identify Valid Periods and Calculate Metrics**: Filters valid periods and computes respiratory metrics.
# 4. **Plot the Data**: Visualizes the analyzed data, showing pressure, temperature, and activity over time.
#


# Step 0: Retrieve data
data, behaviour_data = retrieve_telemetry_data()

# Step 1: Extract and process data
pressure_data, smoothed_pressure, temp_data, activity_data, numerical_data, new_reference_timestamp = extract_and_process_data(
    data, behaviour_data)

# Step 2: Select time windows
time_windows = select_time_windows(behaviour_to_plot, behaviour_data,
                                   buffer_time_before, buffer_time_after, min_duration, new_reference_timestamp)

# Step 3: Analyze sleeping peaks
results = analyze_sleeping_peaks_for_windows(
    time_windows, smoothed_pressure, pressure_data)

# start_time, end_time, peaks, peak_times, pre_peak_indies, pre_peak_times = results[0]

all_peak_times = []
all_pre_peak_times = []

for window_key, window_results in results.items():
    for result in window_results:
        # Unpack the result tuple
        _, _, _, peak_times, _, pre_peak_times = result
        all_peak_times.extend(peak_times)
        all_pre_peak_times.extend(pre_peak_times)

# Step 4: Identify valid periods and calculate metrics
valid_peak_times_all, valid_pre_peak_times_all, updated_valid_periods, all_metrics = find_valid_periods(
    results, smoothed_pressure, pressure_data, temp_data, activity_data, time_windows)

# Step 5: Create summary and detailed data for exporting
summary_data = create_summary_data(
    valid_peak_times_all, updated_valid_periods, time_windows)

max_time = pressure_data['TimeSinceReference'].max()
min_time = pressure_data['TimeSinceReference'].min()

# # Remove this for now
# plot_data(pressure_data['TimeSinceReference'].min(), pressure_data['TimeSinceReference'].max(), pressure_data, temp_data,
#           activity_data, all_peak_times, all_pre_peak_times, updated_valid_periods)

# Make folders for exporting graphs
html_save_folder, svg_save_folder, full_trace_folder, file_base = create_folders_for_graphs(
    data_file_path)

# Step 6: Export full time range plot
export_full_time_range_plot(pressure_data, temp_data, activity_data,
                            valid_peak_times_all, valid_pre_peak_times_all,
                            min_time, max_time, behaviour_to_plot, full_trace_folder, file_base)

print("Exported full time range plot")

# Step 7: Export images of the behavior
export_behavior_images_interactive(time_windows, pressure_data, temp_data, activity_data,
                                   valid_peak_times_all, valid_pre_peak_times_all, behaviour_to_plot, html_save_folder, svg_save_folder, file_base)

# Step 8: Export data to Excel
export_sleep_data_to_excel(summary_data, all_metrics, data_file_path)
