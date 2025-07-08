# File paths to the data files
# Modify this to the correct path for your data file
data_file_path = 'data/Day 2 (25-04-24) Pro - this is the NaNs file/B1 virgin Day 2 (25-04-24) ponemah.csv'

# Modify this to the correct path for your behavior file
behaviour_file_path = 'data/Day 2 (25-04-24) Pro - this is the NaNs file/B1 virgin Day 2 (25-04-24).csv'

# Specify the date and time of interest
# Format: Month/Day/Year Hour:Minute:Second AM/PM
probe_date_time = '04/25/2024 9:45:09 AM'
video_date_time = '04/25/2024 9:38:10 AM'

# Specify the behavior you want to plot
# Set to behaviour_to_plot 'None' to plot the entire trace or specify the behavior of interest
behaviour_to_plot = 'Time spent sleeping'

# Set time_windows to 'None' to plot the behaviour of interest or specify the time windows of interest

# Define buffer times around the behavior
buffer_time_before = 60  # Time in seconds before the behavior starts
buffer_time_after = 60  # Time in seconds after the behavior ends

# Toggle whether to show peaks in the pressure data
show_peaks = True  # Set to 'False' if you do not want to display peaks

# Select which occurrence of the behavior to plot
nth_longest = 2  # 1 for the longest, 2 for the second longest, etc.

# Restrict the duration of the behavior
restrict_duration = True  # Set to 'False' to ignore duration restrictions

# Minimum duration of the behavior in seconds
min_duration = 30  # Set minimum duration to 30 seconds

# Maximum duration in seconds (use 'float('inf')' for no limit)
max_duration = float('inf')

# Bin size for the exported data in seconds
bin_size_sec = 10
