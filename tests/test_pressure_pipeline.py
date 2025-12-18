from src.controllers.pressure_controller import (
    load_data,
    run_pressure_pipeline,
)
import sys
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# Ensure project root (NeuroTelemetry) is on sys.path
# -------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------
# Data paths
# -------------------------------------------------
data_root = (
    Path.home()
    / "Documents"
    / "Code"
    / "NeuroTelemetry"
    / "data"
    / "Day 2 (25-04-24) Pro"
)

telemetry_path = data_root / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
event_path = data_root / "B5 Pro NIGHT 11-01-2025 BORIS.csv"

telemetry_df, event_df = load_data(telemetry_path, event_path)

# -------------------------------------------------
# Parameters
# -------------------------------------------------
behaviour_to_plot = "Time spent sleeping"
probe_time = "01/11/2025 05:05:09 PM"
video_time = "01/11/2025 04:59:59 PM"
bin_size_sec = 10

# -------------------------------------------------
# Output
# -------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

repo_root = (
    Path.home()
    / "Documents"
    / "Code"
    / "NeuroTelemetry"
)

output_path = repo_root / "test_outputs" / f"pressure_run_{timestamp}"

# -------------------------------------------------
# Run pipeline
# -------------------------------------------------
run_pressure_pipeline(
    telemetry_df=telemetry_df,
    event_df=event_df,
    behaviour_to_plot=behaviour_to_plot,
    probe_time=probe_time,
    video_time=video_time,
    bin_size_sec=bin_size_sec,
    output_path=output_path,
    log_callback=print,
)
