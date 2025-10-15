import sys
from datetime import datetime
from pathlib import Path

# Add the project root (parent of "tests") to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.controllers.pressure_controller import load_data, run_pressure_pipeline

# --- 1. Load your data files ---
base_dir = (
    Path.home() / "Documents" / "code" / "pressure_analysis" / "data" / "B5 night"
)

telemetry_path = base_dir / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
event_path = base_dir / "B5 Pro NIGHT 11-01-2025 BORIS.csv"


telemetry_df, event_df = load_data(telemetry_path, event_path)

# --- 2. Set test parameters ---
# pick one from event_df['event'].unique()
behaviour_to_plot = "Time spent sleeping"
# match recording start/reference
probe_time = "01/11/2025 05:05:09 PM"
video_time = "01/11/2025 04:59:59 PM"
bin_size_sec = 10

# Create unique timestamp folder (e.g. 2025-01-11_09-58-02)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_path = (
    Path.home()
    / "documents"
    / "code"
    / "pressure_analysis"
    / "test_outputs"
    / f"pressure_run_{timestamp}"
)

# --- 3. Run the pipeline ---
run_pressure_pipeline(
    telemetry_df=telemetry_df,
    event_df=event_df,
    behaviour_to_plot=behaviour_to_plot,
    probe_time=probe_time,
    video_time=video_time,
    bin_size_sec=bin_size_sec,
    output_path=output_path,
    log_callback=print,  # or None
)
