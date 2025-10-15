import sys
from pathlib import Path

from src.controllers.pressure_controller import load_data, run_pressure_pipeline

# Add the project root (parent of "tests") to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- 1. Load your data files ---
base_dir = Path.home() / "code" / "pressure_analysis" / \
    "data" / "B5" / "PM" / "Pro"

telemetry_path = base_dir / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
event_path = base_dir / "B5 Pro NIGHT 11-01-2025 BORIS.csv"


telemetry_df, event_df = load_data(telemetry_path, event_path)

# --- 2. Set test parameters ---
# pick one from event_df['event'].unique()
behaviour_to_plot = "Time spent sleeping"
# match recording start/reference
probe_time = "01/11/2025 05:05:09 PM"
video_time = "01/11/2025 04:59:59 PM"
bin_size_sec = 60
output_path = Path.home() / "code" / "pressure_analysis" / "test_outputs" / "pressure"

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
