import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.controllers.pressure_controller import load_data, run_pressure_pipeline

data_dir = PROJECT_ROOT / "data" / "B5" / "PM" / "Pro"

telemetry_path = data_dir / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
event_path = data_dir / "B5 Pro NIGHT 11-01-2025 BORIS.csv"

telemetry_df, event_df = load_data(telemetry_path, event_path)

behaviour_to_plot = "Time spent sleeping"
probe_time = "01/11/2025 05:05:09 PM"
video_time = "01/11/2025 04:59:59 PM"
bin_size_sec = 10

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = PROJECT_ROOT / "test_outputs" / f"pressure_run_{timestamp}"

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

print("Done:", output_path)
