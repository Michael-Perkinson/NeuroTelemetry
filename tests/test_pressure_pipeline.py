import sys
from pathlib import Path

from src.controllers.pressure_controller import load_data, run_pressure_pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))

base_dir = Path(r"C:\Users\permi73p\Documents\Code\pressure_analysis\data\night")

telemetry_path = base_dir / "B5 Pro NIGHT 11-01-2025 Ponemah.csv"
event_path = base_dir / "B5 Pro NIGHT 11-01-2025 BORIS.csv"


telemetry_df, event_df = load_data(telemetry_path, event_path)

behaviour_to_plot = "Time spent sleeping"
probe_time = "01/11/2025 05:05:09 PM"
video_time = "01/11/2025 04:59:59 PM"
bin_size_sec = 60
output_path = Path(
    r"C:\Users\permi73p\Documents\Code\pressure_analysis\extracted_data\pressure"
)

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
