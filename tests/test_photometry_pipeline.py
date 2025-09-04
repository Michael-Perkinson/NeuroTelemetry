import sys
from pathlib import Path

from src.controllers.photometry_controller import (
    load_photometry_data,
    run_photometry_pipeline,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

base_path = Path("C:/Users/permi73p/Documents/Code/pressure_analysis/data/Andys")

photometry_path = base_path / (
    "2025-07-16 VP3 MET DIESTRUS SALINE_Data OPTION 2 INC.csv"
)

telemetry_path = base_path / ("2025-07-16 VP3 Saline TEMP and ACT ascii.csv")

photometry_df, telemetry_df = load_photometry_data(photometry_path, telemetry_path)

photometry_align_time = "16/07/2025 11:08:00 AM"
injection_sec = 2820
pre_minutes = 30
post_minutes = 360
bin_minutes = 30
base_path = Path("C:/Users/permi73p/Documents/Code/pressure_analysis")
output_path = base_path / "test_outputs" / "photometry"

run_photometry_pipeline(
    telemetry_df=telemetry_df,
    photometry_df=photometry_df,
    photometry_align_time=photometry_align_time,
    injection_sec=injection_sec,
    pre_minutes=pre_minutes,
    post_minutes=post_minutes,
    bin_minutes=bin_minutes,
    output_path=output_path,
    log_callback=print,
)
