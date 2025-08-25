import sys
from pathlib import Path

from src.controllers.photometry_controller import (
    load_photometry_data,
    run_photometry_pipeline,
)

# Make sure src/ is importable
sys.path.append(str(Path(__file__).resolve().parent.parent))


# --- 1. Point to your files ---
base_path = Path("C:/Users/permi73p/Documents/Code/pressure_analysis/data/Andys")

photometry_path = base_path / (
    "2025-07-16 VP3 MET DIESTRUS SALINE_Data OPTION 2 INC.csv"
)

telemetry_path = base_path / ("2025-07-16 VP3 Saline TEMP and ACT ascii.csv")

# --- 2. Load with your wrapper (does pre-processing + sanity checks) ---
photometry_df, telemetry_df = load_photometry_data(photometry_path, telemetry_path)

# --- 3. Parameters (match what you’d normally set in the GUI) ---
# or whatever your align reference is
photometry_align_time = "16/07/2025 11:08:00 AM"
injection_sec = 2820
pre_minutes = 30
post_minutes = 360
bin_minutes = 30
base_path = Path("C:/Users/permi73p/Documents/Code/pressure_analysis")
output_path = base_path / "test_outputs" / "photometry"

# --- 4. Run pipeline ---
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
