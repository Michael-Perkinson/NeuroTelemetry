from src.controllers.photometry_controller import (
    load_photometry_data,
    run_photometry_pipeline,
)
import sys
from pathlib import Path

# -------------------------------------------------
# Ensure project root is on sys.path
# -------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------
# Data paths
# -------------------------------------------------
data_root = Path.home() / "data" / "Andys"

photometry_path = (
    data_root
    / "2025-07-16 VP3 MET DIESTRUS SALINE_Data OPTION 2 INC.csv"
)

telemetry_path = (
    data_root
    / "2025-07-16 VP3 Saline TEMP and ACT ascii.csv"
)

# -------------------------------------------------
# Load data
# -------------------------------------------------
photometry_df, telemetry_df = load_photometry_data(
    photometry_path, telemetry_path
)

# -------------------------------------------------
# Run pipeline
# -------------------------------------------------
repo_root = Path.home() / "code" / "pressure_analysis"
output_path = repo_root / "test_outputs" / "photometry"

run_photometry_pipeline(
    telemetry_df=telemetry_df,
    photometry_df=photometry_df,
    photometry_align_time="16/07/2025 11:08:00 AM",
    injection_sec=2820,
    pre_min=30,
    post_min=360,
    bin_min=30,
    telemetry_path=output_path,
    log_callback=print,
)
