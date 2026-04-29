# Telemetry Alignment Analysis Tool

A Python application for analysing physiological telemetry and photometry data aligned to behavioural events or experimental manipulations.

The tool is designed for real experimental data: noisy signals, imperfect timestamps, and heterogeneous file formats. It provides a simple GUI for configuration and produces analysis‑ready metrics and figures.

---

## What this does

The application supports two analysis modes.

### Behaviour‑aligned telemetry analysis

- Aligns pressure, temperature, and activity data to behavioural events
- Parses atmospheric pressure (APR/BARO column) from newer Ponemah recordings where present
- Detects respiratory cycles from pressure signals
- Identifies valid respiratory periods
- Computes respiratory timing, duty cycle, frequency, and drive
- Exports Excel summaries, interactive HTML plots, SVG figures, and breath‑by‑breath CSVs
- Exports a Session Summary sheet with overall and binned atmospheric pressure metrics (mean, SD, min, max) across the full recording

### Injection‑aligned photometry analysis

- Aligns photometry signals to an injection or event time
- Detects photometry peaks and per‑peak metrics
- Bins signals and peaks relative to injection (mean ± SEM)
- Optionally integrates temperature and activity data from telelmetry implants
- Exports Excel summaries and plots

---

## Requirements

- Python 3.10+
- Windows or macOS

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # macOS
venv\Scripts\activate.bat   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the application

From the project root:

```bash
python main.py
```

This launches the graphical interface.

---

## Batch Runs

To run multiple pressure analyses without the GUI, create a CSV with one row
per recording:

```csv
telemetry_file,event_file,behaviour,probe_time,video_time,bin_size
C:/path/to/recording_1_ponemah.csv,C:/path/to/recording_1_boris.csv,Time spent sleeping,15/05/2024 10:46:38 AM,15/05/2024 10:46:46 AM,10
C:/path/to/recording_2_ponemah.csv,C:/path/to/recording_2_boris.csv,Time spent sleeping,12/01/2025 09:01:53 AM,12/01/2025 09:01:48 AM,10
```

Then run:

```bash
python scripts/batch_run.py batch_config.csv
```

Outputs are written beside each telemetry file under
`extracted_data/<recording_name>_PressureAnalysis/`.

---

## Outputs

Each analysis run creates a self‑contained output folder containing:

- Run configuration (JSON)
- Excel workbook with sheets:
  - **Summary Data** — valid respiratory periods per behaviour window
  - **Atmospheric Pressure** — overall session stats (one row: mean/SD/min/max) followed by time-binned atmospheric pressure stats at a configurable interval (default 5 minutes); only present when an APR/BARO column exists in the recording
  - **Global Summary**, **Per Bin**, **Per Period**, **Per Window** — respiratory metrics at each scope
- Interactive HTML plots
- Static SVG figures
- CSV exports (where applicable)

---

## Notes

- `requirements.txt` contains runtime dependencies only
- The pipeline prioritises robustness and interpretability over speed
- Intended for research and analysis use, not real‑time or clinical deployment
- The GUI remembers the last folder you browsed to and reopens there on the next run
- The "Reset Saved Settings" option is in the `···` menu at the top left of the window
