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
- Computes Ttot variability power spectral density (PSD) for each behavioural window
- Exports Excel summaries, interactive HTML plots, SVG figures, and breath-by-breath CSVs
- Exports PSD metrics, spectra, rejection/QC data, comparison spectra, and summary figures
- Exports a Session Summary sheet with overall and binned atmospheric pressure metrics (mean, SD, min, max) across the full recording

The pressure timeline uses the absolute video start as `0 s`. Telemetry samples
before the video therefore have negative times, and BORIS event times remain in
their original video-relative coordinates. The pipeline does not snap an event
to the nearest telemetry sample.

Only behaviour windows at least 30 seconds long, within the 90-minute analysis
limit, and fully covered by telemetry are analysed. Partially covered and
unavailable windows are excluded and reported in the run log and
`analysis_config_<timestamp>.json`. The full telemetry session is still retained
for signal processing, the full-recording plot, and session atmospheric-pressure
summaries. Pressure outages longer than one second split the continuous coverage
range and are filtered as separate segments. A one-second margin at each segment
edge is also excluded to avoid filter-edge contamination, so a behaviour
crossing an outage or margin is not treated as fully covered.

### Injection‑aligned photometry analysis

- Aligns photometry signals to an injection or event time
- Detects photometry peaks and per‑peak metrics
- Bins signals and peaks relative to injection (mean ± SEM)
- Optionally integrates temperature and activity data from telemetry implants
- Exports Excel summaries and plots

---

## Requirements

- Python 3.11+
- Windows or macOS

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # macOS
venv\Scripts\activate.bat   # Windows
```

Install the project and dependencies:

```bash
python -m pip install -e .
```

For development checks, install the optional tooling:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests main.py scripts
```

If you use the optional clustering tuning script:

```bash
python -m pip install -e ".[tuning]"
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

To run multiple pressure analyses without the GUI, create an Excel `.xlsx` file
with one row per recording. Native Excel files are recommended because dates are
stored independently of their display formatting.

```text
telemetry_file,event_file,behaviour,video_time,bin_size
C:/path/to/recording_1_ponemah.csv,C:/path/to/recording_1_boris.csv,Time spent sleeping,15/05/2024 10:46:46 AM,10
C:/path/to/recording_2_ponemah.csv,C:/path/to/recording_2_boris.csv,Time spent sleeping,12/01/2025 09:01:48 AM,10
```

| Column | Description |
| --- | --- |
| `telemetry_file` | Full path to the Ponemah `.csv` or `.ascii` file |
| `event_file` | Full path to the BORIS `.csv` file |
| `behaviour` | Behaviour label to analyse — must match the BORIS label exactly |
| `video_time` | Absolute date/time at video elapsed time zero (e.g. `15/05/2024 10:46:46 AM`) |
| `bin_size` | Bin size in seconds (integer, e.g. `10`) |

Older batch files containing a `probe_time` column remain valid. The batch runner
warns once and ignores that legacy column because telemetry timestamps now define
the telemetry reference.

Enter `video_time` as a real Excel date/time including seconds. The cell may use
any display format; showing seconds (for example `dd/mm/yyyy hh:mm:ss AM/PM`) is
recommended only so they can be checked visually. ISO text such as
`2024-04-25T10:07:06` is also accepted. CSV configs remain supported for backward
compatibility, but `.xlsx` avoids Excel reinterpreting timestamp text whenever a
CSV is opened or saved.

From the repository root (the folder containing this `README.md`), run the
script using its full relative path:

```bash
python scripts/batch_run.py batch_config.xlsx
```

The batch runner is inside the `scripts` folder, so do **not** run
`python batch_run.py` from the repository root. If your config file is elsewhere,
provide its path, for example:

```bash
python scripts/batch_run.py C:/path/to/batch_config.xlsx
```

By default, outputs are written beside each telemetry file under
`extracted_data/<recording_name>_PressureAnalysis/`, identical to a GUI run. To
put every per-recording result folder under one chosen directory, run:

```bash
python scripts/batch_run.py batch_config.xlsx --output-dir C:/path/to/batch-results
```

Each recording normally gets `<telemetry-name>_PressureAnalysis`. If any two
rows would resolve to the same analysis folder (including repeated analyses of
one telemetry file), the batch stops with a non-zero exit status before analysis
instead of allowing results to overwrite each other. Rename those telemetry
files or use separate batch runs/output directories.

Failed recordings are skipped and reported at the end — the rest of the batch still completes.

---

## Outputs

Pressure analysis is written beside the telemetry file under
`extracted_data/<recording_name>_PressureAnalysis/`. The folder contains:

- The main Excel workbook with:
  - **Summary Data** — valid respiratory periods per behavioural window
  - **Atmospheric Pressure** — overall and time-binned atmospheric pressure statistics when an APR/BARO channel is present and export is enabled
  - **Global Summary**, **Per Bin**, **Per Period**, and **Per Window** — respiratory metrics at each scope
  - **PSD Per Window**, **PSD Pooled**, and **PSD Segment Summary** — Ttot variability spectra and quality-control summaries
- `PSD/` with the standalone `ttot_psd_summary.xlsx` workbook, machine-readable CSV exports, rejected-segment details, and PNG/SVG QC figures
- `Ttot_Traces/` with breath-by-breath Ttot traces for each valid behavioural window
- Interactive HTML plots and static SVG figures for full recordings and behavioural windows
- `logs/analysis_config_<timestamp>.json` describing alignment, telemetry
  coverage, per-window inclusion status, and PSD parameters

Photometry analysis is written under
`extracted_data/<recording_name>_PhotometryAnalysis/` and contains:

- An Excel workbook with **Binned Data** and **Raw Data** sheets
- Interactive HTML and static SVG plots
- `analysis_config_<timestamp>.json` describing the alignment and binning settings

### Ttot PSD method

The primary PSD calculation follows the supplied MATLAB workflow:

- Breath durations are divided into 50-second segments and resampled at 10 Hz using cubic-spline interpolation
- Each valid respiratory period is trimmed by 10 breaths at each end before segmentation
- Resampled Ttot is mean-centred without additional detrending
- Welch PSD uses a symmetric 100-sample Hamming window, 50% overlap, and a 2048-point FFT
- Segment spectra are averaged within each behavioural window; a separate `N/3` Welch-window result is exported for comparison

At 10 Hz, the primary 100-sample Welch window has a nominal frequency resolution of approximately 0.1 Hz. Activity near 0.02 Hz should therefore be interpreted as very-low-frequency or near-DC activity, not as a precisely resolved 0.02 Hz peak. The exported AUC band is an exploratory mouse adaptation and should be interpreted alongside the PSD curve and QC fields.

---

## Notes

- `pyproject.toml` contains package metadata and runtime dependencies
- Optional clustering tuning dependencies are available with `.[tuning]`
- The pipeline prioritises robustness and interpretability over speed
- Intended for research and analysis use, not real‑time or clinical deployment
- The GUI remembers the last folder you browsed to and reopens there on the next run
- The "Reset Saved Settings" option is in the `···` menu at the top left of the window
