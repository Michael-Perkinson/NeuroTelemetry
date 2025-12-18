# Telemetry Alignment Analysis Tool

A Python application for analysing physiological telemetry and photometry data aligned to behavioural events or experimental manipulations.

The tool is designed for real experimental data: noisy signals, imperfect timestamps, and heterogeneous file formats. It provides a simple GUI for configuration and produces analysis‑ready metrics and figures.

---

## What this does

The application supports two analysis modes.

### Behaviour‑aligned telemetry analysis

- Aligns pressure, temperature, and activity data to behavioural events
- Detects respiratory cycles from pressure signals
- Identifies valid respiratory periods
- Computes respiratory timing, duty cycle, frequency, and drive
- Exports Excel summaries, interactive HTML plots, SVG figures, and breath‑by‑breath CSVs

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

## Outputs

Each analysis run creates a self‑contained output folder containing:

- Run configuration (JSON)
- Excel summaries
- Interactive HTML plots
- Static SVG figures
- CSV exports (where applicable)

---

## Notes

- `requirements.txt` contains runtime dependencies only
- The pipeline prioritises robustness and interpretability over speed
- Intended for research and analysis use, not real‑time or clinical deployment
