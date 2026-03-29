# Changelog

## [1.1.0] — 2026-03-30

### Added

- **Atmospheric pressure support**: recordings containing an APR/BARO column (1 Hz) are now parsed automatically. No configuration needed — the column is detected from the file metadata.
- **Atmospheric Pressure Excel sheet**: when APR/BARO data is present, a dedicated sheet is written to the output workbook with:
  - One row of overall session stats (mean, SD, min, max) across the full recording
  - Time-binned stats (mean, SD, min, max, N) at a configurable interval (default 5 minutes)
- **GUI: last directory memory**: the file browser reopens in the last folder used, persisted across sessions.
- **GUI: Reset Saved Settings**: moved from a button in the main window to a `···` dropdown menu at the top left — harder to hit accidentally.
- **Run config logs**: `analysis_config_*.json` files are now written to a `logs/` subfolder inside the analysis folder instead of cluttering the top level.

---

## [1.0.16] — 2025-07-13

The following features were in place at the last tagged release.

### Behaviour-aligned telemetry analysis

- Aligns pressure, temperature, and activity signals to behavioural event files
- Detects respiratory cycles, valid periods, and computes timing metrics (T_I, T_E, T_TOT, duty cycle, frequency, drive)
- Exports per-bin, per-period, per-window, and global summary sheets to Excel
- Exports interactive HTML plots, SVG figures, and breath-by-breath Ttot CSVs

### Injection-aligned photometry analysis

- Aligns photometry signals (dF/F) to an injection or event time
- Bins signals and peaks relative to injection
- Optionally integrates temperature and activity from telemetry implants
- Exports Excel summaries and plots
