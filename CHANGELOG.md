# Changelog

## Unreleased

### Alignment corrections

- Behaviour alignment now derives the telemetry offset directly from the video
  start. Video start is explicit time zero for both BORIS events and telemetry;
  the redundant probe-start input and its invalid range check have been removed.
- Behaviour windows are now classified as fully covered, partially covered,
  unavailable, or filtered. Only fully covered eligible windows are analysed;
  every decision and the continuous telemetry intervals are recorded in the run
  log/config. Pressure outages longer than one second are filtered independently,
  split coverage, and reserve a one-second filter-edge margin.
- Photometry mode now expresses telemetry timestamps relative to the entered
  photometry start time instead of incorrectly resetting telemetry to its first
  valid sample.
- Batch CSV files no longer require `probe_time`; existing files containing the
  column remain supported and the value is ignored.
- Batch analysis now supports native `.xlsx` configuration files, normalizes
  Excel/ISO timestamps with exact seconds, and accepts an optional shared output
  directory for all per-recording result folders.
- Batch runs reject rows that resolve to the same analysis folder and exit
  non-zero, preventing result folders from mixing or overwriting data.
- Excel export failures now propagate to the controller/batch runner instead of
  allowing a run with no workbook to be marked complete.
- Telemetry interpolation no longer invents endpoint values outside a channel's
  observed time range.

### Compatibility

- Internal controller/core Python call signatures now use one alignment timestamp.
  In-repository callers have been migrated; external Python integrations must
  update their calls.
- Pressure analysis-config JSON now declares schema version 3 and records window
  coverage; photometry configs use schema version 2. Both include clearer
  `VideoStartTime` or `PhotometryStartTime` fields. The legacy `VideoTime` and
  `AlignTime` aliases remain during transition.

---

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
