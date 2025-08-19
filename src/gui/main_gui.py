# gui_app.py  (updated GUI with optional Photometry mode)

import traceback
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QDateTimeEdit,
    QSpinBox, QPlainTextEdit, QMessageBox, QGridLayout, QSizePolicy
)
from PySide6.QtCore import QDateTime
from pathlib import Path

from src.controllers.pressure_controller import load_data, run_pressure_pipeline
from src.controllers.photometry_controller import run_photometry_pipeline

class DataConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telemetry / Photometry Analysis")
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: white; font-size: 10pt; }
            QLineEdit[readOnly="true"] { background: #1e1e1e; }
        """)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Input files
        self.add_section_title("Input Files")
        self.data_path = self.create_file_selector("Telemetry Data File")
        self.behaviour_path = self.create_file_selector("Behavior Data File")

        # Photometry (optional)
        self.add_section_title("Photometry (optional)")
        self.photometry_path = self.create_file_selector(
            "Photometry Data File (optional)")

        ph_grid = QGridLayout()
        ph_grid.setHorizontalSpacing(20)
        ph_grid.setVerticalSpacing(12)
        ph_grid.setContentsMargins(20, 5, 0, 0)
        r = 0

        ph_grid.addWidget(self.grid_section_title(
            "Injection-centred analysis"), r, 0, 1, 4)
        r += 1

        ph_grid.addWidget(QLabel("Injection Time"), r, 0)
        self.injection_time = QDateTimeEdit()
        self.injection_time.setDisplayFormat("dd/MM/yyyy hh:mm:ss AP")
        self.injection_time.setDateTime(QDateTime.currentDateTime())
        self.injection_time.setCalendarPopup(True)
        ph_grid.addWidget(self.injection_time, r, 1, 1, 3)
        r += 1

        self.photo_pre_min = self.create_spin_box(10)
        self.photo_post_min = self.create_spin_box(60)
        self.photo_bin_min = self.create_spin_box(1)

        ph_grid.addWidget(QLabel("Pre (min)"), r, 0)
        ph_grid.addWidget(self.photo_pre_min, r, 1)
        ph_grid.addWidget(QLabel("Post (min)"), r, 2)
        ph_grid.addWidget(self.photo_post_min, r, 3)
        r += 1

        ph_grid.addWidget(QLabel("Bin (min)"), r, 0)
        ph_grid.addWidget(self.photo_bin_min, r, 1)
        r += 1

        self.main_layout.addLayout(ph_grid)

        # Timestamps (behaviour alignment mode)
        self.add_section_title("Timestamp Alignment (Behaviour mode)")
        self.probe_time = self.create_datetime_input("Probe Start Time")
        self.video_time = self.create_datetime_input("Video Start Time")

        # Behavior to plot
        self.add_section_title("Behavior of Interest")
        self.behavior_input = self.create_line_input(
            "Behavior to Plot", "Time spent sleeping")

        # Output
        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(20)
        form_grid.setVerticalSpacing(12)
        form_grid.setContentsMargins(20, 5, 0, 0)
        row = 0
        form_grid.addWidget(self.grid_section_title(
            "Output Settings"), row, 0, 1, 4)
        row += 1
        self.bin_size = self.create_spin_box(10)
        form_grid.addWidget(QLabel("Bin Size (s) [Behaviour mode]"), row, 0)
        form_grid.addWidget(self.bin_size, row, 1)
        self.main_layout.addLayout(form_grid)

        # Run button
        self.launch_button = QPushButton("Run Analysis")
        self.launch_button.clicked.connect(self.launch_analysis)
        self.main_layout.addWidget(self.launch_button)

        # Log
        log_label = QLabel("Log:")
        log_label.setContentsMargins(10, 10, 0, 0)
        self.main_layout.addWidget(log_label)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.main_layout.addWidget(self.log_box)

        # Enable/disable photometry fields based on file presence
        self.photometry_path.textChanged.connect(self.toggle_photometry_inputs)
        self.toggle_photometry_inputs()

    def add_section_title(self, text):
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 0, 5)
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)
        self.main_layout.addLayout(layout)

    def grid_section_title(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        return label

    def create_file_selector(self, label):
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(220)
        entry = QLineEdit()
        entry.setReadOnly(True)
        entry.setProperty("readOnly", True)
        btn = QPushButton("Browse")
        btn.setFixedWidth(80)
        entry.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Fixed)
        btn.clicked.connect(lambda: self.select_file(entry))
        layout.addWidget(lbl)
        layout.addWidget(entry)
        layout.addWidget(btn)
        self.main_layout.addLayout(layout)
        return entry

    def create_line_input(self, label, default=""):
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(220)
        entry = QLineEdit()
        entry.setText(default)
        layout.addWidget(lbl)
        layout.addWidget(entry)
        self.main_layout.addLayout(layout)
        return entry

    def create_datetime_input(self, label):
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(220)
        dt = QDateTimeEdit()
        dt.setDisplayFormat("dd/MM/yyyy hh:mm:ss AP")
        dt.setDateTime(QDateTime.currentDateTime())
        dt.setCalendarPopup(True)
        layout.addWidget(lbl)
        layout.addWidget(dt)
        self.main_layout.addLayout(layout)
        return dt

    def create_spin_box(self, default):
        box = QSpinBox()
        box.setMaximum(10**9)
        box.setValue(default)
        box.setFixedWidth(100)
        return box

    def select_file(self, target_entry):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            filter="CSV and ASCII Files (*.csv *.ascii *.txt);;All Files (*)"
        )
        if path:
            target_entry.setText(path)

    def toggle_photometry_inputs(self):
        enabled = bool(self.photometry_path.text().strip())
        for w in (self.injection_time, self.photo_pre_min, self.photo_post_min, self.photo_bin_min):
            w.setEnabled(enabled)

    def log(self, msg):
        self.log_box.appendPlainText(msg)

    def launch_analysis(self):
        try:
            telemetry_path = Path(self.data_path.text())
            if not telemetry_path.exists():
                raise FileNotFoundError("Telemetry file does not exist.")

            photometry_mode = bool(self.photometry_path.text().strip())
            if photometry_mode:
                # --- Photometry mode ---
                photometry_path = Path(self.photometry_path.text())
                if not photometry_path.exists():
                    raise FileNotFoundError(
                        "Photometry file was selected but does not exist.")

                self.log("Loading telemetry (for temp/activity/timebase)...")
                # Let the photometry controller read the photometry file itself
                # (we still pass telemetry-derived temp/activity if needed later)
                telemetry_df, _ = load_data(telemetry_path, telemetry_path) if False else (
                    None, None)  # not used here

                injection_time_str = self.injection_time.dateTime().toString("dd/MM/yyyy hh:mm:ss AP")
                pre_min = self.photo_pre_min.value()
                post_min = self.photo_post_min.value()
                bin_min = self.photo_bin_min.value()

                self.log("Running photometry analysis...")
                run_photometry_pipeline(
                    telemetry_path=telemetry_path,        # for naming/output folders
                    photometry_path=photometry_path,      # dF/F file
                    injection_time=injection_time_str,
                    pre_minutes=pre_min,
                    post_minutes=post_min,
                    bin_minutes=bin_min,
                    log_callback=self.log
                )
                return

            # --- Behaviour/pressure mode (original path) ---
            behavior_path = Path(self.behaviour_path.text())
            if not behavior_path.exists():
                raise FileNotFoundError("Behavior file does not exist.")

            self.log("Loading files...")
            telemetry_df, event_df = load_data(telemetry_path, behavior_path)
            if telemetry_df.empty or event_df.empty:
                raise ValueError("One or both files are empty.")

            self.log("Files loaded successfully.")
            self.log(f"Telemetry rows: {len(telemetry_df)}")
            self.log(f"Event rows: {len(event_df)}")

            probe_time_str = self.probe_time.dateTime().toString("dd/MM/yyyy hh:mm:ss AP")
            video_time_str = self.video_time.dateTime().toString("dd/MM/yyyy hh:mm:ss AP")
            behaviour_to_plot = self.behavior_input.text()
            bin_size = self.bin_size.value()
            output_path = telemetry_path  # used for naming and output folders

            self.log("Running analysis...")
            run_pressure_pipeline(
                telemetry_df=telemetry_df,
                event_df=event_df,
                behaviour_to_plot=behaviour_to_plot,
                probe_time=probe_time_str,
                video_time=video_time_str,
                bin_size_sec=bin_size,
                output_path=output_path,
                log_callback=self.log
            )

        except Exception as e:
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication([])
    w = DataConfigGUI()
    w.resize(900, 700)
    w.show()
    app.exec()
