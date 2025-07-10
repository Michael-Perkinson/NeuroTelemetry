import traceback
from PySide6.QtWidgets import QScrollArea, QWidget
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QDateTimeEdit, QCheckBox,
    QSpinBox, QPlainTextEdit, QMessageBox, QGridLayout
)
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import Qt, QDateTime
from pathlib import Path
import sys
from PySide6.QtWidgets import QSizePolicy

from src.controller import load_data
from src.controller import run_analysis_pipeline


class DataConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telemetry Behavior Config")
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-size: 10pt;
            }
        """)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Input files
        self.add_section_title("Input Files")
        self.data_path = self.create_file_selector("Telemetry Data File")
        self.behaviour_path = self.create_file_selector("Behavior Data File")

        # Timestamps
        self.add_section_title("Timestamp Alignment")
        self.probe_time = self.create_datetime_input("Probe Start Time")
        self.video_time = self.create_datetime_input("Video Start Time")

        # Behavior to plot
        self.add_section_title("Behavior of Interest")
        self.behavior_input = self.create_line_input(
            "Behavior to Plot", "Time spent sleeping")

        # --- Two-column grid section ---
        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(20)
        form_grid.setVerticalSpacing(12)
        form_grid.setContentsMargins(20, 5, 0, 0)
        row = 0

        # Buffer
        form_grid.addWidget(self.grid_section_title(
            "Buffer Around Behavior"), row, 0, 1, 4)
        row += 1
        self.buffer_before = self.create_spin_box(60)
        self.buffer_after = self.create_spin_box(60)
        form_grid.addWidget(QLabel("Buffer Before (s)"), row, 0)
        form_grid.addWidget(self.buffer_before, row, 1)
        form_grid.addWidget(QLabel("Buffer After (s)"), row, 2)
        form_grid.addWidget(self.buffer_after, row, 3)
        row += 1

        # Duration
        form_grid.addWidget(self.grid_section_title(
            "Duration Filtering"), row, 0, 1, 4)
        row += 1
        self.restrict_duration = QCheckBox("Restrict by Duration")
        self.restrict_duration.setChecked(True)
        form_grid.addWidget(self.restrict_duration, row, 0, 1, 2)
        row += 1
        self.min_duration = self.create_spin_box(30)
        self.max_duration = self.create_spin_box(10000)
        form_grid.addWidget(QLabel("Min Duration (s)"), row, 0)
        form_grid.addWidget(self.min_duration, row, 1)
        form_grid.addWidget(QLabel("Max Duration (s)"), row, 2)
        form_grid.addWidget(self.max_duration, row, 3)
        self.restrict_duration.toggled.connect(self.toggle_duration_inputs)
        self.toggle_duration_inputs(True)
        row += 1

        # Output
        form_grid.addWidget(self.grid_section_title(
            "Output Settings"), row, 0, 1, 4)
        row += 1
        self.bin_size = self.create_spin_box(10)
        form_grid.addWidget(QLabel("Bin Size (s)"), row, 0)
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
        lbl.setFixedWidth(180)
        entry = QLineEdit()
        entry.setReadOnly(True)
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
        lbl.setFixedWidth(180)
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
        lbl.setFixedWidth(180)
        dt = QDateTimeEdit()
        dt.setDisplayFormat("MM/dd/yyyy hh:mm:ss AP")
        dt.setDateTime(QDateTime.currentDateTime())
        dt.setCalendarPopup(True)
        layout.addWidget(lbl)
        layout.addWidget(dt)
        self.main_layout.addLayout(layout)
        return dt

    def create_spin_box(self, default):
        box = QSpinBox()
        box.setMaximum(100000000)
        box.setValue(default)
        box.setFixedWidth(100)
        return box

    def toggle_duration_inputs(self, enabled):
        self.min_duration.setEnabled(enabled)
        self.max_duration.setEnabled(enabled)

    def select_file(self, target_entry):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            filter="CSV and ASCII Files (*.csv *.ascii);;All Files (*)"
        )
        if path:
            target_entry.setText(path)

    def log(self, msg):
        self.log_box.appendPlainText(msg)

    def launch_analysis(self):
        try:
            telemetry_path = Path(self.data_path.text())
            behavior_path = Path(self.behaviour_path.text())

            if not telemetry_path.exists() or not behavior_path.exists():
                raise FileNotFoundError(
                    "One or both selected files do not exist.")

            self.log("Loading files...")
            telemetry_df, event_df = load_data(
                telemetry_path, behavior_path)

            if telemetry_df.empty or event_df.empty:
                raise ValueError("One or both files are empty.")

            self.log("Files loaded successfully.")
            self.log(f"Telemetry rows: {len(telemetry_df)}")
            self.log(f"Event rows: {len(event_df)}")

            # Extract values from GUI
            probe_time_str = self.probe_time.dateTime().toString("MM/dd/yyyy hh:mm:ss AP")
            video_time_str = self.video_time.dateTime().toString("MM/dd/yyyy hh:mm:ss AP")
            behaviour_to_plot = self.behavior_input.text()
            buffer_before = self.buffer_before.value()
            buffer_after = self.buffer_after.value()
            min_duration = self.min_duration.value() if self.restrict_duration.isChecked() else 0
            output_path = telemetry_path  # used for naming and output folders

            self.log("Running analysis...")
            run_analysis_pipeline(
                telemetry_df=telemetry_df,
                event_df=event_df,
                behaviour_to_plot=behaviour_to_plot,
                probe_time=probe_time_str,
                video_time=video_time_str,
                buffer_before=buffer_before,
                buffer_after=buffer_after,
                min_duration=min_duration,
                output_path=output_path,
                log_callback=self.log
            )

        except Exception as e:
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))
