# gui_app.py

import traceback
import pandas as pd
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QDateTimeEdit,
    QSpinBox, QPlainTextEdit, QMessageBox, QGridLayout, QSizePolicy,
    QStackedWidget
)
from PySide6.QtCore import QDateTime, Qt

from src.controllers.pressure_controller import load_data, run_pressure_pipeline
from src.controllers.photometry_controller import run_photometry_pipeline
from src.core.file_handling import detect_file_type


class DataConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telemetry Alignment Analysis")
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: white; font-size: 10pt; }
            QLineEdit[readOnly="true"] { background: #1e1e1e; }
        """)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Input files
        self.add_global_section_title("Input Files")
        self.telemetry_path = self.create_file_selector("Telemetry Data File")
        self.secondary_path = self.create_file_selector(
            "Behaviour or Photometry File"
        )

        # --- Option panels stacked ---
        self.stack = QStackedWidget()
        self.behaviour_panel = self.build_behaviour_panel()
        self.photometry_panel = self.build_photometry_panel()
        self.stack.addWidget(self.behaviour_panel)   # index 0
        self.stack.addWidget(self.photometry_panel)  # index 1
        self.main_layout.addWidget(self.stack)

        # Start hidden
        self.stack.setVisible(False)
        self.file_mode = "unknown"

        # Run button
        self.launch_button = QPushButton("Run Analysis")
        self.launch_button.clicked.connect(self.launch_analysis)
        self.main_layout.addWidget(self.launch_button)

        # Log
        self.add_global_section_title("Log")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.main_layout.addWidget(self.log_box)

        # React to file selection / typing
        self.secondary_path.textChanged.connect(self.on_secondary_file_changed)

    # -------------------- UI Builders --------------------

    def add_global_section_title(self, text):
        """Adds a title into the main layout (global section)."""
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 0, 5)
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)
        self.main_layout.addLayout(layout)

    def make_section_title(self, text):
        """Returns a QLabel for titles inside panels."""
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
        return layout, entry

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
        return layout, dt

    def create_spin_box(self, default):
        box = QSpinBox()
        box.setMaximum(10**9)
        box.setValue(default)
        box.setFixedWidth(100)
        return box

    def build_behaviour_panel(self):
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setHorizontalSpacing(15)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(10, 5, 10, 5)

        # Centered title
        title = QLabel("Behaviour Mode")
        title.setStyleSheet("font-weight: bold; font-size: 18pt;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(title, 0, 0, 1, 2)  # row 0, col 0, span across 2 columns

        # Probe + video start
        probe_lbl = QLabel("Probe Start Time")
        self.probe_time = QDateTimeEdit()
        self.probe_time.setDisplayFormat("dd/MM/yyyy hh:mm:ss AP")
        self.probe_time.setDateTime(QDateTime.currentDateTime())
        self.probe_time.setCalendarPopup(True)
        grid.addWidget(probe_lbl, 1, 0)
        grid.addWidget(self.probe_time, 1, 1)

        video_lbl = QLabel("Video Start Time")
        self.video_time = QDateTimeEdit()
        self.video_time.setDisplayFormat("dd/MM/yyyy hh:mm:ss AP")
        self.video_time.setDateTime(QDateTime.currentDateTime())
        self.video_time.setCalendarPopup(True)
        grid.addWidget(video_lbl, 2, 0)
        grid.addWidget(self.video_time, 2, 1)

        # Behaviour input
        beh_lbl = QLabel("Behaviour to Plot")
        self.behaviour_input = QLineEdit("Time spent sleeping")
        grid.addWidget(beh_lbl, 3, 0)
        grid.addWidget(self.behaviour_input, 3, 1)

        # Bin size
        bin_lbl = QLabel("Bin Size (s)")
        self.bin_size = self.create_spin_box(10)
        grid.addWidget(bin_lbl, 4, 0)
        grid.addWidget(self.bin_size, 4, 1)

        return panel

    def build_photometry_panel(self):
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)
        grid.setContentsMargins(10, 5, 10, 5)

        # Centered title
        title = QLabel("Photometry Mode")
        title.setStyleSheet("font-weight: bold; font-size: 18pt;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(title, 0, 0, 1, 4)  # span across all 4 columns

        # Photometry start time
        start_lbl = QLabel("Photometry Start Time")
        self.photometry_start_time = QDateTimeEdit()
        self.photometry_start_time.setDisplayFormat("dd/MM/yyyy hh:mm:ss AP")
        self.photometry_start_time.setDateTime(QDateTime.currentDateTime())
        self.photometry_start_time.setCalendarPopup(True)
        grid.addWidget(start_lbl, 1, 0)
        grid.addWidget(self.photometry_start_time, 1, 1, 1, 3)

        # Injection + Pre/Post + Bin
        grid.addWidget(QLabel("Injection Time (s from start)"), 2, 0)
        self.injection_sec = self.create_spin_box(0)
        grid.addWidget(self.injection_sec, 2, 1)

        grid.addWidget(QLabel("Pre (min)"), 2, 2)
        self.photo_pre_min = self.create_spin_box(10)
        grid.addWidget(self.photo_pre_min, 2, 3)

        grid.addWidget(QLabel("Post (min)"), 3, 0)
        self.photo_post_min = self.create_spin_box(60)
        grid.addWidget(self.photo_post_min, 3, 1)

        grid.addWidget(QLabel("Bin (min)"), 3, 2)
        self.photo_bin_min = self.create_spin_box(1)
        grid.addWidget(self.photo_bin_min, 3, 3)

        return panel

    # -------------------- Helpers --------------------

    def select_file(self, target_entry):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            filter="CSV and ASCII Files (*.csv *.ascii *.txt);;All Files (*)"
        )
        if path:
            target_entry.setText(path)
            if target_entry is self.secondary_path:
                self.on_secondary_file_changed()

    def on_secondary_file_changed(self):
        path = Path(self.secondary_path.text())
        if not path.exists():
            self.stack.setVisible(False)
            self.file_mode = "unknown"
            return

        ftype = detect_file_type(path)
        if ftype == "behaviour":
            self.stack.setCurrentWidget(self.behaviour_panel)
            self.stack.setVisible(True)
            self.file_mode = "behaviour"
        elif ftype == "photometry":
            self.stack.setCurrentWidget(self.photometry_panel)
            self.stack.setVisible(True)
            self.file_mode = "photometry"
        else:
            self.stack.setVisible(False)
            self.file_mode = "unknown"

    def log(self, msg):
        self.log_box.appendPlainText(msg)

    # -------------------- Analysis --------------------

    def launch_analysis(self):
        try:
            telemetry_path = Path(self.telemetry_path.text())
            if not telemetry_path.exists():
                raise FileNotFoundError("Telemetry file does not exist.")

            second_path = Path(self.secondary_path.text())
            if not second_path.exists():
                raise FileNotFoundError("Second file does not exist.")

            if self.file_mode == "photometry":
                self.log("Loading telemetry and photometry...")
                telemetry_df = pd.read_csv(telemetry_path)
                photometry_df = pd.read_csv(second_path)

                self.log("Running photometry analysis...")
                run_photometry_pipeline(
                    telemetry_df=telemetry_df,
                    photometry_df=photometry_df,
                    injection_sec=self.injection_sec.value(),
                    pre_minutes=self.photo_pre_min.value(),
                    post_minutes=self.photo_post_min.value(),
                    bin_minutes=self.photo_bin_min.value(),
                    output_path=telemetry_path,
                    log_callback=self.log,
                )

            elif self.file_mode == "behaviour":
                self.log("Loading telemetry and behaviour data...")
                telemetry_df, behaviour_df = load_data(
                    telemetry_path, second_path)

                self.log("Running behaviour analysis...")
                run_pressure_pipeline(
                    telemetry_df=telemetry_df,
                    event_df=behaviour_df,
                    behaviour_to_plot=self.behaviour_input.text(),
                    probe_time=self.probe_time.dateTime().toString("dd/MM/yyyy hh:mm:ss AP"),
                    video_time=self.video_time.dateTime().toString("dd/MM/yyyy hh:mm:ss AP"),
                    bin_size_sec=self.bin_size.value(),
                    output_path=telemetry_path,
                    log_callback=self.log
                )

            else:
                raise ValueError(
                    "Unknown file type. Please select a valid Behaviour or Photometry file.")

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
