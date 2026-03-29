# gui_app.py

import traceback
from pathlib import Path
from datetime import datetime


from PySide6.QtCore import QDateTime, QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDateTimeEdit,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.controllers.photometry_controller import (
    load_photometry_data,
    run_photometry_pipeline,
)
from src.controllers.pressure_controller import load_data, run_pressure_pipeline
from src.core.file_handling import detect_file_type


class DataConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: white; font-size: 10pt; }
            QLineEdit[readOnly="true"] { background: #1e1e1e; }
        """)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Input Files title row with discreet settings menu tucked on the right
        self._settings_menu = QMenu(self)
        self._settings_menu.addAction("Reset Saved Settings", self.clear_settings)
        menu_btn = QPushButton("···")
        menu_btn.setFixedSize(20, 14)
        menu_btn.setStyleSheet(
            "color: #999; font-size: 9pt; border: none; background: transparent;"
        )
        menu_btn.setMenu(self._settings_menu)
        input_header = QHBoxLayout()
        input_header.setContentsMargins(20, 15, 4, 5)
        input_title = QLabel("Input Files")
        input_title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        input_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_header.addWidget(menu_btn)
        input_header.addWidget(input_title)
        self.main_layout.addLayout(input_header)
        self.telemetry_path = self.create_file_selector("Telemetry Data File")
        self.secondary_path = self.create_file_selector(
            "Behaviour or Photometry File")

        # --- Option panels stacked ---
        self.stack = QStackedWidget()
        self.behaviour_panel = self.build_behaviour_panel()
        self.photometry_panel = self.build_photometry_panel()
        self.stack.addWidget(self.behaviour_panel)  # index 0
        self.stack.addWidget(self.photometry_panel)  # index 1
        self.main_layout.addWidget(self.stack)

        # Start hidden
        self.stack.setVisible(False)
        self.file_mode = "unknown"

        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.main_layout.addWidget(self.run_button)
        self.run_button.setEnabled(False)

        # Log
        self.add_global_section_title("Log")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.main_layout.addWidget(self.log_box)

        # React to file selection / typing
        self.secondary_path.textChanged.connect(self.on_secondary_file_changed)

        self.restore_settings()

    # -------------------- UI Builders --------------------

    def add_global_section_title(self, text):
        """Adds a title into the main layout (global section)."""
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 0, 5)
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        dt = self.make_datetime_edit()
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
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # row 0, col 0, span across 2 columns
        grid.addWidget(title, 0, 0, 1, 2)

        # Probe + video start
        probe_lbl = QLabel("Probe Start Time")
        self.probe_time = self.make_datetime_edit()
        grid.addWidget(probe_lbl, 1, 0)
        grid.addWidget(self.probe_time, 1, 1)

        video_lbl = QLabel("Video Start Time")
        self.video_time = self.make_datetime_edit()
        grid.addWidget(video_lbl, 2, 0)
        grid.addWidget(self.video_time, 2, 1)

        # Behaviour input
        beh_lbl = QLabel("Behaviour to Plot")
        self.behaviour_input = QLineEdit("Time spent sleeping")
        grid.addWidget(beh_lbl, 3, 0)
        grid.addWidget(self.behaviour_input, 3, 1)

        # Respiratory bin size
        bin_lbl = QLabel("Respiratory Bin Size (s)")
        self.bin_size = self.create_spin_box(10)
        grid.addWidget(bin_lbl, 4, 0)
        grid.addWidget(self.bin_size, 4, 1)

        # Atm. pressure summary export toggle (appears before its bin size)
        self.export_atm_summary = QCheckBox("Export Atmospheric Pressure Session Summary")
        self.export_atm_summary.setChecked(True)
        grid.addWidget(self.export_atm_summary, 5, 0, 1, 2)

        # Atm. pressure bin size (hidden when checkbox is unchecked)
        self.atm_bin_lbl = QLabel("Atm. Pressure Bin Size (s)")
        self.atm_bin_size = self.create_spin_box(300)
        grid.addWidget(self.atm_bin_lbl, 6, 0)
        grid.addWidget(self.atm_bin_size, 6, 1)

        self.export_atm_summary.toggled.connect(self._toggle_atm_bin_visibility)

        return panel

    def build_photometry_panel(self):
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)
        grid.setContentsMargins(10, 5, 10, 5)

        # Centered title
        title = QLabel("Photometry Mode")
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(title, 0, 0, 1, 4)  # span across all 4 columns

        # Photometry start time (keep as datetime)
        start_lbl = QLabel("Photometry Start Time")
        self.photometry_start_time = self.make_datetime_edit()
        grid.addWidget(start_lbl, 1, 0)
        grid.addWidget(self.photometry_start_time, 1, 1, 1, 3)

        # Injection + Pre/Post + Bin (line edits instead of spinboxes)
        grid.addWidget(QLabel("Injection Time (s from start)"), 2, 0)
        self.injection_sec = QSpinBox()
        self.injection_sec.setRange(0, 99999)  # adjust max as needed
        self.injection_sec.setValue(0)
        grid.addWidget(self.injection_sec, 2, 1)

        grid.addWidget(QLabel("Pre (min)"), 2, 2)
        self.photo_pre_min = QSpinBox()
        self.photo_pre_min.setRange(0, 1000)
        self.photo_pre_min.setValue(30)
        grid.addWidget(self.photo_pre_min, 2, 3)

        grid.addWidget(QLabel("Post (min)"), 3, 0)
        self.photo_post_min = QSpinBox()
        self.photo_post_min.setRange(0, 10000)
        self.photo_post_min.setValue(360)
        grid.addWidget(self.photo_post_min, 3, 1)

        grid.addWidget(QLabel("Bin (min)"), 3, 2)
        self.photo_bin_min = QSpinBox()
        self.photo_bin_min.setRange(1, 1000)  # bins shouldn’t be 0
        self.photo_bin_min.setValue(30)
        grid.addWidget(self.photo_bin_min, 3, 3)

        return panel

    # -------------------- Helpers --------------------

    def _toggle_atm_bin_visibility(self, checked: bool):
        self.atm_bin_lbl.setVisible(checked)
        self.atm_bin_size.setVisible(checked)

    def clear_settings(self):
        QSettings("NeuroTelemetry", "DataConfigGUI").clear()
        self.log("Saved settings cleared.")

    def save_settings(self):
        s = QSettings("NeuroTelemetry", "DataConfigGUI")
        s.setValue("behaviour_input", self.behaviour_input.text())
        s.setValue("bin_size", self.bin_size.value())
        s.setValue("atm_bin_size", self.atm_bin_size.value())
        s.setValue("export_atm_summary", self.export_atm_summary.isChecked())
        s.setValue("injection_sec", self.injection_sec.value())
        s.setValue("photo_pre_min", self.photo_pre_min.value())
        s.setValue("photo_post_min", self.photo_post_min.value())
        s.setValue("photo_bin_min", self.photo_bin_min.value())

    def restore_settings(self):
        s = QSettings("NeuroTelemetry", "DataConfigGUI")
        self.behaviour_input.setText(s.value("behaviour_input", "Time spent sleeping"))
        self.bin_size.setValue(int(s.value("bin_size", 10)))
        self.atm_bin_size.setValue(int(s.value("atm_bin_size", 300)))
        checked = s.value("export_atm_summary", True, type=bool)
        self.export_atm_summary.setChecked(checked)
        self.injection_sec.setValue(int(s.value("injection_sec", 0)))
        self.photo_pre_min.setValue(int(s.value("photo_pre_min", 30)))
        self.photo_post_min.setValue(int(s.value("photo_post_min", 360)))
        self.photo_bin_min.setValue(int(s.value("photo_bin_min", 30)))

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def make_datetime_edit(self):
        dt = QDateTimeEdit()
        dt.setDisplayFormat("dd/MM/yyyy hh:mm:ss AP")
        dt.setDateTime(QDateTime.currentDateTime())
        dt.setCalendarPopup(True)
        return dt

    def select_file(self, target_entry):
        s = QSettings("NeuroTelemetry", "DataConfigGUI")
        last_dir = s.value("last_directory", "")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            dir=last_dir,
            filter="CSV and ASCII Files (*.csv *.ascii *.txt);;All Files (*)",
        )
        if path:
            s.setValue("last_directory", str(Path(path).parent))
            target_entry.setText(path)
            if target_entry is self.secondary_path:
                self.on_secondary_file_changed()
        else:
            target_entry.clear()

    def on_secondary_file_changed(self):
        path = Path(self.secondary_path.text())
        if not path.exists():
            self.stack.setVisible(False)
            self.file_mode = "unknown"
            return

        ftype = detect_file_type(path)
        mapping = {
            "behaviour": self.behaviour_panel,
            "photometry": self.photometry_panel,
        }
        panel = mapping.get(ftype)
        if panel:
            self.stack.setCurrentWidget(panel)
            self.stack.setVisible(True)
            self.file_mode = ftype
            self.run_button.setEnabled(True)

        else:
            self.stack.setVisible(False)
            self.file_mode = "unknown"
            QMessageBox.warning(
                self,
                "Unrecognized File",
                "Could not detect file type.\nPlease choose a valid Behaviour or Photometry file.",
            )

    def log(self, msg):
        ts = datetime.now().strftime("[%H:%M:%S]")
        self.log_box.appendPlainText(f"{ts} {msg}")

    # -------------------- Analysis --------------------

    def run_analysis(self):
        self.save_settings()
        self.run_button.setEnabled(False)
        try:
            telemetry_path = Path(self.telemetry_path.text())
            if not telemetry_path.exists():
                raise FileNotFoundError("Telemetry file does not exist.")

            second_path = Path(self.secondary_path.text())
            if not second_path.exists():
                raise FileNotFoundError("Second file does not exist.")

            if self.file_mode == "photometry":
                self.log("Loading telemetry and photometry...")

                photometry_df, telemetry_df = load_photometry_data(
                    second_path, telemetry_path
                )

                self.log("Running photometry analysis...")
                run_photometry_pipeline(
                    telemetry_df=telemetry_df,
                    photometry_df=photometry_df,
                    photometry_align_time=self.photometry_start_time.dateTime().toString(
                        "dd/MM/yyyy hh:mm:ss AP"
                    ),
                    injection_sec=self.injection_sec.value(),
                    pre_min=self.photo_pre_min.value(),
                    post_min=self.photo_post_min.value(),
                    bin_min=self.photo_bin_min.value(),
                    telemetry_path=telemetry_path,
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
                    probe_time=self.probe_time.dateTime().toString(
                        "dd/MM/yyyy hh:mm:ss AP"
                    ),
                    video_time=self.video_time.dateTime().toString(
                        "dd/MM/yyyy hh:mm:ss AP"
                    ),
                    bin_size_sec=self.bin_size.value(),
                    output_path=telemetry_path,
                    atm_bin_size_sec=self.atm_bin_size.value(),
                    export_atm_summary=self.export_atm_summary.isChecked(),
                    log_callback=self.log,
                )

            else:
                raise ValueError(
                    "Unknown file type. Must be Behaviour or Photometry.")

        except Exception as e:
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))

        finally:
            self.run_button.setEnabled(True)
            self.log("Analysis complete.")
            self.log("=" * 40)
