# main.py
import sys

from PySide6.QtWidgets import QApplication

from src.gui.analysis_window import AnalysisWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalysisWindow()
    window.setWindowTitle("Telemetry Alignment Analysis Tool")
    window.resize(640, 480)
    window.show()
    sys.exit(app.exec())
