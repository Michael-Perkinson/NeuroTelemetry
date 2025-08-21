# main.py
from src.gui.main_gui import DataConfigGUI
from PySide6.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataConfigGUI()
    window.resize(640, 480)
    window.show()
    sys.exit(app.exec())
