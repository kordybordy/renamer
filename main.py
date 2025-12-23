import os
import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from config import BASE_DIR, GLOBAL_STYLESHEET
from gui import RenamerGUI


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Renamer")
    app.setApplicationDisplayName("Renamer")
    app.setStyleSheet(GLOBAL_STYLESHEET)

    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
    icon_path = os.path.join(BASE_DIR, "assets", "logo.ico")
    icon_file = icon_path if os.path.exists(icon_path) else logo_path
    if os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))

    gui = RenamerGUI()
    if icon_file and os.path.exists(icon_file):
        gui.setWindowIcon(QIcon(icon_file))
    gui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
