from PyQt6.QtWidgets import QMessageBox, QWidget


def show_friendly_error(
    parent: QWidget,
    title: str,
    friendly: str,
    details: str,
    *,
    icon: QMessageBox.Icon = QMessageBox.Icon.Critical,
):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(friendly)
    if details:
        box.setInformativeText("Show technical details below if you need them.")
        box.setDetailedText(details)
    box.setIcon(icon)
    box.exec()
