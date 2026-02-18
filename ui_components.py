from __future__ import annotations

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget


class Card(QFrame):
    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._title = QLabel(title) if title else None
        if self._title:
            self._title.setObjectName("CardTitle")
        self._body = QWidget(self)
        self._body.setObjectName("CardBody")

    def setLayout(self, layout):
        self._body.setLayout(layout)
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(12, 12, 12, 12)
        wrapper.setSpacing(10)
        if self._title:
            wrapper.addWidget(self._title)
        wrapper.addWidget(self._body)
        super().setLayout(wrapper)

    def set_title(self, title: str):
        if self._title:
            self._title.setText(title)


class SidebarButton(QPushButton):
    def __init__(self, text: str, icon_path: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("SidebarButton")
        self.setIconSize(QSize(18, 18))
        if icon_path:
            self.setIcon(QIcon(icon_path))


class TopBarIconButton(QToolButton):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setText(text)
        self.setObjectName("TopBarIconButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
