"""
Controls – reusable toolbar and in-game control widgets.
These are imported by main_window.py and board_view.py as needed.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from utils.config import (
    COLOR_UI_BG, COLOR_UI_PANEL, COLOR_UI_TEXT,
    COLOR_UI_ACCENT, COLOR_BUTTON, COLOR_BUTTON_HOVER,
)


BTN_BASE = f"""
    QPushButton {{
        background: {COLOR_BUTTON};
        color: {COLOR_UI_TEXT};
        border: none;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 11px;
        font-weight: bold;
    }}
    QPushButton:hover {{ background: {COLOR_BUTTON_HOVER}; }}
    QPushButton:pressed {{ background: {COLOR_UI_ACCENT}; }}
    QPushButton:disabled {{ background: #333344; color: #555566; }}
"""


def styled_button(text: str, bg: str = COLOR_BUTTON,
                  width: int = 0) -> QPushButton:
    btn = QPushButton(text)
    style = BTN_BASE.replace(COLOR_BUTTON, bg, 1)
    btn.setStyleSheet(style)
    if width:
        btn.setFixedWidth(width)
    return btn


def section_label(text: str, size: int = 10,
                  color: str = COLOR_UI_ACCENT) -> QLabel:
    lbl = QLabel(text)
    font = QFont("Arial", size, QFont.Weight.Bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl


def divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {COLOR_UI_ACCENT}; border: none;")
    return f


class SpeedControl(QWidget):
    """
    Horizontal slider that maps 0-4 to human-readable speed labels.
    Emits ``speed_changed(delay_ms)`` when the slider moves.
    """

    DELAYS = [2000, 1000, 600, 200, 50]
    LABELS = ["Very Slow", "Slow", "Normal", "Fast", "Instant"]

    speed_changed = pyqtSignal(int)   # delay in ms

    def __init__(self, default_level: int = 2, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel("Speed:")
        lbl.setStyleSheet(f"color: {COLOR_UI_TEXT}; font-size: 10px;")
        layout.addWidget(lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, len(self.DELAYS) - 1)
        self._slider.setValue(default_level)
        self._slider.setFixedWidth(110)
        self._slider.valueChanged.connect(self._on_change)
        layout.addWidget(self._slider)

        self._val_label = QLabel(self.LABELS[default_level])
        self._val_label.setStyleSheet(f"color: {COLOR_UI_ACCENT}; font-size: 10px; min-width: 55px;")
        layout.addWidget(self._val_label)

    def _on_change(self, level: int):
        self._val_label.setText(self.LABELS[level])
        self.speed_changed.emit(self.DELAYS[level])

    @property
    def current_delay(self) -> int:
        return self.DELAYS[self._slider.value()]


class AgentSelector(QWidget):
    """Labelled combo-box for selecting an agent type."""

    def __init__(self, label: str = "Agent:", default: str = "Heuristic",
                 parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {COLOR_UI_TEXT}; font-size: 10px;")
        layout.addWidget(lbl)

        self._combo = QComboBox()
        self._combo.addItems(["Heuristic", "Random", "RL (DQN)"])
        idx = self._combo.findText(default)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        self._combo.setStyleSheet(
            f"background: #333344; color: {COLOR_UI_TEXT}; "
            "padding: 3px; border-radius: 4px; font-size: 10px;"
        )
        layout.addWidget(self._combo)

    @property
    def selected(self) -> str:
        return self._combo.currentText()
