"""Small reusable form widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class LabeledSlider(QWidget):
    """Horizontal slider over a float range with a live value label."""

    valueChanged = Signal(float)

    def __init__(self, lo: float, hi: float, value: float, step: float = 1.0, decimals: int = 0, parent=None):
        super().__init__(parent)
        self.lo, self.step, self.decimals = lo, step, decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((hi - lo) / step)))
        self.label = QLabel()
        self.label.setMinimumWidth(44)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._changed)
        self.setValue(value)

    def value(self) -> float:
        return self.lo + self.slider.value() * self.step

    def setValue(self, v: float) -> None:  # noqa: N802
        self.slider.setValue(int(round((v - self.lo) / self.step)))
        self._update_label()

    def _update_label(self) -> None:
        self.label.setText(f"{self.value():.{self.decimals}f}")

    def _changed(self, _):
        self._update_label()
        self.valueChanged.emit(self.value())
