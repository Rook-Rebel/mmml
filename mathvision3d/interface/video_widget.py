"""Video preview with QPainter overlays (QPainter renders Georgian text; cv2.putText cannot)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..i18n import tr

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


@dataclass
class Overlay:
    obj_id: int
    points: np.ndarray       # image coordinates
    label: str
    color: str
    selected: bool = False
    closed: bool = True


def to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        h, w = img.shape
        return QImage(np.ascontiguousarray(img).data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = img.shape
    return QImage(np.ascontiguousarray(img).data, w, h, 3 * w, QImage.Format_BGR888).copy()


class VideoWidget(QWidget):
    clicked = Signal(float, float)            # image coordinates
    roi_selected = Signal(tuple)              # (x, y, w, h) image coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._img_size = (1, 1)
        self.overlays: list[Overlay] = []
        self.polylines: list[tuple[np.ndarray, str, bool]] = []   # (pts, color, dashed)
        self.hands: list[np.ndarray] = []
        self.roi: tuple | None = None
        self.roi_mode = False
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None
        self.banner = ""

    # ------------------------------------------------------------------ data
    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            self._image = None
        else:
            self._image = to_qimage(img)
            self._img_size = (img.shape[1], img.shape[0])
        self.update()

    def has_image(self) -> bool:
        return self._image is not None

    # -------------------------------------------------------------- geometry
    def _target_rect(self) -> QRectF:
        w, h = self._img_size
        s = min(self.width() / w, self.height() / h)
        tw, th = w * s, h * s
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def _to_widget(self, x: float, y: float) -> QPointF:
        r = self._target_rect()
        return QPointF(r.x() + x / self._img_size[0] * r.width(), r.y() + y / self._img_size[1] * r.height())

    def _to_image(self, p: QPointF) -> tuple[float, float]:
        r = self._target_rect()
        return ((p.x() - r.x()) / r.width() * self._img_size[0], (p.y() - r.y()) / r.height() * self._img_size[1])

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#06090d"))
        if self._image is None:
            p.setPen(QColor("#8b98a9"))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, tr("no_video"))
            return
        p.drawImage(self._target_rect(), self._image)
        for pts, color, dashed in self.polylines:
            pen = QPen(QColor(color), 1.6, Qt.DashLine if dashed else Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF([self._to_widget(*q) for q in pts]))
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        for ov in self.overlays:
            if len(ov.points) < 2:
                continue
            step = max(1, len(ov.points) // 400)
            poly = QPolygonF([self._to_widget(*q) for q in ov.points[::step]])
            color = QColor("#ffd54f") if ov.selected else QColor(ov.color)
            p.setPen(QPen(color, 3.2 if ov.selected else 2.0))
            fill = QColor(color)
            fill.setAlpha(45 if ov.selected else 18)
            p.setBrush(QBrush(fill) if ov.closed else Qt.NoBrush)
            if ov.closed:
                p.drawPolygon(poly)
            else:
                p.drawPolyline(poly)
            br = poly.boundingRect()
            text_rect = QRectF(br.x(), br.y() - 20, max(br.width(), 140), 18)
            bg = QColor("#0d1117")
            bg.setAlpha(190)
            p.setPen(Qt.NoPen)
            p.setBrush(bg)
            metrics = p.fontMetrics()
            p.drawRoundedRect(QRectF(text_rect.x(), text_rect.y(), metrics.horizontalAdvance(ov.label) + 10, 18), 4, 4)
            p.setPen(color)
            p.drawText(text_rect.adjusted(5, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, ov.label)
        self._paint_hands(p)
        if self.roi is not None:
            x, y, w, h = self.roi
            p.setPen(QPen(QColor("#e040fb"), 1.5, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(self._to_widget(x, y), self._to_widget(x + w, y + h)))
        if self._drag_start is not None and self._drag_now is not None:
            p.setPen(QPen(QColor("#e040fb"), 1.5))
            p.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        if self.banner:
            p.setPen(QColor("#ffb74d"))
            p.drawText(self.rect().adjusted(8, 6, -8, -6), Qt.AlignTop | Qt.AlignRight, self.banner)

    def _paint_hands(self, p: QPainter) -> None:
        w, h = self._img_size
        for lm in self.hands:
            pts = [self._to_widget(x * w, y * h) for x, y, _ in lm]
            p.setPen(QPen(QColor("#69f0ae"), 2))
            for a, b in HAND_CONNECTIONS:
                p.drawLine(pts[a], pts[b])
            p.setBrush(QColor("#69f0ae"))
            for q in pts:
                p.drawEllipse(q, 2.5, 2.5)

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, e) -> None:  # noqa: N802
        if self._image is None or e.button() != Qt.LeftButton:
            return
        if self.roi_mode:
            self._drag_start = self._drag_now = e.position()
        else:
            self.clicked.emit(*self._to_image(e.position()))

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_now = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._drag_start is None:
            return
        a, b = self._to_image(self._drag_start), self._to_image(e.position())
        self._drag_start = self._drag_now = None
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(self._img_size[0]), x1), min(float(self._img_size[1]), y1)
        if x1 - x0 > 10 and y1 - y0 > 10:
            self.roi_selected.emit((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
        self.update()
