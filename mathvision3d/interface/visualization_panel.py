"""Centre panel: interactive 3D viewport and its toolbar."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from ..i18n import tr
from ..rendering.interaction import ViewportInteraction
from ..rendering.scene_manager import SceneManager


class VisualizationPanel(QWidget):
    screenshot_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.btn_reset = QPushButton(tr("reset_camera"))
        bar.addWidget(self.btn_reset)
        self.view_buttons = {}
        for key, view in (("view_front", "front"), ("view_top", "top"), ("view_side", "side"), ("view_iso", "iso")):
            b = QPushButton(tr(key))
            b.clicked.connect(lambda _=False, v=view: self.scene.reset_camera(v))
            bar.addWidget(b)
            self.view_buttons[view] = b
        bar.addStretch(1)
        lay.addLayout(bar)
        # second row: toggles (two rows keep the viewport narrowable with long Georgian labels)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.btn_ortho = QPushButton(tr("orthographic"))
        self.btn_ortho.setCheckable(True)
        self.btn_grid = QPushButton(tr("grid"))
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(True)
        self.btn_axes = QPushButton(tr("axes"))
        self.btn_axes.setCheckable(True)
        self.btn_axes.setChecked(True)
        for b in (self.btn_ortho, self.btn_grid, self.btn_axes):
            bar.addWidget(b)
        bar.addSpacing(10)
        self.btn_mode_camera = QPushButton(tr("mode_camera"))
        self.btn_mode_object = QPushButton(tr("mode_object"))
        group = QButtonGroup(self)
        for b in (self.btn_mode_camera, self.btn_mode_object):
            b.setCheckable(True)
            group.addButton(b)
            bar.addWidget(b)
        self.btn_mode_camera.setChecked(True)
        self.btn_auto = QPushButton(tr("auto_rotate"))
        self.btn_auto.setCheckable(True)
        bar.addWidget(self.btn_auto)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.plotter = QtInteractor(self, auto_update=False)
        self.plotter.interactor.setMinimumWidth(320)
        lay.addWidget(self.plotter.interactor, 1)
        self.scene = SceneManager(self.plotter)
        self.interaction = ViewportInteraction(self.plotter, self.scene)

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(30)
        self._auto_timer.timeout.connect(self._auto_step)

        self.btn_reset.clicked.connect(lambda: self.scene.reset_camera("front"))
        self.btn_ortho.toggled.connect(self.scene.set_parallel_projection)
        self.btn_grid.toggled.connect(self.scene.set_grid_visible)
        self.btn_axes.toggled.connect(self.scene.set_axes_visible)
        self.btn_mode_camera.toggled.connect(lambda on: on and self.interaction.set_mode("camera"))
        self.btn_mode_object.toggled.connect(lambda on: on and self.interaction.set_mode("object"))
        self.btn_auto.toggled.connect(lambda on: self._auto_timer.start() if on else self._auto_timer.stop())

    def _auto_step(self) -> None:
        uid = self.scene.selected
        if uid is None:
            return
        self.scene.objects[uid].transform.rotate_world((0, 1, 0), 1.2)
        self.scene.update_transform(uid)
        self.interaction.transform_changed.emit(uid)

    def close_plotter(self) -> None:
        self._auto_timer.stop()
        try:
            self.plotter.close()
        except Exception:  # pragma: no cover
            pass
