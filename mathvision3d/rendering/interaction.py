"""Mouse interaction in the 3D viewport.

Camera mode  : VTK trackball camera (orbit / pan / zoom); a click selects an object.
Object mode  : left-drag rotates the selected object, shift+left or middle-drag
               translates it in the screen plane, right-drag rolls it, wheel scales it.

The same screen-space operations are reused by the hand-gesture controller.
"""

from __future__ import annotations

import math

import numpy as np
import vtk
from PySide6.QtCore import QObject, Signal

from .scene_manager import SceneManager

ROTATE_DEG_PER_PX = 0.45
CLICK_TOLERANCE_PX = 4


class ViewportInteraction(QObject):
    object_picked = Signal(object)        # uid or None
    transform_changed = Signal(str)       # live updates while dragging
    transform_finished = Signal(str)

    def __init__(self, plotter, scene: SceneManager):
        super().__init__()
        self.plotter = plotter
        self.scene = scene
        self.mode = "camera"
        self._drag: str | None = None      # rotate | translate | roll
        self._press_pos = (0, 0)
        self._last_pos = (0, 0)
        self._moved = False
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        for event, handler in (
            ("LeftButtonPressEvent", self._left_down), ("LeftButtonReleaseEvent", self._left_up),
            ("MiddleButtonPressEvent", self._middle_down), ("MiddleButtonReleaseEvent", self._middle_up),
            ("RightButtonPressEvent", self._right_down), ("RightButtonReleaseEvent", self._right_up),
            ("MouseMoveEvent", self._move), ("MouseWheelForwardEvent", self._wheel_fwd),
            ("MouseWheelBackwardEvent", self._wheel_back), ("CharEvent", self._char),
        ):
            self.style.AddObserver(event, handler)
        plotter.iren.style = self.style

    # ------------------------------------------------------------- helpers
    @property
    def _iren(self):
        return self.style.GetInteractor()

    def _pos(self) -> tuple[int, int]:
        return tuple(self._iren.GetEventPosition())

    def _object_active(self) -> bool:
        return self.mode == "object" and self.scene.selected is not None

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def camera_basis(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(right, up, world_units_per_pixel) of the current camera."""
        cam = self.plotter.renderer.GetActiveCamera()
        up = np.array(cam.GetViewUp(), float)
        dop = np.array(cam.GetDirectionOfProjection(), float)
        right = np.cross(dop, up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, dop)
        up /= np.linalg.norm(up) + 1e-12
        h = max(self.plotter.renderer.GetSize()[1], 1)
        if cam.GetParallelProjection():
            wpp = 2.0 * cam.GetParallelScale() / h
        else:
            wpp = 2.0 * cam.GetDistance() * math.tan(math.radians(cam.GetViewAngle()) / 2.0) / h
        return right, up, wpp

    # ------------------------------------------- screen-space object operations
    def rotate_selected(self, dx_px: float, dy_px: float, deg_per_px: float = ROTATE_DEG_PER_PX) -> None:
        uid = self.scene.selected
        if uid is None:
            return
        right, up, _ = self.camera_basis()
        t = self.scene.objects[uid].transform
        t.rotate_world(up, dx_px * deg_per_px)
        t.rotate_world(right, -dy_px * deg_per_px)
        self.scene.update_transform(uid)
        self.transform_changed.emit(uid)

    def roll_selected(self, deg: float) -> None:
        uid = self.scene.selected
        if uid is None:
            return
        cam = self.plotter.renderer.GetActiveCamera()
        self.scene.objects[uid].transform.rotate_world(cam.GetDirectionOfProjection(), deg)
        self.scene.update_transform(uid)
        self.transform_changed.emit(uid)

    def translate_selected(self, dx_px: float, dy_px: float) -> None:
        uid = self.scene.selected
        if uid is None:
            return
        right, up, wpp = self.camera_basis()
        self.scene.objects[uid].transform.translate((right * dx_px + up * dy_px) * wpp)
        self.scene.update_transform(uid)
        self.transform_changed.emit(uid)

    def scale_selected(self, factor: float) -> None:
        uid = self.scene.selected
        if uid is None:
            return
        self.scene.objects[uid].transform.scale_by(factor)
        self.scene.update_transform(uid)
        self.transform_changed.emit(uid)

    # ---------------------------------------------------------------- events
    def _begin(self, kind: str) -> None:
        self._drag = kind
        self._press_pos = self._last_pos = self._pos()
        self._moved = False

    def _end(self) -> None:
        if self._drag and self.scene.selected:
            self.transform_finished.emit(self.scene.selected)
        self._drag = None

    def _left_down(self, obj, event):
        self._press_pos = self._last_pos = self._pos()
        self._moved = False
        if self._object_active():
            self._begin("translate" if self._iren.GetShiftKey() else "rotate")
        else:
            self.style.OnLeftButtonDown()

    def _left_up(self, obj, event):
        pos = self._pos()
        was_click = (abs(pos[0] - self._press_pos[0]) <= CLICK_TOLERANCE_PX
                     and abs(pos[1] - self._press_pos[1]) <= CLICK_TOLERANCE_PX)
        if self._drag:
            self._end()
        else:
            self.style.OnLeftButtonUp()
        if was_click:
            uid = self.scene.pick(*pos)
            if uid is not None or self.mode == "camera":
                self.object_picked.emit(uid)

    def _middle_down(self, obj, event):
        if self._object_active():
            self._begin("translate")
        else:
            self.style.OnMiddleButtonDown()

    def _middle_up(self, obj, event):
        if self._drag:
            self._end()
        else:
            self.style.OnMiddleButtonUp()

    def _right_down(self, obj, event):
        if self._object_active():
            self._begin("roll")
        else:
            self.style.OnRightButtonDown()

    def _right_up(self, obj, event):
        if self._drag:
            self._end()
        else:
            self.style.OnRightButtonUp()

    def _move(self, obj, event):
        if not self._drag:
            self.style.OnMouseMove()
            return
        x, y = self._pos()
        dx, dy = x - self._last_pos[0], y - self._last_pos[1]
        self._last_pos = (x, y)
        if dx == 0 and dy == 0:
            return
        self._moved = True
        if self._drag == "rotate":
            self.rotate_selected(dx, dy)
        elif self._drag == "translate":
            self.translate_selected(dx, dy)
        elif self._drag == "roll":
            self.roll_selected(-dx * ROTATE_DEG_PER_PX)

    def _wheel_fwd(self, obj, event):
        if self._object_active():
            self.scale_selected(1.08)
            self.transform_finished.emit(self.scene.selected)
        else:
            self.style.OnMouseWheelForward()

    def _wheel_back(self, obj, event):
        if self._object_active():
            self.scale_selected(1 / 1.08)
            self.transform_finished.emit(self.scene.selected)
        else:
            self.style.OnMouseWheelBackward()

    def _char(self, obj, event):
        # Swallow VTK's default key bindings ('q'/'e' would close the window,
        # 'w'/'s' would silently restyle every actor). Only keep 'r' = reset view.
        if self._iren.GetKeyCode() in ("r", "R"):
            self.scene.reset_camera("front")
