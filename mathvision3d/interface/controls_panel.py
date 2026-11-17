"""Right panel: object info, equation, analysis, display mode, transform, rendering, calibration."""

from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

from ..core.math_object import DisplayMode, MathObject
from ..i18n import FAMILY_NAMES, tr
from ..mathematics.expressions import ExpressionError, compile_expression
from ..mathematics.function_fitting import fmt
from ..vision.models import ShapeKind
from .widgets import LabeledSlider

MODE_KEYS = {
    DisplayMode.FLAT: "mode_flat", DisplayMode.EXTRUDE: "mode_extrude", DisplayMode.ANALYSIS: "mode_analysis",
    DisplayMode.CURVE_2D: "mode_curve2d", DisplayMode.SPACE_CURVE: "mode_space", DisplayMode.SURFACE: "mode_surface",
}
USER_TRANSFORM_MODES = (DisplayMode.EXTRUDE, DisplayMode.SPACE_CURVE, DisplayMode.SURFACE)


def _spin(lo, hi, val, step=0.1, decimals=2) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    s.setKeyboardTracking(False)
    return s


class ControlsPanel(QWidget):
    object_selected = Signal(object)        # uid or None
    delete_requested = Signal(str)
    clear_requested = Signal()
    changed = Signal(str, str)              # uid, geometry | transform | render | labels
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.obj: MathObject | None = None
        self.objects: dict[str, MathObject] = {}
        self._populating = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        self.setMinimumWidth(380)
        body = QWidget()
        scroll.setWidget(body)
        self.body_layout = lay = QVBoxLayout(body)
        lay.setSpacing(8)

        self._build_workspace(lay)
        self._build_info(lay)
        self._build_analysis(lay)
        self._build_display(lay)
        self._build_transform(lay)
        self._build_rendering(lay)
        self._build_calibration(lay)
        lay.addStretch(1)
        for combo in self.findChildren(QComboBox):
            # long Georgian item texts must not dictate the panel's minimum width
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        self._set_enabled(False)

    # ================================================================ build
    def _build_workspace(self, lay):
        g = QGroupBox(tr("workspace_objects"))
        v = QVBoxLayout(g)
        self.ws_list = QListWidget()
        self.ws_list.setMaximumHeight(110)
        v.addWidget(self.ws_list)
        row = QHBoxLayout()
        self.btn_delete = QPushButton(tr("delete"))
        self.btn_clear = QPushButton(tr("clear_all"))
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_clear)
        v.addLayout(row)
        lay.addWidget(g)
        self.ws_list.currentItemChanged.connect(
            lambda cur, _: self.object_selected.emit(cur.data(Qt.UserRole) if cur else None))
        self.btn_delete.clicked.connect(lambda: self.obj and self.delete_requested.emit(self.obj.uid))
        self.btn_clear.clicked.connect(self.clear_requested)

    def _build_info(self, lay):
        g = self.grp_info = QGroupBox(tr("object_info"))
        v = QVBoxLayout(g)
        self.lbl_title = QLabel(tr("no_object"))
        self.lbl_title.setObjectName("title")
        self.lbl_title.setWordWrap(True)
        self.lbl_equation = QLabel("")
        self.lbl_equation.setObjectName("equation")
        self.lbl_equation.setWordWrap(True)
        self.lbl_equation.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.bar_conf = QProgressBar()
        self.bar_conf.setRange(0, 100)
        self.lbl_fit = QLabel("")
        self.lbl_fit.setObjectName("muted")
        self.lbl_fit.setWordWrap(True)
        self.lbl_coords = QLabel("")
        self.lbl_coords.setObjectName("chip")
        self.lbl_coords.setWordWrap(True)
        self.lbl_candidates = QLabel("")
        self.lbl_candidates.setObjectName("muted")
        self.lbl_candidates.setWordWrap(True)
        self.lbl_note = QLabel("")
        self.lbl_note.setObjectName("warn")
        self.lbl_note.setWordWrap(True)
        v.addWidget(self.lbl_title)
        v.addWidget(self.lbl_equation)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("confidence")))
        row.addWidget(self.bar_conf, 1)
        v.addLayout(row)
        v.addWidget(self.lbl_fit)
        v.addWidget(self.lbl_coords)
        v.addWidget(self.lbl_candidates)
        v.addWidget(self.lbl_note)
        lay.addWidget(g)

    def _build_analysis(self, lay):
        g = self.grp_analysis = QGroupBox(tr("analysis"))
        v = QVBoxLayout(g)
        self.lbl_measure = QLabel("")
        self.lbl_measure.setWordWrap(True)
        self.lbl_measure.setTextFormat(Qt.RichText)
        self.lbl_measure.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_measure)
        lay.addWidget(g)

    def _build_display(self, lay):
        g = self.grp_display = QGroupBox(tr("display"))
        f = self.display_form = QFormLayout(g)
        self.cmb_mode = QComboBox()
        f.addRow(self.cmb_mode)
        self.sp_depth = _spin(0.01, 50, 1.0, 0.1)
        f.addRow(tr("extrude_depth"), self.sp_depth)
        self.ed_z = QLineEdit("0.5*x")
        self.ed_z.setToolTip("x, y = f(x), t ∈ [0, 1]; sin cos tan exp log sqrt abs pi")
        f.addRow(tr("space_z"), self.ed_z)
        self.cmb_surface = QComboBox()
        for key, val in (("surf_translational", "translational"), ("surf_product", "product"),
                         ("surf_radial", "radial"), ("surf_revolution", "revolution")):
            self.cmb_surface.addItem(tr(key), val)
        f.addRow(tr("surface_type"), self.cmb_surface)
        yr = QWidget()
        h = QHBoxLayout(yr)
        h.setContentsMargins(0, 0, 0, 0)
        self.sp_ymin = _spin(-100, 100, -2.0)
        self.sp_ymax = _spin(-100, 100, 2.0)
        h.addWidget(self.sp_ymin)
        h.addWidget(self.sp_ymax)
        f.addRow(tr("y_range"), yr)
        self._yrange_row = yr
        self.chk_deriv = QCheckBox(tr("show_derivative"))
        self.chk_tangent = QCheckBox(tr("show_tangent"))
        self.chk_samples = QCheckBox(tr("show_samples"))
        f.addRow(self.chk_deriv)
        f.addRow(self.chk_tangent)
        self.sl_tangent = QSlider(Qt.Horizontal)
        self.sl_tangent.setRange(0, 1000)
        self.lbl_tangent = QLabel("")
        tr_row = QWidget()
        h = QHBoxLayout(tr_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.sl_tangent, 1)
        h.addWidget(self.lbl_tangent)
        f.addRow(tr("tangent_at"), tr_row)
        self._tangent_row = tr_row
        f.addRow(self.chk_samples)
        self.lbl_notice = QLabel(tr("transform_notice"))
        self.lbl_notice.setObjectName("warn")
        self.lbl_notice.setWordWrap(True)
        f.addRow(self.lbl_notice)
        lay.addWidget(g)

        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        self.sp_depth.valueChanged.connect(lambda v: self._set_opt("extrude_depth", v))
        self.ed_z.editingFinished.connect(self._z_changed)
        self.cmb_surface.currentIndexChanged.connect(lambda _: self._set_opt("surface_type", self.cmb_surface.currentData()))
        self.sp_ymin.valueChanged.connect(lambda v: self._set_opt("y_min", v))
        self.sp_ymax.valueChanged.connect(lambda v: self._set_opt("y_max", v))
        self.chk_deriv.toggled.connect(lambda v: self._set_opt("show_derivative", v))
        self.chk_tangent.toggled.connect(lambda v: (self._set_opt("show_tangent", v), self._refresh_measurements()))
        self.chk_samples.toggled.connect(lambda v: self._set_opt("show_samples", v))
        self.sl_tangent.valueChanged.connect(self._tangent_moved)

    def _build_transform(self, lay):
        g = self.grp_transform = QGroupBox(tr("transform"))
        grid = QGridLayout(g)
        grid.addWidget(QLabel(tr("rotation")), 0, 0, 1, 4)
        self.sp_rot = []
        for i, axis in enumerate("XYZ"):
            grid.addWidget(QLabel(axis), i + 1, 0)
            s = _spin(-180, 180, 0.0, 5.0, 1)
            s.setWrapping(True)
            minus, plus = QPushButton("−15°"), QPushButton("+15°")
            for b in (minus, plus):
                b.setMaximumWidth(56)
            minus.clicked.connect(lambda _=False, a=i: self._rotate_step(a, -15))
            plus.clicked.connect(lambda _=False, a=i: self._rotate_step(a, 15))
            grid.addWidget(s, i + 1, 1)
            grid.addWidget(minus, i + 1, 2)
            grid.addWidget(plus, i + 1, 3)
            s.valueChanged.connect(self._euler_changed)
            self.sp_rot.append(s)
        grid.addWidget(QLabel(tr("position")), 4, 0, 1, 4)
        self.sp_pos = []
        for i, axis in enumerate("XYZ"):
            s = _spin(-100, 100, 0.0, 0.1)
            s.valueChanged.connect(self._position_changed)
            grid.addWidget(QLabel(axis), 5 + i, 0)
            grid.addWidget(s, 5 + i, 1, 1, 3)
            self.sp_pos.append(s)
        grid.addWidget(QLabel(tr("scale")), 8, 0, 1, 4)
        self.sp_scale = _spin(0.02, 50, 1.0, 0.1)
        self.sp_scale.valueChanged.connect(self._scale_changed)
        grid.addWidget(self.sp_scale, 9, 1, 1, 3)
        self.btn_reset_orient = QPushButton(tr("reset_orientation"))
        self.btn_restore = QPushButton(tr("restore_original"))
        grid.addWidget(self.btn_reset_orient, 10, 0, 1, 4)
        grid.addWidget(self.btn_restore, 11, 0, 1, 4)
        self.btn_reset_orient.clicked.connect(self._reset_orientation)
        self.btn_restore.clicked.connect(self._restore)
        lay.addWidget(g)

    def _build_rendering(self, lay):
        g = self.grp_render = QGroupBox(tr("rendering"))
        f = QFormLayout(g)
        self.cmb_style = QComboBox()
        for key, val in (("style_surface", "surface"), ("style_wireframe", "wireframe"),
                         ("style_points", "points"), ("style_edges", "edges")):
            self.cmb_style.addItem(tr(key), val)
        f.addRow(tr("style"), self.cmb_style)
        self.btn_color = QPushButton(" ")
        f.addRow(tr("color"), self.btn_color)
        self.sl_width = LabeledSlider(1, 12, 4, 0.5, 1)
        f.addRow(tr("line_width"), self.sl_width)
        self.sl_opacity = LabeledSlider(0.1, 1.0, 1.0, 0.05, 2)
        f.addRow(tr("opacity"), self.sl_opacity)
        self.chk_vertices = QCheckBox(tr("show_vertices"))
        self.chk_labels = QCheckBox(tr("show_labels"))
        f.addRow(self.chk_vertices)
        f.addRow(self.chk_labels)
        lay.addWidget(g)
        self.cmb_style.currentIndexChanged.connect(lambda _: self._set_render("style", self.cmb_style.currentData()))
        self.btn_color.clicked.connect(self._pick_color)
        self.sl_width.valueChanged.connect(lambda v: self._set_render("line_width", v))
        self.sl_opacity.valueChanged.connect(lambda v: self._set_render("opacity", v))
        self.chk_vertices.toggled.connect(lambda v: self._set_render("show_vertices", v))
        self.chk_labels.toggled.connect(lambda v: self._set_render("show_labels", v))

    def _build_calibration(self, lay):
        g = self.grp_cal = QGroupBox(tr("calibration"))
        f = QFormLayout(g)
        self.lbl_ref = QLabel("")
        self.lbl_ref.setObjectName("muted")
        self.lbl_ref.setWordWrap(True)
        f.addRow(self.lbl_ref)
        self.sp_ref = _spin(0.0001, 1e6, 1.0, 0.5, 4)
        f.addRow(tr("reference_length"), self.sp_ref)
        self.ed_unit = QLineEdit("cm")
        f.addRow(tr("unit"), self.ed_unit)
        self.chk_xstart = QCheckBox(tr("x_start"))
        self.sp_xstart = _spin(-1e6, 1e6, 0.0, 0.5, 4)
        row = QWidget()
        h = QVBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.chk_xstart)
        h.addWidget(self.sp_xstart)
        f.addRow(row)
        self._xstart_row = row
        rowb = QHBoxLayout()
        self.btn_cal = QPushButton(tr("apply"))
        self.btn_cal.setObjectName("primary")
        self.btn_cal_clear = QPushButton(tr("clear_calibration"))
        rowb.addWidget(self.btn_cal)
        rowb.addWidget(self.btn_cal_clear)
        f.addRow(rowb)
        lay.addWidget(g)
        self.btn_cal.clicked.connect(self._apply_calibration)
        self.btn_cal_clear.clicked.connect(self._clear_calibration)

    # ============================================================= workspace
    def add_object(self, obj: MathObject) -> None:
        self.objects[obj.uid] = obj
        it = QListWidgetItem(obj.title)
        it.setData(Qt.UserRole, obj.uid)
        it.setForeground(QColor(obj.render.color))
        self.ws_list.addItem(it)

    def remove_object(self, uid: str) -> None:
        self.objects.pop(uid, None)
        for i in range(self.ws_list.count()):
            if self.ws_list.item(i).data(Qt.UserRole) == uid:
                self.ws_list.blockSignals(True)
                self.ws_list.takeItem(i)
                self.ws_list.blockSignals(False)
                break
        if self.obj is not None and self.obj.uid == uid:
            self.show_object(None)

    def clear(self) -> None:
        self.objects.clear()
        self.ws_list.blockSignals(True)
        self.ws_list.clear()
        self.ws_list.blockSignals(False)
        self.show_object(None)

    def _sync_list(self, uid) -> None:
        self.ws_list.blockSignals(True)
        self.ws_list.setCurrentRow(-1)
        for i in range(self.ws_list.count()):
            it = self.ws_list.item(i)
            if it.data(Qt.UserRole) == uid:
                self.ws_list.setCurrentRow(i)
            o = self.objects.get(it.data(Qt.UserRole))
            if o is not None:
                it.setText(o.title)
                it.setForeground(QColor(o.render.color))
        self.ws_list.blockSignals(False)

    # ============================================================ populate
    def _set_enabled(self, on: bool) -> None:
        for g in (self.grp_display, self.grp_transform, self.grp_render, self.grp_cal):
            g.setEnabled(on)

    def show_object(self, obj: MathObject | None) -> None:
        self.obj = obj
        self._sync_list(obj.uid if obj else None)
        self._set_enabled(obj is not None)
        if obj is None:
            self.lbl_title.setText(tr("no_object"))
            for lbl in (self.lbl_equation, self.lbl_fit, self.lbl_coords, self.lbl_candidates, self.lbl_note,
                        self.lbl_measure, self.lbl_ref):
                lbl.setText("")
            self.bar_conf.setValue(0)
            return
        self._populating = True
        try:
            self.cmb_mode.clear()
            for m in obj.available_modes():
                self.cmb_mode.addItem(tr(MODE_KEYS[m]), m.value)
            self.cmb_mode.setCurrentIndex(max(0, self.cmb_mode.findData(obj.options.mode.value)))
            o = obj.options
            self.sp_depth.setValue(o.extrude_depth)
            self.ed_z.setText(o.z_expression)
            self.cmb_surface.setCurrentIndex(max(0, self.cmb_surface.findData(o.surface_type)))
            self.sp_ymin.setValue(o.y_min)
            self.sp_ymax.setValue(o.y_max)
            self.chk_deriv.setChecked(o.show_derivative)
            self.chk_tangent.setChecked(o.show_tangent)
            self.chk_samples.setChecked(o.show_samples)
            r = obj.render
            self.cmb_style.setCurrentIndex(max(0, self.cmb_style.findData(r.style)))
            self.btn_color.setStyleSheet(f"background: {r.color};")
            self.sl_width.setValue(r.line_width)
            self.sl_opacity.setValue(r.opacity)
            self.chk_vertices.setChecked(r.show_vertices)
            self.chk_labels.setChecked(r.show_labels)
            self._update_tangent_slider()
            self.refresh_transform()
            self._refresh_calibration()
        finally:
            self._populating = False
        self._update_mode_visibility()
        self.refresh_info()

    def refresh_transform(self) -> None:
        if self.obj is None:
            return
        was = self._populating
        self._populating = True
        t = self.obj.transform
        for s, v in zip(self.sp_rot, t.euler_deg()):
            s.setValue(float(v))
        for s, v in zip(self.sp_pos, t.position):
            s.setValue(float(v))
        self.sp_scale.setValue(t.scale)
        self._populating = was

    def refresh_info(self) -> None:
        obj = self.obj
        if obj is None:
            return
        self.lbl_title.setText(obj.title)
        eq = obj.equation()
        conf = obj.detection.confidence
        notes = []
        if obj.is_recognized_function:
            fit = obj.fit
            conf = fit.confidence
            self.lbl_equation.setText(eq)
            self.lbl_fit.setText(f"{tr('fit_error')}: {fit.best.nrmse:.2%}")
            cands = ", ".join(f"{FAMILY_NAMES.get(c.family, c.family)}: {c.nrmse:.1%}" for c in fit.candidates[:4])
            self.lbl_candidates.setText(f"{tr('candidates')}: {cands}")
            if not obj.coords.calibrated:
                notes.append(tr("estimated_shape"))
            if fit.best.family == "sine" and not obj.origin_meaningful:
                notes.append(tr("phase_note"))
        elif obj.kind in (ShapeKind.FREEHAND, ShapeKind.FUNCTION):
            self.lbl_equation.setText("—")
            nrmse = obj.fit.nrmse if obj.fit is not None else float("nan")
            self.lbl_fit.setText(f"{tr('fit_error')}: {nrmse:.2%}" if np.isfinite(nrmse) else "")
            if obj.fit is not None and obj.fit.candidates:
                best = obj.fit.candidates[0]
                self.lbl_candidates.setText(f"{tr('candidates')}: {FAMILY_NAMES.get(best.family, best.family)} "
                                            f"{best.nrmse:.1%}")
            else:
                self.lbl_candidates.setText("")
            notes.append(tr("freehand_note"))
        else:
            self.lbl_equation.setText("")
            res = obj.detection.params.get("residual", obj.detection.params.get("rms"))
            self.lbl_fit.setText(f"{tr('fit_error')}: {float(res):.2%}" if isinstance(res, float) and
                                 obj.kind != ShapeKind.LINE else "")
            self.lbl_candidates.setText("")
        self.bar_conf.setValue(int(round(conf * 100)))
        self.lbl_coords.setText(f"{tr('coordinates_status')}: " + (
            f"{tr('calibrated')} ({obj.coords.unit})" if obj.coords.calibrated else tr("normalized")))
        self.lbl_note.setText("\n".join(notes))
        self._refresh_measurements()
        self._sync_list(obj.uid)

    def _refresh_measurements(self) -> None:
        obj = self.obj
        if obj is None:
            return
        rows = "".join(
            f"<tr><td style='color:#8b98a9; padding-right:8px; vertical-align:top'>{html.escape(m.label)}</td>"
            f"<td>{html.escape(m.value)}</td></tr>" for m in obj.measurements())
        note = (tr("measure_note_cal", obj.coords.unit) if obj.coords.calibrated else tr("measure_note_norm"))
        self.lbl_measure.setText(f"<table cellspacing='2'>{rows}</table>"
                                 f"<p style='color:#ffb74d'>{html.escape(note)}</p>")

    def _refresh_calibration(self) -> None:
        obj = self.obj
        px, key = obj.reference_px()
        self.lbl_ref.setText(tr("reference_is", tr(key)) + f" — {px:.1f} {tr('pixels')}")
        self._xstart_row.setVisible(obj.kind.is_curve)
        if obj.coords.calibrated:
            self.sp_ref.setValue(px / obj.coords.px_per_unit_x)
            self.ed_unit.setText(obj.coords.unit)
        self.chk_xstart.setChecked(obj.x_start is not None)
        if obj.x_start is not None:
            self.sp_xstart.setValue(obj.x_start)

    def _update_mode_visibility(self) -> None:
        obj = self.obj
        if obj is None:
            return
        mode = obj.options.mode
        f = self.display_form
        is_curve = obj.kind.is_curve
        f.setRowVisible(self.sp_depth, mode == DisplayMode.EXTRUDE)
        f.setRowVisible(self.ed_z, mode == DisplayMode.SPACE_CURVE)
        f.setRowVisible(self.cmb_surface, mode == DisplayMode.SURFACE)
        f.setRowVisible(self._yrange_row, mode == DisplayMode.SURFACE and obj.options.surface_type == "translational")
        analysable = is_curve and obj.is_function_graph and mode == DisplayMode.CURVE_2D
        f.setRowVisible(self.chk_deriv, analysable)
        f.setRowVisible(self.chk_tangent, analysable)
        f.setRowVisible(self._tangent_row, analysable and obj.options.show_tangent)
        f.setRowVisible(self.chk_samples, is_curve)
        f.setRowVisible(self.lbl_notice, mode in USER_TRANSFORM_MODES)

    def _update_tangent_slider(self) -> None:
        obj = self.obj
        if obj is None or not obj.is_function_graph:
            return
        x0, x1 = obj.domain()
        if not (x0 <= obj.options.tangent_x <= x1):
            obj.options.tangent_x = (x0 + x1) / 2
        frac = (obj.options.tangent_x - x0) / max(x1 - x0, 1e-9)
        self.sl_tangent.setValue(int(round(frac * 1000)))
        self.lbl_tangent.setText(fmt(obj.options.tangent_x))

    # ============================================================== handlers
    def _emit(self, what: str) -> None:
        if self.obj is not None and not self._populating:
            self.changed.emit(self.obj.uid, what)

    def _set_opt(self, name, value) -> None:
        if self.obj is None or self._populating:
            return
        setattr(self.obj.options, name, value)
        self._update_mode_visibility()
        self._emit("geometry")

    def _set_render(self, name, value) -> None:
        if self.obj is None or self._populating:
            return
        setattr(self.obj.render, name, value)
        self._emit("render")

    def _mode_changed(self, _):
        if self.obj is None or self._populating:
            return
        self.obj.options.mode = DisplayMode(self.cmb_mode.currentData())
        self._update_mode_visibility()
        self._emit("geometry")

    def _z_changed(self) -> None:
        if self.obj is None or self._populating:
            return
        text = self.ed_z.text()
        try:
            compile_expression(text)
        except ExpressionError as exc:
            self.message.emit(tr("expression_error", exc))
            return
        self._set_opt("z_expression", text)

    def _tangent_moved(self, v: int) -> None:
        obj = self.obj
        if obj is None or self._populating or not obj.is_function_graph:
            return
        x0, x1 = obj.domain()
        obj.options.tangent_x = x0 + (x1 - x0) * v / 1000.0
        self.lbl_tangent.setText(fmt(obj.options.tangent_x))
        self._refresh_measurements()
        self._emit("geometry")

    def _euler_changed(self, _):
        if self.obj is None or self._populating:
            return
        self.obj.transform.set_euler_deg([s.value() for s in self.sp_rot])
        self._emit("transform")

    def _rotate_step(self, axis: int, deg: float) -> None:
        if self.obj is None:
            return
        v = np.zeros(3)
        v[axis] = 1.0
        self.obj.transform.rotate_world(v, deg)
        self.refresh_transform()
        self._emit("transform")

    def _position_changed(self, _):
        if self.obj is None or self._populating:
            return
        self.obj.transform.position = np.array([s.value() for s in self.sp_pos])
        self._emit("transform")

    def _scale_changed(self, v):
        if self.obj is None or self._populating:
            return
        self.obj.transform.scale = float(v)
        self._emit("transform")

    def _reset_orientation(self) -> None:
        if self.obj is None:
            return
        self.obj.transform.reset_orientation()
        self.refresh_transform()
        self._emit("transform")

    def _restore(self) -> None:
        if self.obj is None:
            return
        self.obj.restore_original()
        self.show_object(self.obj)
        self.changed.emit(self.obj.uid, "geometry")

    def _pick_color(self) -> None:
        if self.obj is None:
            return
        c = QColorDialog.getColor(QColor(self.obj.render.color), self, tr("color"))
        if c.isValid():
            self.obj.render.color = c.name()
            self.btn_color.setStyleSheet(f"background: {c.name()};")
            self._sync_list(self.obj.uid)
            self._emit("render")

    def _apply_calibration(self) -> None:
        obj = self.obj
        if obj is None:
            return
        x_start = self.sp_xstart.value() if (self.chk_xstart.isChecked() and obj.kind.is_curve) else None
        try:
            obj.calibrate(self.sp_ref.value(), self.ed_unit.text().strip(), x_start)
        except ValueError as exc:
            self.message.emit(str(exc))
            return
        self.message.emit(tr("calibration_applied"))
        self._after_coords_change()

    def _clear_calibration(self) -> None:
        if self.obj is None:
            return
        self.obj.clear_calibration()
        self._after_coords_change()

    def _after_coords_change(self) -> None:
        self._populating = True
        self._update_tangent_slider()
        self._refresh_calibration()
        self._populating = False
        self.refresh_info()
        self._update_mode_visibility()
        self._emit("geometry")
