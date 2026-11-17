"""Left panel: video source, live preview with detections, object list, recognition settings."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QListWidget,
                               QListWidgetItem, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..i18n import kind_name, tr
from ..synthetic import SCENES
from ..vision.preprocessing import PreprocessParams
from .video_widget import VideoWidget
from .widgets import LabeledSlider

KIND_COLORS = {
    "triangle": "#4fc3f7", "rectangle": "#81c784", "square": "#81c784", "quadrilateral": "#aed581",
    "polygon": "#4db6ac", "circle": "#ba68c8", "ellipse": "#f06292", "closed_curve": "#90a4ae",
    "line": "#fff176", "arc": "#ffb74d", "function": "#ff8a65", "freehand": "#b0bec5",
}


class CameraPanel(QWidget):
    start_camera = Signal(int)
    stop_camera = Signal()
    load_image = Signal()
    load_sample = Signal(str)
    freeze_toggled = Signal(bool)
    capture_extract = Signal()
    extract_selected = Signal()
    params_changed = Signal(object)
    roi_mode_toggled = Signal(bool)
    roi_cleared = Signal()
    detection_selected = Signal(int)           # detection id (0 = none)
    preview_mode_changed = Signal(str)
    gestures_toggled = Signal(bool)
    gesture_mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = PreprocessParams()
        self._updating = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        self.setMinimumWidth(390)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setSpacing(8)

        # --- source ---------------------------------------------------------
        src = QGroupBox(tr("camera"))
        sl = QVBoxLayout(src)
        row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItem(tr("camera_searching"), -1)
        self.camera_combo.setToolTip(tr("camera_select"))
        self.btn_camera = QPushButton(tr("camera_start"))
        self.btn_camera.setCheckable(True)
        row.addWidget(self.camera_combo, 1)
        row.addWidget(self.btn_camera)
        sl.addLayout(row)
        row = QHBoxLayout()
        self.btn_image = QPushButton(tr("load_image"))
        self.sample_combo = QComboBox()
        self.sample_combo.addItem(tr("load_sample"), "")
        for name in SCENES:
            self.sample_combo.addItem(name, name)
        row.addWidget(self.btn_image)
        row.addWidget(self.sample_combo, 1)
        sl.addLayout(row)
        lay.addWidget(src)

        # --- preview --------------------------------------------------------
        self.video = VideoWidget()
        self.video.setMinimumHeight(250)
        lay.addWidget(self.video, 3)
        row = QHBoxLayout()
        self.preview_combo = QComboBox()
        for key, mode in (("preview_camera", "camera"), ("preview_rectified", "rectified"),
                          ("preview_binary", "binary")):
            self.preview_combo.addItem(tr(key), mode)
        self.chk_overlays = QCheckBox(tr("show_overlays"))
        self.chk_overlays.setChecked(True)
        row.addWidget(self.preview_combo, 1)
        row.addWidget(self.chk_overlays)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.btn_freeze = QPushButton(tr("freeze"))
        self.btn_freeze.setCheckable(True)
        self.btn_capture = QPushButton(tr("capture_extract"))
        self.btn_capture.setObjectName("primary")
        row.addWidget(self.btn_freeze)
        row.addWidget(self.btn_capture, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.btn_roi = QPushButton(tr("manual_region"))
        self.btn_roi.setCheckable(True)
        self.btn_roi_clear = QPushButton(tr("clear_region"))
        row.addWidget(self.btn_roi)
        row.addWidget(self.btn_roi_clear)
        lay.addLayout(row)

        # --- detected objects ----------------------------------------------
        det = QGroupBox(tr("detected_objects"))
        dl = QVBoxLayout(det)
        self.object_list = QListWidget()
        self.object_list.setMinimumHeight(90)
        self.object_list.setMaximumHeight(150)
        dl.addWidget(self.object_list)
        self.btn_extract = QPushButton(tr("extract_selected"))
        dl.addWidget(self.btn_extract)
        lay.addWidget(det)

        # --- recognition settings -----------------------------------------
        rec = QGroupBox(tr("recognition_settings"))
        form = QFormLayout(rec)
        form.setLabelAlignment(Qt.AlignLeft)
        self.thr_mode = QComboBox()
        for key, mode in (("thr_adaptive", "adaptive"), ("thr_otsu", "otsu"), ("thr_manual", "manual")):
            self.thr_mode.addItem(tr(key), mode)
        self.s_threshold = LabeledSlider(20, 240, self.params.manual_threshold)
        self.s_sensitivity = LabeledSlider(2, 40, self.params.adaptive_c)
        self.s_block = QSpinBox()
        self.s_block.setRange(11, 151)
        self.s_block.setSingleStep(2)
        self.s_block.setValue(self.params.adaptive_block)
        self.s_contrast = LabeledSlider(1.0, 4.0, self.params.contrast, 0.1, 1)
        self.s_noise = LabeledSlider(0, 5, self.params.noise_reduction)
        self.s_edges = LabeledSlider(0.0, 1.0, self.params.edge_sensitivity, 0.05, 2)
        self.s_min_size = LabeledSlider(0.01, 0.4, self.params.min_object_size, 0.01, 2)
        self.chk_shadows = QCheckBox(tr("remove_shadows"))
        self.chk_paper = QCheckBox(tr("detect_paper"))
        self.chk_axes = QCheckBox(tr("detect_axes"))
        self.chk_border = QCheckBox(tr("ignore_border"))
        form.addRow(tr("threshold_mode"), self.thr_mode)
        form.addRow(tr("threshold"), self.s_threshold)
        form.addRow(tr("sensitivity"), self.s_sensitivity)
        form.addRow(tr("block_size"), self.s_block)
        form.addRow(tr("contrast"), self.s_contrast)
        form.addRow(tr("noise_reduction"), self.s_noise)
        form.addRow(tr("edge_sensitivity"), self.s_edges)
        form.addRow(tr("min_size"), self.s_min_size)
        for chk in (self.chk_shadows, self.chk_paper, self.chk_axes, self.chk_border):
            form.addRow(chk)
        self.btn_defaults = QPushButton(tr("reset_defaults"))
        form.addRow(self.btn_defaults)
        lay.addWidget(rec)

        # --- gestures -------------------------------------------------------
        ges = QGroupBox(tr("gestures"))
        gl = QFormLayout(ges)
        self.chk_gestures = QCheckBox(tr("gestures_enable"))
        self.gesture_mode = QComboBox()
        self.gesture_mode.addItem(tr("gesture_rotate"), "rotate")
        self.gesture_mode.addItem(tr("gesture_translate"), "translate")
        gl.addRow(self.chk_gestures)
        gl.addRow(tr("gesture_mode"), self.gesture_mode)
        lay.addWidget(ges)
        lay.addStretch(1)

        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self._load_params(self.params)
        self._connect()

    # ---------------------------------------------------------------- wiring
    def _connect(self) -> None:
        self.btn_camera.toggled.connect(self._camera_toggled)
        self.btn_image.clicked.connect(self.load_image)
        self.sample_combo.activated.connect(self._sample_chosen)
        self.btn_freeze.toggled.connect(self._freeze)
        self.btn_capture.clicked.connect(self.capture_extract)
        self.btn_extract.clicked.connect(self.extract_selected)
        self.btn_roi.toggled.connect(self._roi_toggled)
        self.btn_roi_clear.clicked.connect(self.roi_cleared)
        self.video.roi_selected.connect(lambda _: self.btn_roi.setChecked(False))
        self.object_list.currentItemChanged.connect(self._list_selection)
        self.preview_combo.currentIndexChanged.connect(
            lambda _: self.preview_mode_changed.emit(self.preview_combo.currentData()))
        self.chk_gestures.toggled.connect(self.gestures_toggled)
        self.gesture_mode.currentIndexChanged.connect(
            lambda _: self.gesture_mode_changed.emit(self.gesture_mode.currentData()))
        for w in (self.s_threshold, self.s_sensitivity, self.s_contrast, self.s_noise, self.s_edges,
                  self.s_min_size):
            w.valueChanged.connect(self._params_from_ui)
        self.s_block.valueChanged.connect(self._params_from_ui)
        self.thr_mode.currentIndexChanged.connect(self._params_from_ui)
        for chk in (self.chk_shadows, self.chk_paper, self.chk_axes, self.chk_border):
            chk.toggled.connect(self._params_from_ui)
        self.btn_defaults.clicked.connect(lambda: (self._load_params(PreprocessParams()), self._params_from_ui()))

    def _camera_toggled(self, on: bool) -> None:
        self.btn_camera.setText(tr("camera_stop") if on else tr("camera_start"))
        if on:
            idx = self.camera_combo.currentData()
            if idx is None or idx < 0:
                self.btn_camera.blockSignals(True)
                self.btn_camera.setChecked(False)
                self.btn_camera.setText(tr("camera_start"))
                self.btn_camera.blockSignals(False)
                return
            self.start_camera.emit(int(idx))
        else:
            self.stop_camera.emit()

    def set_camera_running(self, running: bool) -> None:
        self.btn_camera.blockSignals(True)
        self.btn_camera.setChecked(running)
        self.btn_camera.setText(tr("camera_stop") if running else tr("camera_start"))
        self.btn_camera.blockSignals(False)

    def set_cameras(self, indices: list[int]) -> None:
        self.camera_combo.clear()
        if not indices:
            self.camera_combo.addItem(tr("camera_none"), -1)
        for i in indices:
            self.camera_combo.addItem(tr("camera_index", i), i)

    def _sample_chosen(self, _):
        name = self.sample_combo.currentData()
        if name:
            self.load_sample.emit(name)

    def _freeze(self, on: bool) -> None:
        self.btn_freeze.setText(tr("unfreeze") if on else tr("freeze"))
        self.freeze_toggled.emit(on)

    def set_frozen(self, on: bool) -> None:
        self.btn_freeze.blockSignals(True)
        self.btn_freeze.setChecked(on)
        self.btn_freeze.setText(tr("unfreeze") if on else tr("freeze"))
        self.btn_freeze.blockSignals(False)

    def _roi_toggled(self, on: bool) -> None:
        self.video.roi_mode = on
        if on:
            self.preview_combo.setCurrentIndex(0)
        self.roi_mode_toggled.emit(on)

    def set_gestures_checked(self, on: bool) -> None:
        self.chk_gestures.blockSignals(True)
        self.chk_gestures.setChecked(on)
        self.chk_gestures.blockSignals(False)

    # ---------------------------------------------------------------- params
    def _load_params(self, p: PreprocessParams) -> None:
        self._updating = True
        self.thr_mode.setCurrentIndex(max(0, self.thr_mode.findData(p.threshold_mode)))
        self.s_threshold.setValue(p.manual_threshold)
        self.s_sensitivity.setValue(p.adaptive_c)
        self.s_block.setValue(p.adaptive_block)
        self.s_contrast.setValue(p.contrast)
        self.s_noise.setValue(p.noise_reduction)
        self.s_edges.setValue(p.edge_sensitivity)
        self.s_min_size.setValue(p.min_object_size)
        self.chk_shadows.setChecked(p.remove_shadows)
        self.chk_paper.setChecked(p.detect_paper)
        self.chk_axes.setChecked(p.detect_axes)
        self.chk_border.setChecked(p.ignore_border_objects)
        self._updating = False
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        mode = self.thr_mode.currentData()
        self.s_threshold.setEnabled(mode == "manual")
        self.s_sensitivity.setEnabled(mode == "adaptive")
        self.s_block.setEnabled(mode == "adaptive")

    def _params_from_ui(self, *_):
        if self._updating:
            return
        p = self.params
        p.threshold_mode = self.thr_mode.currentData()
        p.manual_threshold = int(self.s_threshold.value())
        p.adaptive_c = int(self.s_sensitivity.value())
        p.adaptive_block = int(self.s_block.value()) | 1
        p.contrast = float(self.s_contrast.value())
        p.noise_reduction = int(self.s_noise.value())
        p.edge_sensitivity = float(self.s_edges.value())
        p.min_object_size = float(self.s_min_size.value())
        p.remove_shadows = self.chk_shadows.isChecked()
        p.detect_paper = self.chk_paper.isChecked()
        p.detect_axes = self.chk_axes.isChecked()
        p.ignore_border_objects = self.chk_border.isChecked()
        self._sync_enabled()
        self.params_changed.emit(p.copy())

    # ------------------------------------------------------------ detections
    def set_detections(self, objects, selected_id: int) -> None:
        """Refresh the list only when the content changed (keeps the UI calm at video rate)."""
        entries = [(o.id, f"#{o.id}  {kind_name(o.kind)}" + self._family_suffix(o) + f"  ·  {o.confidence:.0%}")
                   for o in objects]
        current = [(self.object_list.item(i).data(Qt.UserRole), self.object_list.item(i).text())
                   for i in range(self.object_list.count())]
        self.object_list.blockSignals(True)
        if entries != current:
            self.object_list.clear()
            for oid, text in entries:
                it = QListWidgetItem(text)
                it.setData(Qt.UserRole, oid)
                it.setForeground(Qt.white)
                self.object_list.addItem(it)
        for i in range(self.object_list.count()):
            if self.object_list.item(i).data(Qt.UserRole) == selected_id:
                self.object_list.setCurrentRow(i)
                break
        else:
            self.object_list.setCurrentRow(-1)
        self.object_list.blockSignals(False)

    @staticmethod
    def _family_suffix(o) -> str:
        from ..i18n import FAMILY_NAMES
        if o.fit is not None and o.fit.best is not None and o.fit.recognized and o.kind.value == "function":
            return f" ({FAMILY_NAMES.get(o.fit.best.family, o.fit.best.family)})"
        return ""

    def _list_selection(self, cur, _prev):
        self.detection_selected.emit(int(cur.data(Qt.UserRole)) if cur is not None else 0)

    @property
    def preview_mode(self) -> str:
        return self.preview_combo.currentData()

    @property
    def overlays_enabled(self) -> bool:
        return self.chk_overlays.isChecked()
