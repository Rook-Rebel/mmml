"""Main window: wires camera, recognition, workspace, controls and gestures together."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QSplitter, QVBoxLayout, QWidget)

from .. import __version__
from ..camera.camera_manager import CameraProbe, CameraWorker
from ..camera.frame_processor import ProcessingWorker
from ..core.math_object import MathObject
from ..gestures.hand_tracker import GestureOutput, HandTrackerWorker, mediapipe_status
from ..i18n import kind_name, tr
from ..synthetic import SCENES
from ..vision.models import DetectedObject, DetectionResult
from .camera_panel import KIND_COLORS, CameraPanel
from .controls_panel import ControlsPanel
from .video_widget import Overlay
from .visualization_panel import VisualizationPanel

USAGE_HTML = """
<h3>MathVision 3D — სწრაფი სახელმძღვანელო</h3>
<ol>
<li>ჩართეთ კამერა (ან ჩატვირთეთ სურათი / ნიმუში).</li>
<li>დახაზეთ ფიგურა ან გრაფიკი თეთრ ფურცელზე მუქი კალმით და აჩვენეთ კამერას.</li>
<li>დააწკაპუნეთ აღმოჩენილ ობიექტზე ვიდეოში ან სიაში.</li>
<li>დააჭირეთ „გადაღება და ამოღება“ — ობიექტი გადავა 3D სივრცეში.</li>
<li>3D ხედში: მარცხენა ღილაკი — ბრუნვა, შუა/Shift — გადაადგილება, ბორბალი — მასშტაბი.
„ობიექტის მართვის“ რეჟიმში იგივე მოქმედებები ობიექტს ეხება.</li>
<li>მარჯვენა პანელში: ჩვენების რეჟიმი, ექსტრუზია, ზედაპირი, წარმოებული, მხები, კალიბრაცია.</li>
</ol>
<p>თუ ამოცნობა ვერ ხერხდება: შეცვალეთ ზღვარი/კონტრასტი, გამოიყენეთ „რეგიონის ხელით არჩევა“.</p>
"""


class MainWindow(QMainWindow):
    def __init__(self, camera_index: int | None = None, auto_camera: bool = True):
        super().__init__()
        self.setWindowTitle(tr("app_title"))
        self.resize(1680, 980)

        self.camera_panel = CameraPanel()
        self.viz = VisualizationPanel()
        self.controls = ControlsPanel()
        self.scene = self.viz.scene

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(self.camera_panel)
        main_split.addWidget(self.viz)
        main_split.addWidget(self.controls)
        main_split.setSizes([430, 900, 400])
        main_split.setStretchFactor(1, 1)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(6, 4, 6, 4)
        chips = QHBoxLayout()
        self.chip_status = QLabel(f"{tr('status')}: {tr('ready')}")
        self.chip_detect = QLabel(f"{tr('detection')}: —")
        self.chip_fps = QLabel(f"{tr('fps')}: —")
        self.chip_gesture = QLabel(tr("gesture_state", "—"))
        for c in (self.chip_status, self.chip_detect, self.chip_fps, self.chip_gesture):
            c.setObjectName("chip")
            chips.addWidget(c)
        chips.addStretch(1)
        bl.addLayout(chips)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(400)
        bl.addWidget(self.log_view)

        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(main_split)
        vsplit.addWidget(bottom)
        vsplit.setSizes([840, 130])
        root = QWidget()
        root.setObjectName("root")
        rl = QVBoxLayout(root)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.addWidget(vsplit)
        self.setCentralWidget(root)
        self._build_menu()

        # ------------------------------------------------------------- state
        self.camera_worker: CameraWorker | None = None
        self.hand_worker: HandTrackerWorker | None = None
        self.latest_frame: np.ndarray | None = None
        self.frozen_frame: np.ndarray | None = None
        self.frozen = False
        self.latest_result: DetectionResult | None = None
        self.selected_det_id = 0
        self._sel_point: np.ndarray | None = None     # frame coords, used to track selection over frames
        self._last_preview = 0.0
        self._auto_camera = auto_camera
        self._requested_camera = camera_index

        self.processor = ProcessingWorker()
        self.processor.result_ready.connect(self._on_result)
        self.processor.error.connect(lambda msg: self.log(msg.strip().splitlines()[-1]))
        self.processor.start()

        self._connect()
        self.log(f"MathVision 3D {__version__} — {tr('ready')}")
        self.probe = CameraProbe()
        self.probe.found.connect(self._on_cameras_found)
        self.probe.start()

    # ================================================================ setup
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m_file = mb.addMenu(tr("file"))
        a = QAction(tr("load_image"), self)
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self.open_image_dialog)
        m_file.addAction(a)
        a = QAction(tr("load_video"), self)
        a.triggered.connect(self.open_video_dialog)
        m_file.addAction(a)
        m_samples = m_file.addMenu(tr("load_sample"))
        for name in SCENES:
            act = QAction(name, self)
            act.triggered.connect(lambda _=False, n=name: self.load_sample(n))
            m_samples.addAction(act)
        m_file.addSeparator()
        a = QAction(tr("save_screenshot"), self)
        a.triggered.connect(self.save_screenshot)
        m_file.addAction(a)
        a = QAction(tr("export_object"), self)
        a.triggered.connect(self.export_object)
        m_file.addAction(a)
        m_file.addSeparator()
        a = QAction(tr("exit"), self)
        a.triggered.connect(self.close)
        m_file.addAction(a)
        m_view = mb.addMenu(tr("view"))
        for key, view in (("view_front", "front"), ("view_top", "top"), ("view_side", "side"), ("view_iso", "iso")):
            act = QAction(tr(key), self)
            act.triggered.connect(lambda _=False, v=view: self.scene.reset_camera(v))
            m_view.addAction(act)
        m_help = mb.addMenu(tr("help"))
        a = QAction(tr("usage"), self)
        a.triggered.connect(lambda: QMessageBox.information(self, tr("usage"), USAGE_HTML))
        m_help.addAction(a)
        a = QAction(tr("about"), self)
        a.triggered.connect(lambda: QMessageBox.about(
            self, tr("about"), f"<b>MathVision 3D {__version__}</b><br>{tr('app_title')}<br><br>"
                               "OpenCV · NumPy · SciPy · PySide6 · PyVista/VTK · MediaPipe"))
        m_help.addAction(a)

    def _connect(self) -> None:
        cp = self.camera_panel
        cp.start_camera.connect(self.start_camera)
        cp.stop_camera.connect(self.stop_camera)
        cp.load_image.connect(self.open_image_dialog)
        cp.load_sample.connect(self.load_sample)
        cp.freeze_toggled.connect(self.set_frozen)
        cp.capture_extract.connect(self.capture_and_extract)
        cp.extract_selected.connect(self.extract_selected)
        cp.params_changed.connect(self._on_params)
        cp.roi_mode_toggled.connect(lambda on: on and self.log(tr("roi_hint")))
        cp.roi_cleared.connect(self._clear_roi)
        cp.video.roi_selected.connect(self._set_roi)
        cp.video.clicked.connect(self._video_clicked)
        cp.detection_selected.connect(self._detection_selected)
        cp.preview_mode_changed.connect(lambda _: self._update_preview(force=True))
        cp.chk_overlays.toggled.connect(lambda _: self._update_preview(force=True))
        cp.gestures_toggled.connect(self.set_gestures)
        cp.gesture_mode_changed.connect(lambda m: self.hand_worker and self.hand_worker.set_move_mode(m))

        self.controls.object_selected.connect(self.select_object)
        self.controls.delete_requested.connect(self.delete_object)
        self.controls.clear_requested.connect(self.clear_objects)
        self.controls.changed.connect(self._on_object_changed)
        self.controls.message.connect(self.log)
        self.viz.interaction.object_picked.connect(self.select_object)
        self.viz.interaction.transform_changed.connect(self._on_transform_live)
        self.viz.interaction.transform_finished.connect(self._on_transform_live)

    # ============================================================= logging
    def log(self, text: str) -> None:
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")
        self.chip_status.setText(f"{tr('status')}: {text[:70]}")

    # ============================================================== sources
    def _on_cameras_found(self, indices: list[int]) -> None:
        self.camera_panel.set_cameras(indices)
        if not indices:
            self.log(tr("camera_none") + " — " + tr("no_video"))
            return
        if self._auto_camera and self.frozen_frame is None:
            idx = self._requested_camera if self._requested_camera in indices else indices[0]
            self.camera_panel.camera_combo.setCurrentIndex(indices.index(idx))
            self.start_camera(idx)

    def open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("load_video"), str(Path.cwd() / "samples"),
                                              "Video (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if path:
            self.start_camera(path)

    def start_camera(self, index: int | str) -> None:
        """Start streaming from a camera index or a video file path."""
        self.stop_camera(log=False)
        self.set_frozen(False)
        w = CameraWorker(index)
        w.frame_ready.connect(self._on_frame)
        w.fps_updated.connect(lambda fps: self.chip_fps.setText(f"{tr('fps')}: {fps:.1f}"))
        w.failed.connect(self._camera_failed)
        w.opened.connect(lambda i: (self.log(tr("camera_opened", i)), self.camera_panel.set_camera_running(True)))
        self.camera_worker = w
        w.start()

    def _camera_failed(self, index: int) -> None:
        self.log(tr("camera_failed", index))
        self.camera_panel.set_camera_running(False)
        self.camera_worker = None

    def stop_camera(self, log: bool = True) -> None:
        if self.camera_worker is not None:
            self.camera_worker.frame_ready.disconnect(self._on_frame)
            self.camera_worker.stop()
            self.camera_worker = None
            self.chip_fps.setText(f"{tr('fps')}: —")
            if log:
                self.log(tr("camera_stopped"))
        self.camera_panel.set_camera_running(False)

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("load_image"), str(Path.cwd() / "samples"),
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> bool:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  # unicode-safe on Windows
        if img is None:
            self.log(tr("image_failed", path))
            return False
        self._use_static_image(img)
        self.log(tr("image_loaded", Path(path).name))
        return True

    def load_sample(self, name: str) -> None:
        img, _ = SCENES[name]()
        self._use_static_image(img)
        self.log(tr("image_loaded", name))

    def _use_static_image(self, img: np.ndarray) -> None:
        self.stop_camera(log=False)
        self.latest_frame = img
        self.set_frozen(True, frame=img)

    def set_frozen(self, on: bool, frame: np.ndarray | None = None) -> None:
        self.frozen = on
        if on:
            src = frame if frame is not None else self.latest_frame
            if src is None:
                self.frozen = False
                self.camera_panel.set_frozen(False)
                return
            self.frozen_frame = src.copy()
            self.processor.submit(self.frozen_frame)
            self.log(tr("frozen"))
        elif self.frozen_frame is not None:
            self.frozen_frame = None
            self.log(tr("live"))
        self.camera_panel.set_frozen(self.frozen)
        self._update_preview(force=True)

    def current_frame(self) -> np.ndarray | None:
        return self.frozen_frame if self.frozen else self.latest_frame

    # ============================================================ pipeline
    def _on_frame(self, frame: np.ndarray) -> None:
        if self.frozen:
            return
        self.latest_frame = frame
        self.processor.submit(frame)
        if self.hand_worker is not None:
            self.hand_worker.submit(frame)
        self._update_preview()

    def _on_params(self, params) -> None:
        self.processor.set_params(params)
        if self.frozen and self.frozen_frame is not None:
            self.processor.submit(self.frozen_frame)

    def _set_roi(self, roi) -> None:
        self.camera_panel.video.roi = roi
        self.processor.set_roi(roi)
        self.log(tr("roi_set"))
        if self.frozen and self.frozen_frame is not None:
            self.processor.submit(self.frozen_frame)

    def _clear_roi(self) -> None:
        self.camera_panel.video.roi = None
        self.processor.set_roi(None)
        if self.frozen and self.frozen_frame is not None:
            self.processor.submit(self.frozen_frame)
        self._update_preview(force=True)

    def _on_result(self, result: DetectionResult) -> None:
        self.latest_result = result
        self._retrack_selection(result)
        self.camera_panel.set_detections(result.objects, self.selected_det_id)
        n = len(result.objects)
        parts = [tr("objects_found", n) if n else tr("no_objects"), f"{result.processing_ms:.0f} ms"]
        if result.paper_quad is not None:
            parts.append(tr("paper_found"))
        if result.axes is not None:
            parts.append(tr("axes_found"))
        if result.work_image.size and float(result.work_image.mean()) < 45:
            parts.append(tr("too_dark"))
        self.chip_detect.setText(f"{tr('detection')}: " + " · ".join(parts))
        self._update_preview(force=True)

    # ----------------------------------------------------------- selection
    def _retrack_selection(self, result: DetectionResult) -> None:
        if self._sel_point is None:
            self.selected_det_id = 0
            return
        best, best_d = None, float("inf")
        for o in result.objects:
            cf = o.contour_frame
            if cf is None or len(cf) < 3:
                continue
            c = cf.mean(axis=0)
            d = float(np.linalg.norm(c - self._sel_point))
            inside = cv2.pointPolygonTest(cf.astype(np.float32).reshape(-1, 1, 2),
                                          tuple(map(float, self._sel_point)), False) >= 0
            if inside:
                d *= 0.25
            if d < best_d:
                best, best_d = o, d
        frame = self.current_frame()
        limit = 0.12 * math.hypot(*frame.shape[:2]) if frame is not None else 100
        if best is not None and best_d < limit:
            self.selected_det_id = best.id
            self._sel_point = best.contour_frame.mean(axis=0)
        else:
            self.selected_det_id = 0

    def _detection_by_id(self, det_id: int) -> DetectedObject | None:
        if self.latest_result is None:
            return None
        return next((o for o in self.latest_result.objects if o.id == det_id), None)

    def _detection_selected(self, det_id: int) -> None:
        det = self._detection_by_id(det_id)
        self.selected_det_id = det.id if det else 0
        self._sel_point = det.contour_frame.mean(axis=0) if det is not None else None
        self._update_preview(force=True)

    def _video_clicked(self, x: float, y: float) -> None:
        res = self.latest_result
        if res is None:
            return
        use_frame = self.camera_panel.preview_mode == "camera"
        p = (float(x), float(y))
        best, best_score = None, float("inf")
        for o in res.objects:
            pts = o.contour_frame if use_frame else o.contour
            if pts is None or len(pts) < 3:
                continue
            c = pts.astype(np.float32).reshape(-1, 1, 2)
            dist = cv2.pointPolygonTest(c, p, True)          # >0 inside
            area = abs(cv2.contourArea(c)) + 1.0
            score = area if dist >= 0 else (1e12 + abs(dist) if abs(dist) < 30 else float("inf"))
            if score < best_score:
                best, best_score = o, score
        self._detection_selected(best.id if best is not None else 0)
        self.camera_panel.set_detections(res.objects, self.selected_det_id)

    # -------------------------------------------------------------- preview
    def _update_preview(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_preview < 1 / 40:
            return
        self._last_preview = now
        video = self.camera_panel.video
        mode = self.camera_panel.preview_mode
        res = self.latest_result
        video.overlays, video.polylines = [], []
        if mode == "camera" or res is None:
            video.set_image(self.current_frame())
        elif mode == "rectified":
            video.set_image(res.work_image)
        else:
            video.set_image(res.binary)
        if res is None or not self.camera_panel.overlays_enabled:
            video.update()
            return
        in_frame = mode == "camera"
        for o in res.objects:
            pts = o.contour_frame if in_frame else o.contour
            if pts is None:
                continue
            label = f"#{o.id} {kind_name(o.kind)} {o.confidence:.0%}"
            video.overlays.append(Overlay(o.id, pts, label, KIND_COLORS.get(o.kind.value, "#ffffff"),
                                          o.id == self.selected_det_id, closed=o.kind.is_closed or len(pts) > 20))
        if in_frame and res.paper_quad is not None:
            q = res.paper_quad
            video.polylines.append((np.vstack([q, q[:1]]), "#26c6da", True))
        if res.axes is not None:
            for seg in (res.axes.x_segment, res.axes.y_segment):
                video.polylines.append((res.map_to_frame(seg) if in_frame else seg, "#e040fb", True))
        video.banner = tr("frozen") if self.frozen else ""
        video.update()

    # ============================================================ extraction
    def capture_and_extract(self) -> None:
        if self.current_frame() is None:
            self.log(tr("no_video"))
            return
        if not self.frozen:
            self.set_frozen(True)
        res = self.latest_result
        det = self._detection_by_id(self.selected_det_id)
        if det is None and res is not None and len(res.objects) == 1:
            det = res.objects[0]
        if det is None:
            self.log(tr("select_first") if res and res.objects else tr("no_objects"))
            return
        self.extract(det, res)

    def extract_selected(self) -> None:
        det = self._detection_by_id(self.selected_det_id)
        if det is None:
            self.log(tr("select_first"))
            return
        self.extract(det, self.latest_result)

    def extract(self, det: DetectedObject, result: DetectionResult | None) -> MathObject:
        obj = MathObject(det, result.axes if result else None)
        bounds = self.scene.objects_bounds()
        if bounds is not None:
            # place new objects to the right of what is already in the workspace
            obj.transform.position[0] = bounds[1] + 0.5 - float(obj.points[:, 0].min())
        self.scene.add_object(obj)
        self.controls.add_object(obj)
        self.select_object(obj.uid)
        self.scene.reset_camera("front")
        self.log(tr("extracted", obj.title) + (f" — {obj.equation()}" if obj.equation() else ""))
        return obj

    # =========================================================== workspace
    def select_object(self, uid) -> None:
        obj = self.scene.objects.get(uid) if uid else None
        self.scene.set_selected(obj.uid if obj else None)
        if self.controls.obj is not obj:
            self.controls.show_object(obj)

    def delete_object(self, uid: str) -> None:
        self.scene.remove_object(uid)
        self.controls.remove_object(uid)

    def clear_objects(self) -> None:
        self.scene.clear()
        self.controls.clear()

    def _on_object_changed(self, uid: str, what: str) -> None:
        if uid not in self.scene.objects:
            return
        try:
            if what == "transform":
                self.scene.update_transform(uid)
            else:
                self.scene.rebuild(uid)
        except Exception as exc:  # invalid expression, degenerate geometry …
            self.log(tr("expression_error", exc))
        if what == "geometry":
            self.controls.refresh_info()

    def _on_transform_live(self, uid: str) -> None:
        if self.controls.obj is not None and self.controls.obj.uid == uid:
            self.controls.refresh_transform()

    # ============================================================== gestures
    def set_gestures(self, on: bool) -> None:
        if not on:
            if self.hand_worker is not None:
                self.hand_worker.stop()
                self.hand_worker = None
            self.camera_panel.video.hands = []
            self.chip_gesture.setText(tr("gesture_state", "—"))
            self.log(tr("gestures_off"))
            return
        reason = mediapipe_status()
        if reason:
            self.log(tr("gesture_unavailable", reason))
            self.camera_panel.set_gestures_checked(False)
            return
        w = HandTrackerWorker(self.camera_panel.gesture_mode.currentData())
        w.gesture.connect(self._on_gesture)
        w.failed.connect(lambda r: (self.log(tr("gesture_unavailable", r)), self.camera_panel.set_gestures_checked(False)))
        self.hand_worker = w
        w.start()
        self.log(tr("gestures_on"))

    def _on_gesture(self, out: GestureOutput) -> None:
        self.camera_panel.video.hands = out.hands
        self.chip_gesture.setText(tr("gesture_state", out.state))
        inter = self.viz.interaction
        if self.scene.selected is None:
            return
        changed = False
        if out.state == "grabbing":
            if out.rotate != (0.0, 0.0):
                inter.rotate_selected(out.rotate[0], out.rotate[1], deg_per_px=1.0)
                changed = True
            if out.translate != (0.0, 0.0):
                w, h = self.viz.plotter.renderer.GetSize()
                inter.translate_selected(out.translate[0] * w, out.translate[1] * h)
                changed = True
        elif out.state == "scaling" and out.scale != 1.0:
            inter.scale_selected(out.scale)
            changed = True
        if changed:
            self.controls.refresh_transform()

    # ================================================================ files
    def save_screenshot(self, path: str | None = None) -> str | None:
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, tr("save_screenshot"), "mathvision3d.png", "PNG (*.png)")
        if path:
            self.viz.plotter.screenshot(path)
            self.log(tr("saved", path))
        return path

    def export_object(self) -> None:
        uid = self.scene.selected
        if uid is None:
            self.log(tr("select_first"))
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("export_object"), f"{uid}.vtk",
                                              "VTK (*.vtk);;STL (*.stl);;OBJ (*.obj);;PLY (*.ply)")
        if path:
            try:
                self.scene.export_object(uid, path)
                self.log(tr("saved", path))
            except Exception as exc:
                self.log(str(exc))

    # ============================================================== closing
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.hand_worker is not None:
            self.hand_worker.stop()
        self.stop_camera(log=False)
        self.processor.stop()
        self.probe.wait(3000)
        self.viz.close_plotter()
        super().closeEvent(event)
