"""Automated end-to-end GUI check (``python -m mathvision3d --smoke-test out.png``).

Drives the real window through the same code paths a user triggers:
load image -> background detection -> select by clicking -> capture & extract ->
change display modes -> rotate/scale -> screenshot. Exits with status 0 on success.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QScrollArea

from .core.math_object import DisplayMode


class SmokeTest:
    def __init__(self, win, screenshot: str):
        self.win = win
        self.out = Path(screenshot)
        self.video = Path(__file__).resolve().parents[1] / "samples" / "demo_webcam.mp4"
        self.steps = [
            self.step_video_start, self.step_video_wait, self.step_video_select, self.step_video_track,
            self.step_triangle_load, self.step_wait_detection, self.step_triangle_extract,
            self.step_sine_load, self.step_wait_detection, self.step_sine_extract,
            self.step_interact, self.step_screenshot_curve, self.step_surface, self.step_space_curve,
            self.step_finish,
        ]
        self.i = 0
        self.deadline = 0.0
        self.failures: list[str] = []
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def start(self) -> None:
        QTimer.singleShot(800, self._run)

    def log(self, msg: str) -> None:
        print(f"[smoke] {msg}", flush=True)

    def check(self, cond: bool, msg: str) -> None:
        self.log(("OK   " if cond else "FAIL ") + msg)
        if not cond:
            self.failures.append(msg)

    def _run(self) -> None:
        try:
            done = self.steps[self.i]()
        except Exception:
            self.failures.append(traceback.format_exc())
            self.log(traceback.format_exc())
            self.i = len(self.steps) - 1
            done = True
        if done:
            self.i += 1
        if self.i < len(self.steps):
            QTimer.singleShot(150, self._run)

    # ----------------------------------------------------------------- steps
    def step_video_start(self):
        """Live path: a looping video stands in for the webcam (no camera needed)."""
        if not self.video.exists():
            self.log(f"SKIP live video stage ({self.video.name} missing; run tools/generate_samples.py)")
            self.i = self.steps.index(self.step_triangle_load) - 1
            return True
        self.win.start_camera(str(self.video))
        self.deadline = time.time() + 20
        self._frames = 0
        self.win.camera_worker.frame_ready.connect(self._count_frame)
        return True

    def _count_frame(self, _):
        self._frames += 1

    def step_video_wait(self):
        res = self.win.latest_result
        if res is not None and len(res.objects) >= 3 and self._frames > 20:
            self.check(not self.win.frozen, "live stream running (not frozen)")
            self.check(res.paper_quad is not None, "paper detected in live frames")
            self.log(f"live detections: {[o.kind.value for o in res.objects]}, frames so far {self._frames}")
            return True
        if time.time() > self.deadline:
            self.check(False, f"live detection (frames={self._frames})")
            return True
        return False

    def step_video_select(self):
        self._click_kind("function")
        self._track_until = time.time() + 1.5
        self._frames_at_select = self._frames
        return True

    def step_video_track(self):
        if time.time() < self._track_until:
            return False
        det = self.win._detection_by_id(self.win.selected_det_id)
        moved = self._frames - self._frames_at_select
        self.check(det is not None and det.kind.value == "function",
                   f"selection tracked across {moved} moving frames")
        self.win.capture_and_extract()
        obj = self.win.controls.obj
        self.check(self.win.frozen, "capture froze the frame")
        self.check(obj is not None and obj.is_recognized_function, f"live sine extracted: {obj and obj.equation()}")
        self.win.stop_camera()
        return True

    def step_triangle_load(self):
        self.win.load_sample("multiple_objects")
        self.deadline = time.time() + 15
        return True

    def step_wait_detection(self):
        res = self.win.latest_result
        if res is not None and res.objects and self.win.frozen:
            kinds = [o.kind.value for o in res.objects]
            self.log(f"detected: {kinds} in {res.processing_ms:.0f} ms")
            return True
        if time.time() > self.deadline:
            self.check(False, "detection result arrived")
            return True
        return False

    def _click_kind(self, kind: str):
        res = self.win.latest_result
        det = next((o for o in res.objects if o.kind.value == kind), None)
        self.check(det is not None, f"{kind} detected")
        if det is None:
            return None
        c = det.contour_frame.mean(axis=0) if kind != "function" else det.contour_frame[len(det.contour_frame) // 4]
        # click on the outline itself (works for open curves too)
        p = det.contour_frame[0]
        self.win._video_clicked(float(p[0]), float(p[1]))
        self.check(self.win.selected_det_id == det.id, f"click selects {kind} (#{det.id})")
        return det

    def step_triangle_extract(self):
        det = self._click_kind("triangle")
        if det is None:
            return True
        self.win.capture_and_extract()
        obj = self.win.controls.obj
        self.check(obj is not None and obj.kind.value == "triangle", "triangle extracted into workspace")
        angles = sum(float(m.value.rstrip("°")) for m in obj.measurements() if m.label == "კუთხეების ჯამი")
        self.check(abs(angles - 180) < 0.5, f"triangle angle sum = {angles:.2f}°")
        cp = self.win.controls
        cp.cmb_mode.setCurrentIndex(cp.cmb_mode.findData(DisplayMode.EXTRUDE.value))
        self.check(obj.options.mode == DisplayMode.EXTRUDE, "triangle -> prism (extrude)")
        cp.sp_depth.setValue(1.5)
        self.check(abs(obj.options.extrude_depth - 1.5) < 1e-9, "extrusion depth control")
        cp.cmb_mode.setCurrentIndex(cp.cmb_mode.findData(DisplayMode.ANALYSIS.value))
        cp._rotate_step(1, 30)
        return True

    def step_sine_load(self):
        self.win.load_sample("sine_with_axes")
        self.win.latest_result = None
        self.deadline = time.time() + 15
        return True

    def step_sine_extract(self):
        det = self._click_kind("function")
        if det is None:
            return True
        self.win.capture_and_extract()
        obj = self.win.controls.obj
        eq = obj.equation() if obj else None
        self.check(obj is not None and obj.is_recognized_function and obj.fit.best.family == "sine",
                   f"sine recognised: {eq}")
        b = obj.fit.best.params[1]
        periods = b * (obj.domain()[1] - obj.domain()[0]) / 6.283185
        self.check(1.5 < periods < 2.1, f"periods visible ≈ {periods:.2f}")
        self.check(obj.origin_meaningful, "origin taken from detected axes")
        cp = self.win.controls
        cp.chk_deriv.setChecked(True)
        cp.chk_tangent.setChecked(True)
        cp.sl_tangent.setValue(300)
        self.check(obj.tangent() is not None, "tangent at selected point")
        return True

    def _display_pos(self, world):
        ren = self.win.viz.plotter.renderer
        ren.SetWorldPoint(*world, 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        return int(round(x)), int(round(y))

    def step_interact(self):
        inter = self.win.viz.interaction
        obj = self.win.controls.obj
        # simulated mouse click in the 3D viewport on a point of the curve -> picks the object
        self.win.scene.reset_camera("front")
        self.win.select_object(None)
        world = obj.transform.apply(obj.curve_points()[len(obj.curve_points()) // 3:][:1])[0]
        x, y = self._display_pos(world)
        iren = inter.style.GetInteractor()
        iren.SetEventInformation(x, y, 0, 0)
        inter.style.InvokeEvent("LeftButtonPressEvent")
        inter.style.InvokeEvent("LeftButtonReleaseEvent")
        self.check(self.win.scene.selected == obj.uid, f"viewport click at ({x}, {y}) picks the curve")
        self.win.select_object(obj.uid)
        before = obj.transform.rotation.copy()
        inter.rotate_selected(40, 25)
        inter.scale_selected(1.2)
        inter.translate_selected(30, -10)
        import numpy as np
        self.check(not np.allclose(before, obj.transform.rotation), "mouse-style rotation applied")
        self.check(abs(obj.transform.scale - 1.2) < 1e-6, "scaling applied")
        self.check(abs(self.win.controls.sp_scale.value() - 1.2) < 1e-6, "controls follow interaction")
        obj.transform.reset_orientation()
        self.win.controls.refresh_transform()
        self.win.scene.update_transform(obj.uid)
        self.win.scene.reset_camera("front")
        return True

    def step_screenshot_curve(self):
        QApplication.processEvents()
        p = self.out.with_name(self.out.stem + "_curve" + self.out.suffix)
        self.win.save_screenshot(str(p))
        self.check(p.exists() and p.stat().st_size > 10_000, f"screenshot {p.name}")
        return True

    def step_surface(self):
        cp = self.win.controls
        obj = cp.obj
        cp.cmb_mode.setCurrentIndex(cp.cmb_mode.findData(DisplayMode.SURFACE.value))
        self.check(obj.options.mode == DisplayMode.SURFACE, "sine -> surface z = f(x)")
        n_actors = len(self.win.scene._actors.get(obj.uid, {}))
        self.check(n_actors >= 1, f"surface actors rendered ({n_actors})")
        self.win.scene.reset_camera("iso")
        return True

    def step_space_curve(self):
        cp = self.win.controls
        obj = cp.obj
        # screenshot of the surface first
        QApplication.processEvents()
        self.win.save_screenshot(str(self.out))
        self.check(self.out.exists(), f"screenshot {self.out.name}")
        window_png = self.out.with_name(self.out.stem + "_window" + self.out.suffix)
        self.win.grab().save(str(window_png))
        self.check(window_png.exists(), f"window capture {window_png.name}")
        cp.cmb_mode.setCurrentIndex(cp.cmb_mode.findData(DisplayMode.SPACE_CURVE.value))
        cp.ed_z.setText("0.3*x*cos(x)")
        cp._z_changed()
        self.check(obj.options.z_expression == "0.3*x*cos(x)", "space curve expression applied")
        cp.ed_z.setText("__import__('os')")
        cp._z_changed()
        self.check(obj.options.z_expression == "0.3*x*cos(x)", "unsafe expression rejected")
        return True

    def step_finish(self):
        for name, panel in (("left", self.win.camera_panel), ("right", self.win.controls)):
            body = panel.findChild(QScrollArea).widget()
            vp = panel.findChild(QScrollArea).viewport().width()
            fits = body.minimumSizeHint().width() <= vp
            self.check(fits, f"{name} panel content fits ({body.minimumSizeHint().width()} <= {vp}px)")
            if not fits:
                from PySide6.QtWidgets import QGroupBox
                for g in body.findChildren(QGroupBox):
                    self.log(f"    {g.title()}: {g.minimumSizeHint().width()}px")
        n = len(self.win.scene.objects)
        expected = 3 if self.video.exists() else 2
        self.check(n == expected, f"workspace holds {n} objects")
        self.log("RESULT: " + ("PASS" if not self.failures else f"FAIL ({len(self.failures)})"))
        code = 0 if not self.failures else 1
        QTimer.singleShot(200, lambda: (self.win.close(), QApplication.instance().exit(code)))
        return True
