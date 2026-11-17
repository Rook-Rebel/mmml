"""Webcam capture on a background thread."""

from __future__ import annotations

import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal

try:  # failed probes of absent camera indices are expected; keep the console readable
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:  # pragma: no cover - very old OpenCV
    pass


def _backends() -> list[int]:
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    return [cv2.CAP_ANY]


def open_capture(index: int):
    for backend in _backends():
        cap = cv2.VideoCapture(index, backend)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None


class CameraProbe(QThread):
    """Find available camera indices without blocking the UI."""

    found = Signal(list)

    def __init__(self, max_index: int = 4):
        super().__init__()
        self.max_index = max_index

    def run(self) -> None:
        found = []
        for i in range(self.max_index):
            cap = open_capture(i)
            if cap is not None:
                ok, _ = cap.read()
                if ok:
                    found.append(i)
                cap.release()
        self.found.emit(found)


class CameraWorker(QThread):
    """Streams frames from a webcam index or, for testing without a camera, a video file
    (played at its native rate and looped)."""

    frame_ready = Signal(object)      # BGR np.ndarray
    fps_updated = Signal(float)
    failed = Signal(object)           # index or path
    opened = Signal(object)

    def __init__(self, source: int | str, width: int = 1280, height: int = 720):
        super().__init__()
        self.source = source
        self.index = source
        self.size = (width, height)
        self._running = False

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        if self.is_file:
            cap = cv2.VideoCapture(self.source)
            cap = cap if cap.isOpened() else None
        else:
            cap = open_capture(self.source)
        if cap is None:
            self.failed.emit(self.source)
            return
        frame_period = 0.0
        if self.is_file:
            frame_period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self._running = True
        self.opened.emit(self.source)
        fps, last, failures, last_emit = 0.0, time.perf_counter(), 0, 0.0
        while self._running:
            t_read = time.perf_counter()
            ok, frame = cap.read()
            if (not ok or frame is None) and self.is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)      # loop the clip
                ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 50:
                    self.failed.emit(self.index)
                    break
                time.sleep(0.02)
                continue
            failures = 0
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            self.frame_ready.emit(frame)
            if now - last_emit > 0.5:
                self.fps_updated.emit(fps)
                last_emit = now
            if frame_period:
                spare = frame_period - (time.perf_counter() - t_read)
                if spare > 0:
                    time.sleep(spare)
        cap.release()
