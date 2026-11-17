"""Runs the vision pipeline on the most recent frame in a worker thread.

Frames arriving faster than they can be processed are dropped (only the latest
frame is kept), so the UI never queues up stale work.
"""

from __future__ import annotations

import threading
import time
import traceback

from PySide6.QtCore import QThread, Signal

from ..vision.pipeline import VisionPipeline
from ..vision.preprocessing import PreprocessParams


class ProcessingWorker(QThread):
    result_ready = Signal(object)     # DetectionResult
    error = Signal(str)

    def __init__(self, max_rate_hz: float = 12.0):
        super().__init__()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame = None
        self._params = PreprocessParams()
        self._roi = None
        self._running = False
        self._min_interval = 1.0 / max_rate_hz

    def submit(self, frame) -> None:
        with self._lock:
            self._frame = frame
        self._event.set()

    def set_params(self, params: PreprocessParams) -> None:
        with self._lock:
            self._params = params.copy()

    def set_roi(self, roi) -> None:
        with self._lock:
            self._roi = roi

    def stop(self) -> None:
        self._running = False
        self._event.set()
        self.wait(3000)

    def run(self) -> None:
        self._running = True
        pipeline = VisionPipeline()
        while self._running:
            if not self._event.wait(0.2):
                continue
            self._event.clear()
            with self._lock:
                frame, params, roi = self._frame, self._params, self._roi
                self._frame = None
            if frame is None:
                continue
            t0 = time.perf_counter()
            pipeline.params = params
            try:
                result = pipeline.process(frame, roi)
            except Exception:  # keep the worker alive on unexpected input
                self.error.emit(traceback.format_exc(limit=3))
                continue
            self.result_ready.emit(result)
            spare = self._min_interval - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)
