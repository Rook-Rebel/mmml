"""End-to-end recognition: frame -> (paper rectification) -> binary -> objects."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .curve_extractor import detect_axes, remove_axes
from .models import DetectionResult
from .perspective_corrector import find_paper_quad, rectify
from .preprocessing import PreprocessParams, binarize, estimate_stroke_width
from .recognizer import Recognizer, create_recognizer


def _scale_mat(s: float) -> np.ndarray:
    return np.diag([s, s, 1.0])


def _translate_mat(tx: float, ty: float) -> np.ndarray:
    m = np.eye(3)
    m[0, 2], m[1, 2] = tx, ty
    return m


class VisionPipeline:
    def __init__(self, params: PreprocessParams | None = None, recognizer: Recognizer | None = None):
        self.params = params or PreprocessParams()
        self.recognizer = recognizer or create_recognizer()

    def process(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> DetectionResult:
        t0 = time.perf_counter()
        p = self.params
        if frame is None or frame.size == 0:
            empty = np.zeros((10, 10, 3), np.uint8)
            return DetectionResult([], empty, np.zeros((10, 10), np.uint8), np.eye(3), message="empty")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # work -> frame homography is composed step by step
        to_frame = np.eye(3)
        img = frame
        if roi is not None:
            x, y, w, h = (int(v) for v in roi)
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            if w > 8 and h > 8:
                img = frame[y:y + h, x:x + w]
                to_frame = to_frame @ _translate_mat(x, y)

        s = min(1.0, p.working_size / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        to_frame = to_frame @ _scale_mat(1.0 / s)

        paper_quad = None
        message = ""
        if p.detect_paper and roi is None:
            quad = find_paper_quad(img)
            if quad is not None:
                warped, h_mat = rectify(img, quad, p.working_size)
                # crop a small margin so the paper edge itself is not detected as ink
                m = int(0.02 * max(warped.shape[:2]))
                warped = warped[m:-m, m:-m] if min(warped.shape[:2]) > 4 * m + 10 else warped
                to_frame = to_frame @ np.linalg.inv(h_mat) @ _translate_mat(m, m)
                paper_quad = (quad / s).astype(np.float32)
                img = warped
                message = "paper"

        binary = binarize(img, p)
        stroke = estimate_stroke_width(binary)
        axes = None
        obj_mask = binary
        if p.detect_axes:
            axes = detect_axes(binary, stroke)
            if axes is not None:
                obj_mask = remove_axes(binary, axes)
        objects = self.recognizer.recognize(img, obj_mask, stroke, p)
        result = DetectionResult(objects, img, obj_mask, to_frame, paper_quad, axes, stroke,
                                 message=message)
        for o in objects:
            o.contour_frame = result.map_to_frame(o.contour)
        result.processing_ms = (time.perf_counter() - t0) * 1000.0
        return result
