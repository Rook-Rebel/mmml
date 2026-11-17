"""Data structures shared by the recognition pipeline and the rest of the app."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..mathematics.function_fitting import FunctionFitResult


class ShapeKind(str, Enum):
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    QUADRILATERAL = "quadrilateral"
    POLYGON = "polygon"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    CLOSED_CURVE = "closed_curve"
    LINE = "line"
    ARC = "arc"
    FUNCTION = "function"
    FREEHAND = "freehand"

    @property
    def is_polygon(self) -> bool:
        return self in (ShapeKind.TRIANGLE, ShapeKind.RECTANGLE, ShapeKind.SQUARE,
                        ShapeKind.QUADRILATERAL, ShapeKind.POLYGON)

    @property
    def is_conic(self) -> bool:
        return self in (ShapeKind.CIRCLE, ShapeKind.ELLIPSE)

    @property
    def is_closed(self) -> bool:
        return self.is_polygon or self.is_conic or self == ShapeKind.CLOSED_CURVE

    @property
    def is_curve(self) -> bool:
        return self in (ShapeKind.FUNCTION, ShapeKind.FREEHAND, ShapeKind.ARC)


@dataclass
class DetectedObject:
    """One recognised drawing element, in rectified *work image* pixel coordinates."""

    kind: ShapeKind
    contour: np.ndarray            # (N, 2) outline for overlays
    geometry: np.ndarray           # (N, 2) canonical points: vertices / conic samples / ordered curve
    confidence: float
    params: dict = field(default_factory=dict)
    fit: FunctionFitResult | None = None
    id: int = 0
    contour_frame: np.ndarray | None = None   # outline mapped back to the camera frame

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(self.contour.astype(np.float32))
        return int(x), int(y), int(w), int(h)

    @property
    def centroid(self) -> np.ndarray:
        return self.geometry.mean(axis=0)

    @property
    def extent(self) -> float:
        _, _, w, h = self.bbox
        return float(max(w, h))


@dataclass
class AxesInfo:
    origin: np.ndarray             # crossing point (work px)
    x_segment: np.ndarray          # (2, 2)
    y_segment: np.ndarray          # (2, 2)
    thickness: float


@dataclass
class DetectionResult:
    objects: list[DetectedObject]
    work_image: np.ndarray          # BGR image the detection ran on (rectified when paper found)
    binary: np.ndarray              # ink mask (after axis removal)
    work_to_frame: np.ndarray       # 3x3 homography work px -> camera frame px
    paper_quad: np.ndarray | None = None   # in frame coordinates
    axes: AxesInfo | None = None
    stroke_width: float = 3.0
    processing_ms: float = 0.0
    message: str = ""

    def map_to_frame(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if len(p) == 0:
            return p.reshape(-1, 2)
        return cv2.perspectiveTransform(p, self.work_to_frame).reshape(-1, 2)
