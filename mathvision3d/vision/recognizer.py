"""Pluggable recognisers.

The default :class:`ClassicalRecognizer` uses traditional computer vision. A
future learned model (e.g. an ONNX sketch classifier) only has to implement the
:class:`Recognizer` protocol and be registered with :func:`register_recognizer`;
the pipeline, UI and rendering layers stay unchanged.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .models import DetectedObject
from .preprocessing import PreprocessParams
from .shape_detector import ShapeDetector


class Recognizer(Protocol):
    name: str

    def recognize(self, work_bgr: np.ndarray, binary: np.ndarray, stroke_width: float,
                  params: PreprocessParams) -> list[DetectedObject]:
        ...


class ClassicalRecognizer:
    name = "classical"

    def recognize(self, work_bgr, binary, stroke_width, params):
        return ShapeDetector(params).detect(binary, stroke_width)


_REGISTRY: dict[str, Callable[[], Recognizer]] = {"classical": ClassicalRecognizer}


def register_recognizer(name: str, factory: Callable[[], Recognizer]) -> None:
    _REGISTRY[name] = factory


def create_recognizer(name: str = "classical") -> Recognizer:
    return _REGISTRY[name]()


def available_recognizers() -> list[str]:
    return list(_REGISTRY)
