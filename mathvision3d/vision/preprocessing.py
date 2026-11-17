"""Image preprocessing: illumination normalisation, contrast, denoising, binarisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    threshold_mode: str = "adaptive"     # adaptive | otsu | manual
    manual_threshold: int = 150          # ink is darker than this (manual mode)
    adaptive_block: int = 41             # neighbourhood size for adaptive threshold (odd)
    adaptive_c: int = 14                 # higher -> only darker ink is kept
    contrast: float = 1.6                # ink/paper contrast stretch (1 = none)
    noise_reduction: int = 2             # 0..5 blur + speck removal strength
    edge_sensitivity: float = 0.5        # 0..1; higher -> more polygon corners are kept
    min_object_size: float = 0.06        # min object bounding-box diagonal / image diagonal
    remove_shadows: bool = True
    detect_paper: bool = True
    detect_axes: bool = True
    ignore_border_objects: bool = True
    working_size: int = 960              # long side of the image used for detection

    def copy(self) -> "PreprocessParams":
        return PreprocessParams(**asdict(self))

    @property
    def poly_epsilon(self) -> float:
        """approxPolyDP tolerance as a fraction of the perimeter."""
        return float(np.interp(self.edge_sensitivity, [0.0, 1.0], [0.045, 0.012]))


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by a smooth background estimate: removes shadows and uneven lighting."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
    k = max(7, (min(small.shape) // 10) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def enhance(bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    if params.remove_shadows:
        gray = normalize_illumination(gray)
    if params.contrast != 1.0:
        g = gray.astype(np.float32)
        gray = np.clip(255.0 - (255.0 - g) * params.contrast, 0, 255).astype(np.uint8)
    n = int(params.noise_reduction)
    if n > 0:
        gray = cv2.GaussianBlur(gray, (2 * n + 1, 2 * n + 1), 0)
    return gray


def binarize(bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Return a uint8 mask where ink = 255."""
    gray = enhance(bgr, params)
    if gray.size == 0 or float(gray.std()) < 2.0:
        return np.zeros(gray.shape, np.uint8)       # blank / uniform image
    mode = params.threshold_mode
    if mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif mode == "manual":
        _, binary = cv2.threshold(gray, int(params.manual_threshold), 255, cv2.THRESH_BINARY_INV)
    else:
        block = max(3, int(params.adaptive_block) | 1)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block, int(params.adaptive_c))
        binary[gray > 248] = 0                     # bright paper texture is never ink
    n = int(params.noise_reduction)
    if n >= 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return remove_small_components(binary, min_area=12 + 10 * n)


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def estimate_stroke_width(binary: np.ndarray) -> float:
    """Thin-stroke approximation: area ~ L*t and perimeter ~ 2L  =>  t ~ 2*area/perimeter."""
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    area = float(np.count_nonzero(binary))
    if perimeter <= 0:
        return 3.0
    return float(np.clip(2.0 * area / perimeter, 1.0, 40.0))
