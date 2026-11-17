"""Mapping between rectified-image pixels and the object's mathematical coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoordinateSystem:
    """Pixel -> local mathematical coordinates (x right, y up).

    When ``calibrated`` is False the unit is an arbitrary normalised unit and every
    measurement must be presented as an estimate.
    """

    origin_px: np.ndarray
    px_per_unit_x: float
    px_per_unit_y: float
    calibrated: bool = False
    unit: str = ""
    origin_from_axes: bool = False

    def to_local(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        x = (p[:, 0] - self.origin_px[0]) / self.px_per_unit_x
        y = -(p[:, 1] - self.origin_px[1]) / self.px_per_unit_y
        return np.column_stack([x, y])

    def to_px(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        return np.column_stack([
            p[:, 0] * self.px_per_unit_x + self.origin_px[0],
            -p[:, 1] * self.px_per_unit_y + self.origin_px[1],
        ])

    def copy(self) -> "CoordinateSystem":
        return CoordinateSystem(np.array(self.origin_px, float), self.px_per_unit_x, self.px_per_unit_y,
                                self.calibrated, self.unit, self.origin_from_axes)

    @staticmethod
    def normalized(origin_px, extent_px: float, target_extent: float = 4.0,
                   origin_from_axes: bool = False) -> "CoordinateSystem":
        """Default system: the object's largest extent spans ``target_extent`` units."""
        s = max(float(extent_px), 1.0) / target_extent
        return CoordinateSystem(np.asarray(origin_px, float), s, s, False, "", origin_from_axes)
