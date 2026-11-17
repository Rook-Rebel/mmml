"""Rigid + uniform-scale 3D transforms for workspace objects."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


@dataclass
class Transform3D:
    """``world = position + scale * rotation @ local``. Rotation pivot is the local origin."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0

    MIN_SCALE = 0.02
    MAX_SCALE = 50.0

    def copy(self) -> "Transform3D":
        return Transform3D(self.position.copy(), self.rotation.copy(), float(self.scale))

    # --- Euler angles (extrinsic X, then Y, then Z), degrees -----------------
    def euler_deg(self) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # gimbal-lock warning is harmless for display
            return Rotation.from_matrix(self.rotation).as_euler("xyz", degrees=True)

    def set_euler_deg(self, angles) -> None:
        self.rotation = Rotation.from_euler("xyz", np.asarray(angles, float), degrees=True).as_matrix()

    # --- incremental operations ----------------------------------------------
    def rotate_world(self, axis, angle_deg: float) -> None:
        axis = _unit(axis)
        if np.linalg.norm(axis) < 1e-12 or angle_deg == 0:
            return
        r = Rotation.from_rotvec(axis * np.radians(angle_deg)).as_matrix()
        self.rotation = r @ self.rotation
        self._orthonormalize()

    def translate(self, delta) -> None:
        self.position = self.position + np.asarray(delta, dtype=float)

    def scale_by(self, factor: float) -> None:
        self.scale = float(np.clip(self.scale * factor, self.MIN_SCALE, self.MAX_SCALE))

    def reset_orientation(self) -> None:
        self.rotation = np.eye(3)

    def _orthonormalize(self) -> None:
        u, _, vt = np.linalg.svd(self.rotation)
        self.rotation = u @ vt

    # --- matrices --------------------------------------------------------------
    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation * self.scale
        m[:3, 3] = self.position
        return m

    def apply(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=float)
        if p.shape[-1] == 2:
            p = np.column_stack([p, np.zeros(len(p))])
        return p @ (self.rotation * self.scale).T + self.position
