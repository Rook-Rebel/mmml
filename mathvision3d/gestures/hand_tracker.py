"""Hand tracking and a state-based gesture interpreter.

Gestures
--------
* pinch (thumb + index tip together, held for a few frames) -> grab the selected object
* moving the pinching hand                                  -> rotate or translate
* two pinching hands, changing distance                     -> scale
* open hand / hand lost                                     -> release

Accidental motion is suppressed with hysteresis on the pinch ratio, a minimum
hold time before grabbing, exponential smoothing and a dead zone.

The interpreter is pure Python (unit-testable). :class:`HandTrackerWorker` wraps
MediaPipe's Tasks API and needs the model file ``models/hand_landmarker.task``
(run ``python tools/download_hand_model.py`` once).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)


@dataclass
class HandObservation:
    landmarks: np.ndarray       # (21, 3) normalised image coordinates
    score: float = 1.0


@dataclass
class GestureOutput:
    state: str = "idle"                     # idle | armed | grabbing | scaling
    rotate: tuple[float, float] = (0.0, 0.0)  # degrees (yaw, pitch)
    translate: tuple[float, float] = (0.0, 0.0)  # normalised screen fraction
    scale: float = 1.0
    hands: list[np.ndarray] = field(default_factory=list)


def pinch_ratio(lm: np.ndarray) -> float:
    size = np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2]) + 1e-6
    return float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[INDEX_TIP, :2]) / size)


def is_open_hand(lm: np.ndarray) -> bool:
    wrist = lm[WRIST, :2]
    ext = [np.linalg.norm(lm[t, :2] - wrist) > 1.15 * np.linalg.norm(lm[p, :2] - wrist)
           for t, p in zip(FINGER_TIPS, FINGER_PIPS)]
    return all(ext) and pinch_ratio(lm) > 0.6


def pinch_point(lm: np.ndarray) -> np.ndarray:
    return (lm[THUMB_TIP, :2] + lm[INDEX_TIP, :2]) / 2.0


class GestureInterpreter:
    PINCH_ON = 0.32
    PINCH_OFF = 0.48
    HOLD_FRAMES = 3
    LOST_FRAMES = 6
    MIN_SCORE = 0.6
    DEAD_ZONE = 0.0035
    SMOOTHING = 0.45            # EMA weight of the newest sample
    ROTATE_GAIN = 260.0         # degrees per full image width
    MAX_SCALE_STEP = 0.08

    def __init__(self, move_mode: str = "rotate"):
        self.move_mode = move_mode
        self.reset()

    def reset(self) -> None:
        self.state = "idle"
        self._hold = 0
        self._lost = 0
        self._point: np.ndarray | None = None
        self._dist: float | None = None
        self._pinching: dict[int, bool] = {}

    def _is_pinching(self, i: int, lm: np.ndarray) -> bool:
        r = pinch_ratio(lm)
        prev = self._pinching.get(i, False)
        now = r < (self.PINCH_OFF if prev else self.PINCH_ON)   # hysteresis
        self._pinching[i] = now
        return now

    def update(self, hands: list[HandObservation]) -> GestureOutput:
        hands = [h for h in hands if h.score >= self.MIN_SCORE]
        out = GestureOutput(hands=[h.landmarks for h in hands])
        if not hands:
            self._lost += 1
            if self._lost >= self.LOST_FRAMES:
                self.reset()
            out.state = self.state
            return out
        self._lost = 0
        pinching = [h for i, h in enumerate(hands) if self._is_pinching(i, h.landmarks)]

        if any(is_open_hand(h.landmarks) for h in hands) and not pinching:
            self.reset()
            out.state = self.state
            return out

        if len(pinching) >= 2:
            p0, p1 = pinch_point(pinching[0].landmarks), pinch_point(pinching[1].landmarks)
            d = float(np.linalg.norm(p0 - p1))
            if self.state == "scaling" and self._dist:
                ratio = d / self._dist
                if abs(ratio - 1.0) > 0.01:
                    out.scale = float(np.clip(ratio, 1 - self.MAX_SCALE_STEP, 1 + self.MAX_SCALE_STEP))
                    self._dist = d
            else:
                self._dist = d
            self.state = "scaling"
            self._point = None
            out.state = self.state
            return out

        if len(pinching) == 1:
            p = pinch_point(pinching[0].landmarks)
            if self.state in ("idle", "armed", "scaling"):
                self._hold += 1
                self.state = "armed"
                if self._hold >= self.HOLD_FRAMES:
                    self.state = "grabbing"
                    self._point = p
                out.state = self.state
                return out
            smoothed = p if self._point is None else (1 - self.SMOOTHING) * self._point + self.SMOOTHING * p
            delta = smoothed - self._point if self._point is not None else np.zeros(2)
            self._point = smoothed
            if np.linalg.norm(delta) > self.DEAD_ZONE:
                # the camera image is not mirrored: moving the hand to the user's right
                # moves it left in the image, so x is negated to feel like a mirror
                dx, dy = -float(delta[0]), -float(delta[1])
                if self.move_mode == "translate":
                    out.translate = (dx, dy)
                else:
                    out.rotate = (dx * self.ROTATE_GAIN, dy * self.ROTATE_GAIN)
            out.state = self.state
            return out

        # hands visible but none pinching -> release
        if self.state != "idle":
            self.reset()
        out.state = self.state
        return out


def mediapipe_status() -> str | None:
    """None when gestures can run, otherwise a human-readable reason."""
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"mediapipe import failed ({exc})"
    if not MODEL_PATH.exists():
        return f"model file missing: {MODEL_PATH} (run tools/download_hand_model.py)"
    return None


class HandTrackerWorker(QThread):
    gesture = Signal(object)       # GestureOutput
    failed = Signal(str)

    def __init__(self, move_mode: str = "rotate", max_rate_hz: float = 20.0):
        super().__init__()
        self.interpreter = GestureInterpreter(move_mode)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame = None
        self._running = False
        self._min_interval = 1.0 / max_rate_hz

    def set_move_mode(self, mode: str) -> None:
        self.interpreter.move_mode = mode

    def submit(self, frame) -> None:
        with self._lock:
            self._frame = frame
        self._event.set()

    def stop(self) -> None:
        self._running = False
        self._event.set()
        self.wait(3000)

    def run(self) -> None:
        reason = mediapipe_status()
        if reason:
            self.failed.emit(reason)
            return
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        try:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5)
            landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.failed.emit(str(exc))
            return
        self._running = True
        t_start = time.perf_counter()
        last_ts = -1
        with landmarker:
            while self._running:
                if not self._event.wait(0.2):
                    continue
                self._event.clear()
                with self._lock:
                    frame, self._frame = self._frame, None
                if frame is None:
                    continue
                t0 = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = max(int((t0 - t_start) * 1000), last_ts + 1)
                last_ts = ts
                try:
                    res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                except Exception as exc:  # pragma: no cover
                    self.failed.emit(str(exc))
                    break
                hands = []
                for i, lms in enumerate(res.hand_landmarks or []):
                    score = res.handedness[i][0].score if res.handedness and res.handedness[i] else 1.0
                    hands.append(HandObservation(np.array([[p.x, p.y, p.z] for p in lms], float), score))
                self.gesture.emit(self.interpreter.update(hands))
                spare = self._min_interval - (time.perf_counter() - t0)
                if spare > 0:
                    time.sleep(spare)
