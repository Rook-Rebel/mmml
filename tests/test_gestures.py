import numpy as np

from mathvision3d.gestures.hand_tracker import GestureInterpreter, HandObservation, is_open_hand, pinch_ratio


def hand(center=(0.5, 0.5), pinch=False, open_=False, score=0.95):
    """Crude synthetic 21-point hand: wrist below, fingers above."""
    cx, cy = center
    lm = np.zeros((21, 3))
    lm[0] = [cx, cy + 0.15, 0]
    lm[9] = [cx, cy, 0]                       # middle MCP -> hand size 0.15
    for tip, pip, dx in ((8, 6, -0.03), (12, 10, 0.0), (16, 14, 0.03), (20, 18, 0.06)):
        lm[pip] = [cx + dx, cy - 0.05, 0]
        lm[tip] = [cx + dx, cy - (0.12 if open_ else 0.02), 0]
    lm[5] = [cx - 0.03, cy, 0]
    lm[4] = lm[8] + [0.01, 0, 0] if pinch else [cx - 0.12, cy, 0]
    return HandObservation(lm, score)


def test_pinch_and_open_detection():
    assert pinch_ratio(hand(pinch=True).landmarks) < 0.2
    assert is_open_hand(hand(open_=True).landmarks)
    assert not is_open_hand(hand(pinch=True).landmarks)


def test_grab_requires_hold_then_rotates():
    g = GestureInterpreter()
    states = [g.update([hand(pinch=True)]).state for _ in range(3)]
    assert states[-1] == "grabbing" and states[0] == "armed"
    out = None
    for k in range(1, 6):
        out = g.update([hand(center=(0.5 - 0.02 * k, 0.5), pinch=True)])
    assert out.rotate[0] > 0          # hand moved left in the image = user's right -> positive yaw
    assert abs(out.rotate[1]) < 1e-9


def test_small_jitter_is_ignored():
    g = GestureInterpreter()
    for _ in range(4):
        g.update([hand(pinch=True)])
    out = g.update([hand(center=(0.5005, 0.5), pinch=True)])
    assert out.rotate == (0.0, 0.0)


def test_open_hand_releases_and_low_confidence_ignored():
    g = GestureInterpreter()
    for _ in range(4):
        g.update([hand(pinch=True)])
    assert g.update([hand(open_=True)]).state == "idle"
    for _ in range(5):
        out = g.update([hand(pinch=True, score=0.3)])
    assert out.state == "idle"


def test_two_hand_scaling():
    g = GestureInterpreter()
    g.update([hand((0.3, 0.5), pinch=True), hand((0.7, 0.5), pinch=True)])
    out = g.update([hand((0.28, 0.5), pinch=True), hand((0.72, 0.5), pinch=True)])
    assert out.state == "scaling" and out.scale > 1.0
    out = g.update([hand((0.2, 0.5), pinch=True), hand((0.8, 0.5), pinch=True)])
    assert out.scale <= 1.0 + GestureInterpreter.MAX_SCALE_STEP + 1e-9


def test_translate_mode():
    g = GestureInterpreter(move_mode="translate")
    for _ in range(4):
        g.update([hand(pinch=True)])
    out = g.update([hand(center=(0.5, 0.45), pinch=True)])
    assert out.translate[1] > 0 and out.rotate == (0.0, 0.0)
