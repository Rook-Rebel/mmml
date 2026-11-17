"""Download MediaPipe's hand landmark model (needed only for the optional gesture control).

    python tools/download_hand_model.py

Source: Google's official MediaPipe model bucket (Apache 2.0). ~7.8 MB.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")
TARGET = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def main() -> int:
    if TARGET.exists():
        print(f"already present: {TARGET}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    tmp = TARGET.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(TARGET)
    print(f"saved {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
