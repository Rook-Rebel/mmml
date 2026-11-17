"""Write the synthetic sample images (and a demo webcam clip) to ./samples.

    python tools/generate_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mathvision3d.synthetic import SCENES, make_demo_video  # noqa: E402


def main() -> None:
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    for name, scene in SCENES.items():
        img, _ = scene()
        path = out / f"{name}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", path.relative_to(ROOT))
    video = out / "demo_webcam.mp4"
    make_demo_video(str(video))
    print("wrote", video.relative_to(ROOT))


if __name__ == "__main__":
    main()
