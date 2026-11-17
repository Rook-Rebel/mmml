"""Application entry point.

    python -m mathvision3d                      # webcam (first camera found)
    python -m mathvision3d --image samples/sine.png
    python -m mathvision3d --sample triangle --no-camera
    python -m mathvision3d --smoke-test out.png  # automated end-to-end check, then exit
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="mathvision3d", description="MathVision 3D")
    p.add_argument("--image", help="start with an image file instead of the webcam")
    p.add_argument("--sample", help="start with a built-in synthetic sample")
    p.add_argument("--video", help="stream a video file as if it were the webcam")
    p.add_argument("--camera", type=int, default=None, help="camera index to open")
    p.add_argument("--no-camera", action="store_true", help="do not open a webcam automatically")
    p.add_argument("--smoke-test", metavar="PNG", help="run the end-to-end workflow, save a screenshot, exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("MathVision 3D")
    from .interface.main_window import MainWindow
    from .interface.style import STYLESHEET

    app.setStyleSheet(STYLESHEET)
    auto_camera = not (args.no_camera or args.image or args.sample or args.video or args.smoke_test)
    win = MainWindow(camera_index=args.camera, auto_camera=auto_camera)
    win.show()
    if args.video:
        win.start_camera(args.video)
    elif args.image:
        win.load_image(args.image)
    elif args.sample:
        win.load_sample(args.sample)
    if args.smoke_test:
        from .smoke_test import SmokeTest

        win.smoke_test = SmokeTest(win, args.smoke_test)   # keep a reference alive
        win.smoke_test.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
