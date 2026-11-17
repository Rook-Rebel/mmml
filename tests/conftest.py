import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathvision3d.vision.pipeline import VisionPipeline  # noqa: E402


@pytest.fixture
def pipeline():
    return VisionPipeline()
