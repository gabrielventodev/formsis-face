import os

import cv2
import pytest

from app.engine import Engine

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "..", "models")


@pytest.fixture(scope="session")
def engine():
    return Engine(MODELS)


def fixture_path(name):
    return os.path.join(HERE, "fixtures", name)


def read(name):
    return cv2.imread(fixture_path(name))


def jpeg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()
