import cv2
import numpy as np
import pytest

from app.engine import Engine, FrameError, yaw
from conftest import jpeg, read


def test_real_face_scores_high(engine):
    img = read("real.jpg")
    obs, face = engine.observe(img, 0, frontal=True)
    assert obs.faces == 1
    assert obs.real > 0.9


@pytest.mark.parametrize("name", ["spoof.jpg", "spoof2.jpg"])
def test_photos_of_screens_and_prints_score_low(engine, name):
    obs, _ = engine.observe(read(name), 0, frontal=True)
    assert obs.faces == 1
    assert obs.real < 0.4


def test_same_person_is_similar_to_itself_shifted(engine):
    img = read("real.jpg")
    shifted = np.roll(img, 15, axis=1)
    f1, f2 = engine.faces(img)[0], engine.faces(shifted)[0]
    sim = engine.similarity(engine.embedding(img, f1), engine.embedding(shifted, f2))
    assert sim > 0.8


def test_different_people_are_not_similar(engine):
    a, b = read("real.jpg"), read("spoof2.jpg")
    sim = engine.similarity(engine.embedding(a, engine.faces(a)[0]), engine.embedding(b, engine.faces(b)[0]))
    assert sim < 0.363


def test_frontal_face_reads_as_center(engine):
    obs, _ = engine.observe(read("real.jpg"), 0, frontal=False)
    assert abs(obs.yaw) < 0.2


def test_yaw_geometry():
    # Eyes at x=100 and 200, nose at 160: turned slightly towards the image's right.
    lm = np.array([[100, 100], [200, 100], [160, 150], [120, 200], [180, 200]], dtype=np.float32)
    assert yaw(lm) == pytest.approx(0.1)
    # The same face tilted 90 degrees reads the same.
    tilted = np.array([[100, 100], [100, 200], [50, 160], [0, 120], [0, 180]], dtype=np.float32)
    assert yaw(tilted) == pytest.approx(0.1)
    # Mirrored, the sign flips.
    # Mirrored, the sign flips (the detector still lists the image-left eye first).
    mirrored = lm * [-1, 1]
    mirrored[[0, 1]] = mirrored[[1, 0]]
    assert yaw(mirrored) == pytest.approx(-0.1)


def test_decode_rejects_garbage_and_tiny_images():
    with pytest.raises(FrameError):
        Engine.decode(b"not a jpeg")
    with pytest.raises(FrameError):
        Engine.decode(jpeg(np.zeros((100, 100, 3), np.uint8)))
