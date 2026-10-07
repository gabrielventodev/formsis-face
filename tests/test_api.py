import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from conftest import jpeg, read

TOKEN = "test-token"


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine=engine, token=TOKEN))


def post(client, steps, images, frame_steps=None, token=TOKEN):
    frame_steps = frame_steps if frame_steps is not None else [0] * len(images)
    return client.post(
        "/v1/liveness",
        headers={"Authorization": f"Bearer {token}"},
        data={"steps": json.dumps(steps), "frame_steps": json.dumps(frame_steps)},
        files=[("frames", (f"{i}.jpg", jpeg(img), "image/jpeg")) for i, img in enumerate(images)],
    )


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["models"]["detector"] == "yunet-2023mar"


def test_requires_token(client):
    r = post(client, ["center", "left"], [read("real.jpg")], token="wrong")
    assert r.status_code == 401


def test_static_photo_cannot_follow_the_challenge(client):
    img = read("real.jpg")
    r = post(client, ["center", "left"], [img, img, img, img], [0, 0, 1, 1])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "retry"
    assert body["reasons"] == ["challenge_not_completed"]
    assert body["best_frame"] == 0
    assert body["steps"][0]["ok"] is True
    assert body["frames"][0]["real"] > 0.9
    assert body["frames"][2]["real"] is None  # turned step: no passive score


def test_spoof_fails(client):
    img = read("spoof.jpg")
    body = post(client, ["center", "left"], [img, img], [0, 1]).json()
    assert body["decision"] == "fail"
    assert "spoof_suspected" in body["reasons"]


def test_frames_must_share_a_size(client):
    img = read("real.jpg")
    r = post(client, ["center", "left"], [img, cv2.resize(img, (400, 500))], [0, 1])
    assert r.status_code == 400


@pytest.mark.parametrize("steps,frame_steps", [
    (["left", "center"], [0]),          # must start with center
    (["center"], [0]),                  # at least two steps
    (["center", "jump"], [0]),          # unknown step
    (["center", "left"], [0, 0]),       # one entry per frame
    (["center", "left"], [5]),          # index out of range
])
def test_rejects_bad_challenges(client, steps, frame_steps):
    r = post(client, steps, [read("real.jpg")], frame_steps)
    assert r.status_code == 400


def test_rejects_non_images(client):
    r = client.post(
        "/v1/liveness",
        headers={"Authorization": f"Bearer {TOKEN}"},
        data={"steps": '["center","left"]', "frame_steps": "[0]"},
        files=[("frames", ("a.jpg", b"hello", "image/jpeg"))],
    )
    assert r.status_code == 400


def test_no_face(client):
    blank = np.full((480, 640, 3), 128, np.uint8)
    body = post(client, ["center", "left"], [blank, blank], [0, 1]).json()
    assert body["decision"] == "retry"
    assert body["reasons"] == ["no_face"]
