"""Runs the models: face detection (YuNet), passive anti-spoofing (MiniFASNet) and face
embeddings (SFace), all through OpenCV, on CPU, with no network access."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .liveness import FrameObs

DETECTOR = "face_detection_yunet_2023mar.onnx"
RECOGNIZER = "face_recognition_sface_2021dec.onnx"
# MiniFASNet models from Silent-Face-Anti-Spoofing, exported to ONNX with a softmax head.
# Each one looks at a crop of the face scaled by its factor; the scores are averaged.
# Output classes: 0 and 2 are attacks (print, screen), 1 is a real face.
ANTISPOOF = (("2.7_80x80_MiniFASNetV2.onnx", 2.7), ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0))

VERSIONS = {
    "detector": "yunet-2023mar",
    "antispoof": "minifasnet-v2-2.7+v1se-4.0",
    "recognizer": "sface-2021dec",
}

MIN_SIDE, MAX_SIDE = 240, 1920
# Another face counts as a second person when it is at least this fraction of the main face.
SECOND_FACE_RATIO = 0.4


class FrameError(ValueError):
    """A frame that isn't a usable image (corrupt, wrong size)."""


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h
    landmarks: np.ndarray  # 5x2: right eye, left eye, nose tip, right mouth corner, left mouth corner
    score: float
    raw: np.ndarray  # YuNet row, needed by SFace's alignCrop


class Engine:
    def __init__(self, models_dir: str, det_threshold: float = 0.8):
        def path(name: str) -> str:
            p = os.path.join(models_dir, name)
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            return p

        self._lock = threading.Lock()  # OpenCV DNN nets are not safe to share between threads
        self._det = cv2.FaceDetectorYN.create(path(DETECTOR), "", (320, 320), det_threshold, 0.3, 50)
        self._rec = cv2.FaceRecognizerSF.create(path(RECOGNIZER), "")
        self._spoof = [(cv2.dnn.readNetFromONNX(path(name)), scale) for name, scale in ANTISPOOF]

    # ---- images ----

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FrameError("not an image")
        h, w = img.shape[:2]
        if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
            raise FrameError(f"image size {w}x{h} out of range")
        return img

    # ---- models ----

    def faces(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        with self._lock:
            self._det.setInputSize((w, h))
            _, rows = self._det.detect(img)
        if rows is None:
            return []
        faces = [Face(r[0:4].copy(), r[4:14].reshape(5, 2).copy(), float(r[14]), r.copy()) for r in rows]
        faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return faces

    def real_score(self, img: np.ndarray, face: Face) -> float:
        """Probability that the face is live rather than a print or a screen."""
        total = 0.0
        for net, scale in self._spoof:
            crop = _scaled_crop(img, face.box, scale, 80, 80)
            blob = crop.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0-255, as trained
            with self._lock:
                net.setInput(blob)
                prob = net.forward()[0]
            total += float(prob[1])
        return total / len(self._spoof)

    def embedding(self, img: np.ndarray, face: Face) -> np.ndarray:
        with self._lock:
            aligned = self._rec.alignCrop(img, face.raw)
            return self._rec.feature(aligned).copy()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        with self._lock:
            return float(self._rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))

    # ---- per-frame measurements ----

    def observe(self, img: np.ndarray, step: int, frontal: bool) -> tuple[FrameObs, Face | None]:
        h, w = img.shape[:2]
        faces = self.faces(img)
        if not faces:
            return FrameObs(step=step, faces=0, width=w), None
        main = faces[0]
        others = [f for f in faces[1:] if f.box[2] >= SECOND_FACE_RATIO * main.box[2]]
        grey = _face_grey(img, main.box)
        obs = FrameObs(
            step=step,
            faces=1 + len(others),
            width=w,
            face_width=float(main.box[2]),
            det_score=main.score,
            yaw=yaw(main.landmarks),
            brightness=float(grey.mean()),
            sharpness=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        )
        if frontal:
            obs.real = self.real_score(img, main)
        return obs, main


def yaw(landmarks: np.ndarray) -> float:
    """Sideways offset of the nose tip from the eyes' midpoint, along the line between the eyes
    (so tilting the head sideways doesn't change it), in inter-eye distances.

    Positive when the person turns to their own left: in an unmirrored camera frame their left is
    on the image's right, and the nose tip, which sticks out, moves further than the eyes. Five
    landmarks are noisy, so callers compare frames of the same attempt rather than trust the
    absolute value."""
    right_eye, left_eye, nose = (landmarks[i].astype(np.float64) for i in range(3))
    axis = left_eye - right_eye
    dist = float(np.hypot(*axis))
    if dist < 1:
        return 0.0
    mid = (left_eye + right_eye) / 2
    return float(np.dot(nose - mid, axis / dist)) / dist


def _face_grey(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, w, h = (int(round(v)) for v in box)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img
    return cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)


def _scaled_crop(img: np.ndarray, box: np.ndarray, scale: float, out_w: int, out_h: int) -> np.ndarray:
    """Crop around the face enlarged by scale, shifted to stay inside the image. Same as
    CropImage in Silent-Face-Anti-Spoofing, which the models were trained with."""
    src_h, src_w = img.shape[:2]
    x, y, bw, bh = (float(v) for v in box)
    bw, bh = max(bw, 1.0), max(bh, 1.0)
    scale = min((src_h - 1) / bh, min((src_w - 1) / bw, scale))
    nw, nh = bw * scale, bh * scale
    cx, cy = bw / 2 + x, bh / 2 + y
    x0, y0, x1, y1 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > src_w - 1:
        x0 -= x1 - src_w + 1
        x1 = src_w - 1
    if y1 > src_h - 1:
        y0 -= y1 - src_h + 1
        y1 = src_h - 1
    crop = img[int(y0): int(y1) + 1, int(x0): int(x1) + 1]
    return cv2.resize(crop, (out_w, out_h))
