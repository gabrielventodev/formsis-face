"""HTTP API of the face service. It is stateless: it receives frames, answers with numbers and
keeps nothing. Only the FormFlow backend calls it, over the internal Docker network.

Run with: uvicorn app.main:create_app --factory"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .engine import VERSIONS, Engine, FrameError
from .liveness import STEPS, Thresholds, evaluate, needs_passive

log = logging.getLogger("face")

MAX_FRAMES = 20
MAX_FRAMES_PER_STEP = 5
MAX_FRAME_BYTES = 1_500_000


def create_app(engine: Engine | None = None, token: str | None = None) -> FastAPI:
    token = os.environ.get("FACE_TOKEN", "") if token is None else token
    if not token and os.environ.get("FACE_ALLOW_NO_TOKEN") != "true":
        raise RuntimeError("FACE_TOKEN is required (set FACE_ALLOW_NO_TOKEN=true only for local development)")
    if engine is None:
        engine = Engine(os.environ.get("FACE_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models")))
    thresholds = Thresholds()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    app = FastAPI(title="formsis-face", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(authorization: str = Header(default="")) -> None:
        if not token:
            return
        given = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__, "models": VERSIONS}

    @app.post("/v1/liveness", dependencies=[Depends(authorize)])
    def liveness(
        steps: str = Form(..., description='JSON list, e.g. ["center","left","closer"]'),
        frame_steps: str = Form(..., description="JSON list: the step index of each frame, in order"),
        frames: list[UploadFile] = File(...),
    ) -> JSONResponse:
        started = time.monotonic()
        step_list = _json_list(steps, "steps")
        frame_step_list = _json_list(frame_steps, "frame_steps")
        _check_challenge(step_list, frame_step_list, len(frames))

        observations, faces, size = [], [], None
        images = []
        for i, (upload, k) in enumerate(zip(frames, frame_step_list)):
            data = upload.file.read(MAX_FRAME_BYTES + 1)
            if len(data) > MAX_FRAME_BYTES:
                raise HTTPException(status_code=400, detail=f"frame {i} is too large")
            try:
                img = Engine.decode(data)
            except FrameError as e:
                raise HTTPException(status_code=400, detail=f"frame {i}: {e}") from None
            # Every frame comes from the same camera stream, so they share one size.
            if size is None:
                size = img.shape[:2]
            elif img.shape[:2] != size:
                raise HTTPException(status_code=400, detail="frames have different sizes")
            obs, face = engine.observe(img, k, needs_passive(step_list[k]))
            observations.append(obs)
            faces.append(face)
            images.append(img)

        embeddings: dict[int, object] = {}

        def emb(i: int):
            if i not in embeddings:
                embeddings[i] = engine.embedding(images[i], faces[i])
            return embeddings[i]

        result = evaluate(step_list, observations, lambda a, b: engine.similarity(emb(a), emb(b)), thresholds)
        elapsed = int((time.monotonic() - started) * 1000)
        log.info("liveness decision=%s reasons=%s frames=%d ms=%d", result.decision, result.reasons, len(frames), elapsed)
        return JSONResponse({
            "decision": result.decision,
            "reasons": result.reasons,
            "scores": {"passive": _round(result.passive), "consistency": _round(result.consistency)},
            "steps": result.steps,
            "best_frame": result.best_frame,
            "frames": [
                {
                    "index": i,
                    "step": o.step,
                    "faces": o.faces,
                    "face_ratio": _round(o.face_ratio),
                    "yaw": _round(o.yaw),
                    "brightness": _round(o.brightness, 1),
                    "sharpness": _round(o.sharpness, 1),
                    "real": _round(o.real),
                    "similarity": _round(o.similarity),
                    "issues": o.issues,
                }
                for i, o in enumerate(observations)
            ],
            "engine": {**VERSIONS, "service": __version__},
            "elapsed_ms": elapsed,
        })

    return app


def _json_list(raw: str, name: str) -> list:
    try:
        v = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{name} must be a JSON list") from None
    if not isinstance(v, list):
        raise HTTPException(status_code=400, detail=f"{name} must be a JSON list")
    return v


def _check_challenge(steps: list, frame_steps: list, n_frames: int) -> None:
    if not 2 <= len(steps) <= 5 or any(s not in STEPS for s in steps) or steps[0] != "center":
        raise HTTPException(status_code=400, detail=f"steps must start with center and use {list(STEPS)}")
    if not 1 <= n_frames <= MAX_FRAMES or len(frame_steps) != n_frames:
        raise HTTPException(status_code=400, detail=f"send between 1 and {MAX_FRAMES} frames, one frame_steps entry each")
    for k in frame_steps:
        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < len(steps):
            raise HTTPException(status_code=400, detail="frame_steps must be step indexes")
    for k in range(len(steps)):
        if frame_steps.count(k) > MAX_FRAMES_PER_STEP:
            raise HTTPException(status_code=400, detail=f"at most {MAX_FRAMES_PER_STEP} frames per step")


def _round(v: float | None, digits: int = 3) -> float | None:
    return None if v is None else round(float(v), digits)
