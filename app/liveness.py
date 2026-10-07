"""Decides whether a liveness attempt came from a live person in front of the camera.

This module is pure logic over per-frame measurements so it can be tested without models.
The engine (engine.py) turns JPEG frames into FrameObs; evaluate() turns those into a decision.

An attempt is a random challenge chosen by the backend, e.g. ["center", "left", "closer"]. The
browser shows each instruction for a moment and captures a few frames while it is on screen, so
every frame is tagged with the step it was taken during. A live person can follow the challenge;
a printed photo or a replayed video generally can't, and the passive anti-spoofing model looks
for the texture of paper and screens on the frontal frames.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

STEPS = ("center", "left", "right", "closer")

# Decisions, from the applicant's point of view:
#   pass   - live person, nothing to look at.
#   review - completed, but a reviewer should look at the frames (passive score in the grey zone).
#   retry  - we could not tell (no face, too dark, step not followed): try again, nothing suspicious.
#   fail   - looks like a spoof or a different person mid-challenge.
PASS, REVIEW, RETRY, FAIL = "pass", "review", "retry", "fail"

# Issues that make a frame unusable. The applicant can fix all of them.
NO_FACE = "no_face"
MULTIPLE_FACES = "multiple_faces"
FACE_TOO_SMALL = "face_too_small"
TOO_DARK = "too_dark"
TOO_BRIGHT = "too_bright"
BLURRY = "blurry"
# Attempt-level reasons.
CHALLENGE_NOT_COMPLETED = "challenge_not_completed"
SPOOF_SUSPECTED = "spoof_suspected"
PASSIVE_UNCERTAIN = "passive_uncertain"
FACE_CHANGED = "face_changed"


@dataclass(frozen=True)
class Thresholds:
    # Yaw is the nose's sideways offset from the midpoint of the eyes, in inter-eye distances
    # (see engine.yaw). Positive means the person turned to *their* left. Roughly 0.1 is 15
    # degrees and 0.2 is 30 degrees, but landmarks are noisy and biased per face, so a turn is
    # measured against the reference frame of the same attempt.
    center_max_yaw: float = 0.3
    turn_min_delta: float = 0.22
    # "closer" needs the face this much wider than in the reference frame.
    closer_min_ratio: float = 1.18
    # Face width as a fraction of the frame width.
    min_face_ratio: float = 0.15
    # Mean grey level of the face crop (0-255).
    min_brightness: float = 50.0
    max_brightness: float = 215.0
    # Variance of the Laplacian of the 112px grey face crop.
    min_sharpness: float = 20.0
    # Passive anti-spoofing: median probability of "real" across frontal frames.
    passive_pass: float = 0.80
    passive_fail: float = 0.40
    # SFace cosine similarity; 0.363 is the threshold OpenCV recommends for "same person".
    same_person: float = 0.363


@dataclass
class FrameObs:
    step: int  # index into the challenge
    faces: int  # faces big enough to count as another person
    width: int = 0  # frame width in px
    face_width: float = 0.0
    det_score: float = 0.0
    yaw: float = 0.0
    brightness: float = 0.0
    sharpness: float = 0.0
    real: float | None = None  # passive anti-spoofing score, frontal frames only
    issues: list[str] = field(default_factory=list)
    similarity: float | None = None  # to the reference frame, filled in by evaluate()

    @property
    def usable(self) -> bool:
        return not self.issues

    @property
    def face_ratio(self) -> float:
        return self.face_width / self.width if self.width else 0.0


def frame_issues(f: FrameObs, t: Thresholds) -> list[str]:
    """Quality problems that make a frame unusable, most important first."""
    if f.faces == 0:
        return [NO_FACE]
    out = []
    if f.faces > 1:
        out.append(MULTIPLE_FACES)
    if f.face_ratio < t.min_face_ratio:
        out.append(FACE_TOO_SMALL)
    if f.brightness < t.min_brightness:
        out.append(TOO_DARK)
    elif f.brightness > t.max_brightness:
        out.append(TOO_BRIGHT)
    if f.sharpness < t.min_sharpness:
        out.append(BLURRY)
    return out


def needs_passive(step: str) -> bool:
    """The anti-spoofing models were trained on near-frontal faces."""
    return step in ("center", "closer")


@dataclass
class Result:
    decision: str
    reasons: list[str]
    passive: float | None
    consistency: float | None
    steps: list[dict]
    best_frame: int | None


def evaluate(
    steps: list[str],
    frames: list[FrameObs],
    similar: Callable[[int, int], float],
    t: Thresholds = Thresholds(),
) -> Result:
    """Decide on an attempt. similar(i, j) returns the face similarity between frames i and j."""
    for f in frames:
        f.issues = frame_issues(f, t)

    # Reference: the clearest frontal frame of the first step.
    ref = None
    for i, f in enumerate(frames):
        if f.step == 0 and f.usable and abs(f.yaw) <= t.center_max_yaw:
            if ref is None or f.det_score > frames[ref].det_score:
                ref = i
    if ref is None:
        return Result(RETRY, [_main_issue(frames, 0)], None, None, _steps(steps, frames, None, t), None)

    # Every usable frame must show the same person as the reference.
    sims = []
    for i, f in enumerate(frames):
        if f.usable:
            f.similarity = 1.0 if i == ref else similar(ref, i)
            sims.append(f.similarity)
    consistency = min(sims)

    step_results = _steps(steps, frames, ref, t)
    reasons: list[str] = []
    if consistency < t.same_person:
        reasons.append(FACE_CHANGED)

    passive_scores = [f.real for f in frames if f.usable and f.real is not None and needs_passive(steps[f.step])]
    passive = statistics.median(passive_scores) if passive_scores else None
    if passive is not None and passive < t.passive_fail:
        reasons.append(SPOOF_SUSPECTED)

    if reasons:
        return Result(FAIL, reasons, passive, consistency, step_results, ref)

    missing = [s for s in step_results if not s["ok"]]
    if missing:
        return Result(RETRY, [_main_issue(frames, missing[0]["index"])],
                      passive, consistency, step_results, ref)

    if passive is None or passive < t.passive_pass:
        return Result(REVIEW, [PASSIVE_UNCERTAIN], passive, consistency, step_results, ref)
    return Result(PASS, [], passive, consistency, step_results, ref)


def _pose_ok(step: str, f: FrameObs, ref: FrameObs | None, t: Thresholds) -> bool:
    if step == "center":
        return abs(f.yaw) <= t.center_max_yaw
    if step == "left":
        return ref is not None and f.yaw - ref.yaw >= t.turn_min_delta
    if step == "right":
        return ref is not None and ref.yaw - f.yaw >= t.turn_min_delta
    if step == "closer":
        return ref is not None and f.face_width >= t.closer_min_ratio * ref.face_width
    return False


def _steps(steps: list[str], frames: list[FrameObs], ref: int | None, t: Thresholds) -> list[dict]:
    ref_frame = frames[ref] if ref is not None else None
    out = []
    for k, step in enumerate(steps):
        hit = None
        for i, f in enumerate(frames):
            if f.step == k and f.usable and _pose_ok(step, f, ref_frame, t):
                hit = i
                break
        out.append({"index": k, "step": step, "ok": hit is not None, "frame": hit})
    return out


def _main_issue(frames: list[FrameObs], step: int) -> str:
    """What to tell the applicant: the most frequent frame issue in that step, or that the
    frames were fine but the instruction wasn't followed."""
    counts: dict[str, int] = {}
    for f in frames:
        if f.step == step:
            for issue in f.issues[:1]:
                counts[issue] = counts.get(issue, 0) + 1
    if not counts:
        return CHALLENGE_NOT_COMPLETED
    return max(counts, key=lambda k: counts[k])
