from app.liveness import (
    BLURRY, CHALLENGE_NOT_COMPLETED, FACE_CHANGED, FAIL, MULTIPLE_FACES, NO_FACE, PASS,
    PASSIVE_UNCERTAIN, RETRY, REVIEW, SPOOF_SUSPECTED, TOO_DARK, FrameObs, evaluate,
)

STEPS = ["center", "left", "closer"]


def frame(step, yaw=0.0, face_width=200, real=0.95, faces=1, brightness=120, sharpness=80, score=0.95):
    return FrameObs(step=step, faces=faces, width=640, face_width=face_width, det_score=score, yaw=yaw,
                    brightness=brightness, sharpness=sharpness, real=real)


def good_frames():
    return [
        frame(0, yaw=0.02), frame(0, yaw=0.05, score=0.99),
        frame(1, yaw=0.05, real=None), frame(1, yaw=0.30, real=None),
        frame(2, face_width=260, real=0.9),
    ]


def same(a, b):
    return 0.8


def test_live_person_passes():
    r = evaluate(STEPS, good_frames(), same)
    assert r.decision == PASS, r
    assert r.best_frame == 1  # clearest frontal frame of the first step
    assert [s["ok"] for s in r.steps] == [True, True, True]
    assert r.steps[1]["frame"] == 3


def test_turn_not_done_asks_to_retry():
    frames = good_frames()
    frames[3].yaw = 0.15
    r = evaluate(STEPS, frames, same)
    assert r.decision == RETRY
    assert r.reasons == [CHALLENGE_NOT_COMPLETED]
    assert r.steps[1]["ok"] is False


def test_turn_in_wrong_direction_does_not_count():
    r = evaluate(["center", "right"], [frame(0), frame(1, yaw=0.3, real=None)], same)
    assert r.decision == RETRY
    r = evaluate(["center", "right"], [frame(0), frame(1, yaw=-0.3, real=None)], same)
    assert r.decision == PASS


def test_turns_are_measured_from_the_reference_frame():
    # A face whose landmarks read 0.15 when frontal needs to reach 0.37 to count as a left turn.
    assert evaluate(["center", "left"], [frame(0, yaw=0.15), frame(1, yaw=0.3, real=None)], same).decision == RETRY
    assert evaluate(["center", "left"], [frame(0, yaw=0.15), frame(1, yaw=0.4, real=None)], same).decision == PASS


def test_closer_needs_a_bigger_face():
    frames = good_frames()
    frames[4].face_width = 210
    assert evaluate(STEPS, frames, same).decision == RETRY


def test_different_person_fails():
    def changed(a, b):
        return 0.1 if b == 4 else 0.8
    r = evaluate(STEPS, good_frames(), changed)
    assert r.decision == FAIL
    assert FACE_CHANGED in r.reasons
    assert r.consistency == 0.1


def test_print_or_screen_fails():
    frames = good_frames()
    for f in frames:
        if f.real is not None:
            f.real = 0.1
    r = evaluate(STEPS, frames, same)
    assert r.decision == FAIL
    assert r.reasons == [SPOOF_SUSPECTED]


def test_grey_zone_goes_to_review():
    frames = good_frames()
    for f in frames:
        if f.real is not None:
            f.real = 0.6
    r = evaluate(STEPS, frames, same)
    assert r.decision == REVIEW
    assert r.reasons == [PASSIVE_UNCERTAIN]


def test_no_face_in_first_step():
    r = evaluate(STEPS, [frame(0, faces=0), frame(0, faces=0), frame(1, yaw=0.3)], same)
    assert r.decision == RETRY
    assert r.reasons == [NO_FACE]
    assert r.best_frame is None


def test_dark_frames_tell_the_applicant_why():
    r = evaluate(STEPS, [frame(0, brightness=20), frame(0, brightness=25)], same)
    assert r.reasons == [TOO_DARK]


def test_second_person_makes_frame_unusable():
    frames = good_frames()
    frames[3].faces = 2
    r = evaluate(STEPS, frames, same)
    assert r.decision == RETRY
    assert r.reasons == [MULTIPLE_FACES]


def test_blurry_frames_are_skipped_not_failed():
    frames = good_frames() + [frame(1, yaw=0.4, sharpness=2, real=None)]
    frames[3].yaw = 0.0
    r = evaluate(STEPS, frames, same)
    assert r.decision == RETRY
    assert frames[5].issues == [BLURRY]


def test_turned_frames_do_not_feed_the_passive_score():
    frames = good_frames()
    frames[3].real = 0.0  # a spoof score on a turned frame is ignored
    assert evaluate(STEPS, frames, same).decision == PASS
