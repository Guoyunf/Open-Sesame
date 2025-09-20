import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tasks import base_alignment_task as ba


class DummyArm:
    def target2cam_xyzrpy_to_target2base_xyzrpy(self, xyzrpy_cam):
        return list(xyzrpy_cam)


class DummyBase:
    def __init__(self):
        self.commands: list[tuple] = []

    def strafe_T(self, duration, linear_velocity=None, if_p=False):
        self.commands.append(("strafe_T", duration, linear_velocity))

    def move_T(self, duration, linear_velocity=None, if_p=False):
        self.commands.append(("move_T", duration, linear_velocity))


def test_pose_already_in_range(monkeypatch):
    arm = DummyArm()
    base = DummyBase()

    monkeypatch.setattr(ba.time, "sleep", lambda _t: None)

    detections = iter([(-10.0, -100.0, 0.3)])

    def detect(_cam):
        return next(detections)

    status, pose = ba.maintain_base_position(
        cam_base=object(),
        arm=arm,
        base=base,
        detect_fn=detect,
        detection_wait=0.0,
        settle_time=0.0,
        max_iterations=1,
    )

    assert status == "success"
    assert pose == (-10.0, -100.0, 0.3)
    assert base.commands == []


def test_adjusts_x_direction(monkeypatch):
    arm = DummyArm()
    base = DummyBase()

    monkeypatch.setattr(ba.time, "sleep", lambda _t: None)

    detections = iter([(-30.0, -100.0, 0.2), (-20.0, -100.0, 0.2)])

    def detect(_cam):
        return next(detections)

    status, pose = ba.maintain_base_position(
        cam_base=object(),
        arm=arm,
        base=base,
        detect_fn=detect,
        detection_wait=0.0,
        settle_time=0.0,
        max_iterations=2,
        strafe_duration=0.2,
        strafe_velocity=0.15,
    )

    assert status == "success"
    assert pose == (-20.0, -100.0, 0.2)
    assert base.commands == [("strafe_T", 0.2, 0.15)]


def test_adjusts_y_direction_backward(monkeypatch):
    arm = DummyArm()
    base = DummyBase()

    monkeypatch.setattr(ba.time, "sleep", lambda _t: None)

    detections = iter([(-10.0, -80.0, 0.25), (-10.0, -95.0, 0.25)])

    def detect(_cam):
        return next(detections)

    status, pose = ba.maintain_base_position(
        cam_base=object(),
        arm=arm,
        base=base,
        detect_fn=detect,
        detection_wait=0.0,
        settle_time=0.0,
        max_iterations=2,
        forward_duration=0.3,
        forward_velocity=0.12,
    )

    assert status == "success"
    assert pose == (-10.0, -95.0, 0.25)
    assert base.commands == [("move_T", -0.3, 0.12)]


def test_fails_after_max_iterations(monkeypatch):
    arm = DummyArm()
    base = DummyBase()

    monkeypatch.setattr(ba.time, "sleep", lambda _t: None)

    def detect(_cam):
        return -30.0, -120.0, 0.1

    status, pose = ba.maintain_base_position(
        cam_base=object(),
        arm=arm,
        base=base,
        detect_fn=detect,
        detection_wait=0.0,
        settle_time=0.0,
        max_iterations=2,
        strafe_duration=0.1,
        strafe_velocity=0.2,
        forward_duration=0.2,
        forward_velocity=0.15,
    )

    assert status == "failed"
    assert pose == (-30.0, -120.0, 0.1)
    assert base.commands == [
        ("strafe_T", 0.1, 0.2),
        ("move_T", 0.2, 0.15),
        ("strafe_T", 0.1, 0.2),
        ("move_T", 0.2, 0.15),
    ]
