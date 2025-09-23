import sys
import textwrap
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from tasks import button_press_task as bp


class DummyArm:
    def __init__(self):
        self.moves: list[list[float]] = []
        self.open_calls = 0

    def target2cam_xyzrpy_to_target2base_xyzrpy(self, xyzrpy_cam):
        return list(xyzrpy_cam)

    def move_p(self, pose):
        self.moves.append(list(pose))

    def open_gripper(self):
        self.open_calls += 1


def test_press_button_detection_failure(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            grasp_orientation:
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
            press_distance: 0.02
            press_duration: 0.0
            """
        )
    )

    class GuardArm:
        def target2cam_xyzrpy_to_target2base_xyzrpy(self, _):  # pragma: no cover
            raise AssertionError("Arm should not be used when detection fails")

        def move_p(self, _):  # pragma: no cover
            raise AssertionError("Arm should not move when detection fails")

        def open_gripper(self):  # pragma: no cover
            raise AssertionError("Arm should not open gripper when detection fails")

    guard_arm = GuardArm()

    monkeypatch.setattr(bp, "get_button_coords_manual", lambda _cam: (None, None, None))

    result = bp.press_button(
        cfg_path=str(cfg_path),
        use_model=False,
        cam=object(),
        arm=guard_arm,
    )

    assert result == "error"


def test_press_button_success(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            grasp_orientation:
              roll: 1.0
              pitch: 0.0
              yaw: -1.0
            approach_offset: 0.1
            press_distance: 0.05
            press_duration: 0.0
            retreat_offset: 0.07
            """
        )
    )

    arm = DummyArm()

    monkeypatch.setattr(bp, "get_button_coords_manual", lambda _cam: (0.1, 0.2, 0.3))
    monkeypatch.setattr(bp.time, "sleep", lambda _t: None)

    result = bp.press_button(
        cfg_path=str(cfg_path),
        use_model=False,
        cam=object(),
        arm=arm,
    )

    assert result == "success"
    assert arm.open_calls == 1
    expected_moves = [
        [0.1, 0.30000000000000004, 0.3, 1.0, 0.0, -1.0],
        [0.1, 0.2, 0.3, 1.0, 0.0, -1.0],
        [0.1, 0.15000000000000002, 0.3, 1.0, 0.0, -1.0],
        [0.1, 0.2, 0.3, 1.0, 0.0, -1.0],
        [0.1, 0.27, 0.3, 1.0, 0.0, -1.0],
    ]
    assert arm.moves == expected_moves
