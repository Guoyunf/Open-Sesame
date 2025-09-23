"""Unit tests for the elevator button pressing task.

These tests intentionally avoid touching real hardware or the actual
`utils.lib_io`, `camera`, and `arm_kinova` implementations.  The production
modules pull in heavy native dependencies (Realsense SDK, OpenCV, NumPy, etc.)
that are unavailable in the execution environment used for automated testing.

To keep the behaviour under test realistic we provide lightweight stub
modules that mimic the public interfaces required by the task code.  Each test
then exercises the high level logic with deterministic fake detectors and
configuration objects.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Lightweight stub modules for optional hardware / perception dependencies
# ---------------------------------------------------------------------------


if "utils.lib_io" not in sys.modules:
    stub_lib_io = types.ModuleType("utils.lib_io")

    class _Config:
        def __init__(self, data: dict | None = None):
            for key, value in (data or {}).items():
                setattr(self, key, value)

    def _read_yaml_file(_path: str, is_convert_dict_to_class: bool = True):  # noqa: ARG001
        return _Config({})

    stub_lib_io.Config = _Config
    stub_lib_io.read_yaml_file = _read_yaml_file
    sys.modules["utils.lib_io"] = stub_lib_io


if "camera" not in sys.modules:
    stub_camera_module = types.ModuleType("camera")

    class _StubCamera:
        init_args: list[str] = []
        instances: list["_StubCamera"] = []

        def __init__(self) -> None:
            self.cfg_path: str | None = None
            self.disconnect_calls = 0

        @classmethod
        def init_from_yaml(cls, cfg_path: str = "cfg/cfg_cam.yaml") -> "_StubCamera":
            inst = cls()
            inst.cfg_path = cfg_path
            cls.init_args.append(cfg_path)
            cls.instances.append(inst)
            return inst

        def capture_rgbd(self):  # pragma: no cover - kept for interface parity
            return None, None

        def disconnect(self) -> None:
            self.disconnect_calls += 1

    stub_camera_module.Camera = _StubCamera
    sys.modules["camera"] = stub_camera_module


if "arm_kinova" not in sys.modules:
    stub_arm_module = types.ModuleType("arm_kinova")

    class _StubArm:
        init_args: list[str] = []
        instances: list["_StubArm"] = []

        def __init__(self) -> None:
            self.cfg_path: str | None = None
            self.moves: list[list[float]] = []
            self.closed = 0
            self.opened = 0

        @classmethod
        def init_from_yaml(cls, cfg_path: str = "cfg/cfg_arm_left.yaml") -> "_StubArm":
            inst = cls()
            inst.cfg_path = cfg_path
            cls.init_args.append(cfg_path)
            cls.instances.append(inst)
            return inst

        def target2cam_xyzrpy_to_target2base_xyzrpy(self, xyzrpy_cam):
            return list(xyzrpy_cam)

        def move_p(self, pose):
            self.moves.append(list(pose))

        def close_gripper(self):
            self.closed += 1

        def open_gripper(self):
            self.opened += 1

    stub_arm_module.Arm = _StubArm
    sys.modules["arm_kinova"] = stub_arm_module


if "tasks.target_detection" not in sys.modules:
    stub_td_module = types.ModuleType("tasks.target_detection")
    stub_td_module.__package__ = "tasks"

    def _not_implemented(*_args, **_kwargs):  # pragma: no cover - defensive fallback
        raise NotImplementedError("Stub target detection function called")

    stub_td_module.get_target_coords_manual = _not_implemented
    stub_td_module.get_target_coords_model = _not_implemented
    stub_td_module.__all__ = [
        "get_target_coords_manual",
        "get_target_coords_model",
    ]
    sys.modules["tasks.target_detection"] = stub_td_module


from tasks import button_press_task as bp
from tasks import elevator_button_detection as ebd
from tasks import elevator_button_press_task as ebpt


StubCamera = sys.modules["camera"].Camera
StubArm = sys.modules["arm_kinova"].Arm


class DummyArm:
    def __init__(self) -> None:
        self.moves: list[list[float]] = []
        self.open_calls = 0
        self.close_calls = 0

    def target2cam_xyzrpy_to_target2base_xyzrpy(self, xyzrpy_cam):
        return list(xyzrpy_cam)

    def move_p(self, pose):
        self.moves.append(list(pose))

    def close_gripper(self):
        self.close_calls += 1

    def open_gripper(self):
        self.open_calls += 1


def make_cfg(
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    approach_offset: float = 0.0,
    press_distance: float = 0.02,
    press_duration: float = 0.0,
    retreat_offset: float | None = None,
) -> SimpleNamespace:
    orientation = SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw)
    data: dict[str, object] = {
        "grasp_orientation": orientation,
        "approach_offset": approach_offset,
        "press_distance": press_distance,
        "press_duration": press_duration,
    }
    if retreat_offset is not None:
        data["retreat_offset"] = retreat_offset
    return SimpleNamespace(**data)


def patch_config(monkeypatch: pytest.MonkeyPatch, cfg: SimpleNamespace) -> None:
    monkeypatch.setattr(bp, "read_yaml_file", lambda _path: cfg)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bp.time, "sleep", lambda _t: None)


@pytest.fixture(autouse=True)
def _reset_stub_hardware():
    StubCamera.init_args = []
    StubCamera.instances = []
    StubArm.init_args = []
    StubArm.instances = []
    yield
    StubCamera.init_args = []
    StubCamera.instances = []
    StubArm.init_args = []
    StubArm.instances = []


def test_get_elevator_button_coords_manual_forwards_label(monkeypatch):
    captured: dict[str, object] = {}

    def fake_manual(cam, target_name, **kwargs):
        captured["cam"] = cam
        captured["target_name"] = target_name
        captured["kwargs"] = kwargs
        return (1.0, 2.0, 3.0)

    monkeypatch.setattr(ebd, "get_target_coords_manual", fake_manual)

    cam = object()
    result = ebd.get_elevator_button_coords_manual(cam, button_label="2")

    assert result == (1.0, 2.0, 3.0)
    assert captured["cam"] is cam
    assert captured["target_name"] == "elevator button '2'"


def test_get_elevator_button_coords_manual_without_label(monkeypatch):
    captured: dict[str, object] = {}

    def fake_manual(_cam, target_name, **_kwargs):
        captured["target_name"] = target_name
        return (4.0, 5.0, 6.0)

    monkeypatch.setattr(ebd, "get_target_coords_manual", fake_manual)

    result = ebd.get_elevator_button_coords_manual(object())

    assert result == (4.0, 5.0, 6.0)
    assert captured["target_name"] == "elevator button"


def test_get_elevator_button_coords_model_prompts(monkeypatch):
    captured: dict[str, object] = {}

    def fake_model(cam, target_name, **kwargs):
        captured["cam"] = cam
        captured["target_name"] = target_name
        captured["kwargs"] = kwargs
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(ebd, "get_target_coords_model", fake_model)

    result = ebd.get_elevator_button_coords_model(
        cam=object(),
        button_label="3",
        host="http://example",
        sys_prompt="SYS",
        user_prompt_template="Find {label}",
        max_new_tokens=48,
    )

    assert result == (0.0, 0.0, 0.0)
    assert captured["target_name"] == "elevator button '3'"
    kwargs = captured["kwargs"]
    assert kwargs["host"] == "http://example"
    assert kwargs["host_env_var"] == ebd.ELEVATOR_BUTTON_DETECTION_ENV
    assert kwargs["save_dir"] == "elevator_button_images"
    assert kwargs["sys_prompt"] == "SYS"
    assert kwargs["user_prompt"] == "Find 3"
    assert kwargs["max_new_tokens"] == 48


def test_get_elevator_button_coords_model_default_prompt(monkeypatch):
    captured: dict[str, object] = {}

    def fake_model(_cam, *, target_name, **kwargs):
        captured.update(kwargs)
        captured["target_name"] = target_name
        return (1.0, 2.0, 3.0)

    monkeypatch.setattr(ebd, "get_target_coords_model", fake_model)

    ebd.get_elevator_button_coords_model(cam=object(), button_label="Down")

    assert isinstance(captured["user_prompt"], str)
    assert "Down" in captured["user_prompt"]


def test_get_elevator_button_coords_model_env_override(monkeypatch):
    captured: dict[str, object] = {}

    def fake_model(_cam, *, target_name, host=None, host_env_var=None, **kwargs):
        captured["target_name"] = target_name
        captured["host_env_var"] = host_env_var
        captured["raw_host"] = host
        captured["resolved_host"] = host or (
            os.environ.get(host_env_var) if host_env_var else None
        )
        captured["extra_kwargs"] = kwargs
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(ebd, "get_target_coords_model", fake_model)
    monkeypatch.delenv(ebd.ELEVATOR_BUTTON_DETECTION_ENV, raising=False)
    monkeypatch.setenv(ebd.ELEVATOR_BUTTON_DETECTION_ENV, "http://from-env")

    ebd.get_elevator_button_coords_model(cam=object(), button_label="1", host=None)

    assert captured["resolved_host"] == "http://from-env"
    assert captured["raw_host"] is None
    assert captured["host_env_var"] == ebd.ELEVATOR_BUTTON_DETECTION_ENV


def test_press_elevator_button_detection_failure(monkeypatch):
    cfg = make_cfg(press_distance=0.02, press_duration=0.0)
    patch_config(monkeypatch, cfg)

    class GuardArm:
        def target2cam_xyzrpy_to_target2base_xyzrpy(self, _):  # pragma: no cover
            raise AssertionError("Arm should not be used when detection fails")

        def move_p(self, _):  # pragma: no cover
            raise AssertionError("Arm should not move when detection fails")

        def open_gripper(self):  # pragma: no cover
            raise AssertionError("Arm should not open gripper when detection fails")

    guard_arm = GuardArm()

    monkeypatch.setattr(
        ebpt,
        "get_elevator_button_coords_manual",
        lambda _cam, _label: (None, None, None),
    )

    result = ebpt.press_elevator_button(
        "1",
        cfg_path="unused",
        use_model=False,
        cam=object(),
        arm=guard_arm,
    )

    assert result == "error"


def test_press_elevator_button_success(monkeypatch):
    cfg = make_cfg(
        roll=1.0,
        pitch=0.0,
        yaw=-1.0,
        approach_offset=0.1,
        press_distance=0.05,
        press_duration=0.0,
        retreat_offset=0.07,
    )
    patch_config(monkeypatch, cfg)

    arm = DummyArm()

    def fake_manual(cam, button_label):
        assert button_label == "Alarm"
        return (0.1, 0.2, 0.3)

    monkeypatch.setattr(ebpt, "get_elevator_button_coords_manual", fake_manual)

    result = ebpt.press_elevator_button(
        "Alarm",
        cfg_path="unused",
        use_model=False,
        cam=object(),
        arm=arm,
    )

    assert result == "success"
    assert arm.close_calls == 1
    assert arm.open_calls == 0

    target_pose = [0.1, 0.2, 0.3, 1.0, 0.0, -1.0]
    sequence = bp._generate_press_sequence(
        target_pose,
        cfg.approach_offset,
        abs(cfg.press_distance),
        cfg.retreat_offset,
    )

    assert arm.moves == [pose for _label, pose in sequence]


def test_press_elevator_button_use_model(monkeypatch):
    cfg = make_cfg(press_distance=0.0, press_duration=0.0)
    patch_config(monkeypatch, cfg)

    arm = DummyArm()
    called: dict[str, object] = {}

    def fake_model(cam, button_label, **kwargs):
        called["button_label"] = button_label
        called["kwargs"] = kwargs
        return (0.4, 0.5, 0.6)

    monkeypatch.setattr(ebpt, "get_elevator_button_coords_model", fake_model)

    result = ebpt.press_elevator_button(
        "Up",
        cfg_path="unused",
        use_model=True,
        cam=object(),
        arm=arm,
        host="http://detector",
        sys_prompt="SYS",
        user_prompt_template="Locate {label}",
        max_new_tokens=12,
        save_dir="custom_dir",
    )

    assert result == "success"
    assert called["button_label"] == "Up"
    kwargs = called["kwargs"]
    assert kwargs["host"] == "http://detector"
    assert kwargs["save_dir"] == "custom_dir"
    assert kwargs["sys_prompt"] == "SYS"
    assert kwargs["user_prompt_template"] == "Locate {label}"
    assert kwargs["max_new_tokens"] == 12


def test_press_elevator_button_auto_init_and_cleanup(monkeypatch):
    cfg = make_cfg(
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        approach_offset=0.05,
        press_distance=0.02,
        press_duration=0.0,
    )
    patch_config(monkeypatch, cfg)

    captured: dict[str, object] = {}

    def fake_model(camera, button_label, **kwargs):
        captured["camera"] = camera
        captured["button_label"] = button_label
        captured["kwargs"] = kwargs
        return (0.2, 0.25, 0.35)

    monkeypatch.setattr(ebpt, "get_elevator_button_coords_model", fake_model)

    result = ebpt.press_elevator_button(
        "3",
        cfg_path="unused",
        use_model=True,
        cam=None,
        arm=None,
    )

    assert result == "success"

    assert StubCamera.init_args == ["cfg/cfg_cam.yaml"]
    assert StubArm.init_args == ["cfg/cfg_arm_left.yaml"]

    assert captured["camera"] in StubCamera.instances
    assert captured["button_label"] == "3"

    arm_instance = StubArm.instances[0]
    target_pose = [0.2, 0.25, 0.35, 0.0, 0.0, 0.0]
    retreat_cfg = getattr(cfg, "retreat_offset", None)
    retreat_offset = cfg.approach_offset if retreat_cfg is None else retreat_cfg
    sequence = bp._generate_press_sequence(
        target_pose,
        cfg.approach_offset,
        abs(cfg.press_distance),
        retreat_offset,
    )
    assert arm_instance.moves == [pose for _label, pose in sequence]
    assert arm_instance.closed == 1
    assert arm_instance.opened == 1

    camera_instance = StubCamera.instances[0]
    assert camera_instance.disconnect_calls == 1
