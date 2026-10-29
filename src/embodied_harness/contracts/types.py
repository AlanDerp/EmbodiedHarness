"""Validated value objects. Distances are metres; quaternions use WXYZ."""

from dataclasses import dataclass
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar

import numpy as np


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _number(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _array(value, shape, name, dtype=np.float64, finite=True):
    a = np.asarray(value)
    if a.dtype.kind not in "biuf" or a.ndim != len(shape):
        raise ValueError(f"{name} must be a numeric array with shape {shape}")
    if any(n is not None and n != actual for n, actual in zip(shape, a.shape)):
        raise ValueError(f"{name} must have shape {shape}")
    a = np.asarray(a, dtype=dtype)
    if finite and not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    # Immutable bytes backing also prevents callers re-enabling numpy write access.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _mapping(value, cls, name):
    result = dict(value)
    for key, item in result.items():
        _text(key, name)
        if not isinstance(item, cls):
            raise TypeError(f"{name} values must be {cls.__name__}")
    return MappingProxyType(result)


@dataclass(frozen=True, eq=False)
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    frame: str

    def __post_init__(self):
        _text(self.frame, "frame")
        object.__setattr__(self, "position", _array(self.position, (3,), "position"))
        q = _array(self.quaternion_wxyz, (4,), "quaternion_wxyz")
        if not np.isclose(np.linalg.norm(q), 1, atol=1e-6, rtol=0):
            raise ValueError("quaternion must have unit norm")
        object.__setattr__(self, "quaternion_wxyz", q)


@dataclass(frozen=True)
class ObservationSource:
    episode_id: str
    snapshot_id: str
    reset_epoch: int
    state_version: int
    native_action_count: int
    captured_monotonic_s: float
    sim_time_s: float | None = None

    def __post_init__(self):
        for name in ("episode_id", "snapshot_id"):
            _text(getattr(self, name), name)
        for name in ("reset_epoch", "state_version", "native_action_count"):
            _count(getattr(self, name), name)
        _number(self.captured_monotonic_s, "captured_monotonic_s")
        if self.sim_time_s is not None:
            _number(self.sim_time_s, "sim_time_s")

    def same_state(self, other: "ObservationSource") -> bool:
        """A new snapshot ID alone does not establish physical progress."""
        return (self.episode_id, self.reset_epoch, self.state_version,
                self.native_action_count) == (other.episode_id, other.reset_epoch,
                                              other.state_version, other.native_action_count)


@dataclass(frozen=True, eq=False)
class CameraObservation:
    camera_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    T_world_from_camera_optical: np.ndarray
    captured_monotonic_s: float | None = None

    def __post_init__(self):
        _text(self.camera_id, "camera_id")
        raw = np.asarray(self.rgb)
        if raw.dtype != np.uint8:
            raise ValueError("rgb must be uint8")
        rgb = _array(raw, (None, None, 3), "rgb", np.uint8)
        h, w = rgb.shape[:2]
        if not h or not w:
            raise ValueError("camera image must not be empty")
        depth = _array(self.depth_m, (h, w), "depth_m", finite=False)
        k = _array(self.intrinsics, (3, 3), "intrinsics")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("intrinsics must be a pinhole matrix with positive focal lengths")
        if not np.allclose([k[0, 1], k[1, 0]], 0):
            raise ValueError("skewed camera intrinsics are unsupported")
        t = _array(self.T_world_from_camera_optical, (4, 4), "camera transform")
        r = t[:3, :3]
        if (not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-6)):
            raise ValueError("camera transform must be a proper rigid transform")
        for name, value in (("rgb", rgb), ("depth_m", depth), ("intrinsics", k),
                            ("T_world_from_camera_optical", t)):
            object.__setattr__(self, name, value)
        if self.captured_monotonic_s is not None:
            _number(self.captured_monotonic_s, "camera capture time")

    @property
    def rgb_sha256(self) -> str:
        """Hash raw row-major uint8 RGB bytes, not a PNG encoding."""
        return hashlib.sha256(self.rgb.tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class ArmState:
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    ee_pose: Pose
    gripper_position: float

    def __post_init__(self):
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint names must be non-empty and unique")
        for name in names:
            _text(name, "joint name")
        if not isinstance(self.ee_pose, Pose):
            raise TypeError("ee_pose must be Pose")
        _number(self.gripper_position, "gripper_position")
        if self.gripper_position > 1:
            raise ValueError("gripper_position must be in [0, 1]")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_positions",
                           _array(self.joint_positions, (len(names),), "joint_positions"))


@dataclass(frozen=True, eq=False)
class RobotState:
    embodiment: str
    arms: Mapping[str, ArmState]

    def __post_init__(self):
        _text(self.embodiment, "embodiment")
        if not self.arms:
            raise ValueError("robot state must contain arms")
        object.__setattr__(self, "arms", _mapping(self.arms, ArmState, "arms"))


@dataclass(frozen=True, eq=False)
class Observation:
    source: ObservationSource
    cameras: Mapping[str, CameraObservation]
    robot_state: RobotState
    instruction: str

    def __post_init__(self):
        if not isinstance(self.source, ObservationSource) or not isinstance(self.robot_state, RobotState):
            raise TypeError("observation requires source and robot_state contracts")
        _text(self.instruction, "instruction")
        cameras = _mapping(self.cameras, CameraObservation, "cameras")
        if not cameras or any(key != c.camera_id for key, c in cameras.items()):
            raise ValueError("camera mapping keys must match camera_id")
        object.__setattr__(self, "cameras", cameras)


@dataclass(frozen=True, eq=False)
class Segmentation:
    source: ObservationSource
    camera_id: str
    mask: np.ndarray
    box_xyxy: np.ndarray
    score: float
    prompt: str
    provider: str
    model_version: str
    input_rgb_sha256: str

    def __post_init__(self):
        if not isinstance(self.source, ObservationSource):
            raise TypeError("source must be ObservationSource")
        for name in ("camera_id", "prompt", "provider", "model_version"):
            _text(getattr(self, name), name)
        digest = self.input_rgb_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("input_rgb_sha256 must be a lowercase SHA256 hex digest")
        if np.asarray(self.mask).dtype != np.bool_:
            raise ValueError("mask must be boolean")
        mask = _array(self.mask, (None, None), "mask", np.bool_)
        if not mask.size or not mask.any():
            raise ValueError("an instance mask must contain foreground pixels")
        box = _array(self.box_xyxy, (4,), "box_xyxy")
        h, w = mask.shape
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError("box must be nonempty XYXY within image bounds (exclusive maximum)")
        _number(self.score, "score")
        if self.score > 1:
            raise ValueError("score must be in [0, 1]")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "box_xyxy", box)

    def validate_against(self, observation: Observation) -> None:
        if self.source != observation.source:
            raise ValueError("segmentation and observation sources differ")
        if self.camera_id not in observation.cameras:
            raise ValueError("segmentation camera is unavailable")
        camera = observation.cameras[self.camera_id]
        if self.mask.shape != camera.depth_m.shape or self.input_rgb_sha256 != camera.rgb_sha256:
            raise ValueError("segmentation does not match the source image")


@dataclass(frozen=True, eq=False)
class PointCloud:
    source: ObservationSource
    camera_id: str
    frame: str
    points: np.ndarray

    def __post_init__(self):
        if not isinstance(self.source, ObservationSource):
            raise TypeError("source must be ObservationSource")
        _text(self.camera_id, "camera_id")
        _text(self.frame, "frame")
        object.__setattr__(self, "points", _array(self.points, (None, 3), "points"))


@dataclass(frozen=True, eq=False)
class ArmCommand:
    """One EE or joint target; gripper values are normalized commands, not force."""

    gripper_position: float
    ee_target: Pose | None = None
    joint_targets: np.ndarray | None = None

    def __post_init__(self):
        if (self.ee_target is None) == (self.joint_targets is None):
            raise ValueError("specify exactly one EE or joint target")
        if self.ee_target is not None and not isinstance(self.ee_target, Pose):
            raise TypeError("ee_target must be Pose")
        if self.joint_targets is not None:
            targets = _array(self.joint_targets, (None,), "joint_targets")
            if not targets.size:
                raise ValueError("joint_targets must not be empty")
            object.__setattr__(self, "joint_targets", targets)
        _number(self.gripper_position, "gripper_position")
        if self.gripper_position > 1:
            raise ValueError("gripper_position must be in [0, 1]")


@dataclass(frozen=True, eq=False)
class RobotAction:
    """A single native submission, with its source state explicit."""

    source: ObservationSource
    arms: Mapping[str, ArmCommand]

    def __post_init__(self):
        if not isinstance(self.source, ObservationSource):
            raise TypeError("source must be ObservationSource")
        if not self.arms:
            raise ValueError("action must contain arm commands")
        object.__setattr__(self, "arms", _mapping(self.arms, ArmCommand, "arms"))

    def validate_against(self, observation: Observation) -> None:
        if not self.source.same_state(observation.source):
            raise ValueError("action source state is stale")
        state = observation.robot_state
        if set(self.arms) != set(state.arms):
            raise ValueError("action must explicitly command every arm")
        modes = {command.ee_target is not None for command in self.arms.values()}
        if len(modes) != 1:
            raise ValueError("all arms must use the same command mode")
        for name, command in self.arms.items():
            measured = state.arms[name]
            if command.ee_target is not None:
                if command.ee_target.frame != measured.ee_pose.frame:
                    raise ValueError("EE target frame does not match measured frame")
            elif len(command.joint_targets) != len(measured.joint_names):
                raise ValueError("joint target count does not match robot profile")


class ErrorCode(str, Enum):
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    NO_CANDIDATES = "no_candidates"
    NO_VALID_DEPTH = "no_valid_depth"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    AMBIGUOUS = "ambiguous"
    STALE_EVIDENCE = "stale_evidence"
    GEOMETRY_FAILED = "geometry_failed"
    EXECUTION_FAILED = "execution_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class HarnessError:
    code: ErrorCode
    message: str

    def __post_init__(self):
        if not isinstance(self.code, ErrorCode):
            raise TypeError("code must be ErrorCode")
        _text(self.message, "message")


T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    """Data may accompany a failure as partial evidence; ok is execution only."""

    data: T | None = None
    error: HarnessError | None = None

    def __post_init__(self):
        if self.error is not None and not isinstance(self.error, HarnessError):
            raise TypeError("error must be HarnessError")
        if self.data is None and self.error is None:
            raise ValueError("result must contain data or an error")

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class StepReceipt:
    action_id: str
    observation: Observation
    terminated: bool
    truncated: bool
    native_task_success: bool

    def __post_init__(self):
        _text(self.action_id, "action_id")
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be Observation")
        for name in ("terminated", "truncated", "native_task_success"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.native_task_success and not self.terminated:
            raise ValueError("native success requires terminal evidence")
