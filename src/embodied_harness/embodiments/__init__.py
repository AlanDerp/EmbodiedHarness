"""Explicit commissioned robot profiles, not simulator-name dimension constants."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from embodied_harness.contracts import Pose
from embodied_harness.contracts.types import _array, _text


@dataclass(frozen=True, eq=False)
class ArmProfile:
    joint_names: tuple[str, ...]
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    workspace_lower: np.ndarray
    workspace_upper: np.ndarray
    T_ee_from_tcp: np.ndarray

    def __post_init__(self):
        from embodied_harness.geometry import _rigid_transform
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("unique joint names required")
        for name in names:
            _text(name, "joint name")
        object.__setattr__(self, "joint_names", names)
        for name, shape in (("joint_lower", (len(names),)), ("joint_upper", (len(names),)),
                            ("workspace_lower", (3,)), ("workspace_upper", (3,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.joint_lower >= self.joint_upper) or np.any(self.workspace_lower >= self.workspace_upper):
            raise ValueError("profile bounds must be increasing")
        object.__setattr__(self, "T_ee_from_tcp", _rigid_transform(self.T_ee_from_tcp))


@dataclass(frozen=True)
class RobotProfile:
    embodiment: str
    calibration_id: str
    camera_convention: str
    commissioned: bool
    arms: dict[str, ArmProfile]

    def __post_init__(self):
        _text(self.embodiment, "embodiment")
        _text(self.calibration_id, "calibration_id")
        if self.camera_convention not in ("optical_verified", "opengl_verified"):
            raise ValueError("explicitly verified camera convention required")
        if type(self.commissioned) is not bool or set(self.arms) != {"left", "right"}:
            raise ValueError("explicit commissioning state and both ARX arms required")
        if any(not isinstance(arm, ArmProfile) for arm in self.arms.values()):
            raise TypeError("arms must contain ArmProfile")
        object.__setattr__(self, "arms", MappingProxyType(dict(self.arms)))

    @classmethod
    def from_file(cls, path):
        value = json.loads(Path(path).read_text())
        value["arms"] = {key: ArmProfile(**arm) for key, arm in value["arms"].items()}
        return cls(**value)

    def validate_action(self, action, observation):
        if not self.commissioned:
            raise PermissionError("motion disabled until profile commissioning")
        action.validate_against(observation)
        if observation.robot_state.embodiment != self.embodiment:
            raise ValueError("embodiment mismatch")
        for name, command in action.arms.items():
            arm = self.arms[name]
            if observation.robot_state.arms[name].joint_names != arm.joint_names:
                raise ValueError("joint order differs from profile")
            if command.ee_target is not None:
                if command.ee_target.frame != "world":
                    raise ValueError("native EE commands require explicit world frame")
                xyz = command.ee_target.position
                if np.any(xyz < arm.workspace_lower) or np.any(xyz > arm.workspace_upper):
                    raise ValueError("EE target outside configured workspace")
            elif np.any(command.joint_targets < arm.joint_lower) or np.any(command.joint_targets > arm.joint_upper):
                raise ValueError("joint target outside configured limits")

    def tcp_target_to_ee(self, arm, target: Pose):
        from embodied_harness.geometry import matrix_to_pose, pose_to_matrix
        if target.frame != "world":
            raise ValueError("TCP target must explicitly be in world frame")
        world_from_ee = pose_to_matrix(target) @ np.linalg.inv(self.arms[arm].T_ee_from_tcp)
        return matrix_to_pose(world_from_ee, frame="world")
