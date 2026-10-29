"""RoboDojo worker and client; no Isaac imports in the harness process."""

from dataclasses import dataclass
import hashlib
import time
import uuid

import numpy as np

from embodied_harness.contracts import (
    ArmState, CameraObservation, Observation, ObservationSource, Pose, RobotState, StepReceipt,
)
from embodied_harness.contracts.types import _count, _text
from embodied_harness.embodiments import RobotProfile
from embodied_harness.wire import dumps


@dataclass(frozen=True)
class NativeFrame:
    public: dict
    native_action_count: int
    terminated: bool = False
    truncated: bool = False
    success: bool = False

    def __post_init__(self):
        _count(self.native_action_count, "native_action_count")
        for name in ("terminated", "truncated", "success"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("native status flags must be boolean")


class ExistingTaskBackend:
    """Remote factory can wrap an already provisioned RoboDojo task instance."""

    def __init__(self, task):
        self.task = task

    def capture(self):
        return NativeFrame(self.task.public_observation(), int(self.task.control_steps),
                           bool(self.task.terminated), bool(self.task.truncated), bool(self.task.success))

    def step(self, action):
        self.task.control_step(action)
        return self.capture()

    def close(self):
        self.task.close()


def convert_frame(frame, source, profile):
    if not isinstance(frame, NativeFrame):
        raise TypeError("backend must return NativeFrame")
    native = frame.public
    cameras = {}
    for name, value in native["vision"].items():
        transform = np.asarray(value["extrinsic_matrix"], dtype=float)
        if profile.camera_convention == "opengl_verified":
            transform = transform @ np.diag([1, -1, -1, 1])
        cameras[name] = CameraObservation(name, value["color"], value["depth"],
                                         value["intrinsic_matrix"], transform)
    arms = {}
    for name, arm in profile.arms.items():
        state = native["state"]
        pose = np.asarray(state[name + "_ee_pose"])
        gripper = np.asarray(state[name + "_ee_joint_state"])
        if pose.shape != (7,) or gripper.shape != (1,):
            raise ValueError("unexpected RoboDojo pose/gripper shape")
        arms[name] = ArmState(arm.joint_names, state[name + "_arm_joint_state"],
                              Pose(pose[:3], pose[3:], "world"), float(gripper[0]))
    return Observation(source, cameras, RobotState(profile.embodiment, arms), native["instruction"])


class RoboDojoWorker:
    """Single-episode authority. RPC server serializes all access to this object."""

    def __init__(self, backend, profile: RobotProfile, episode_id, *, native_action_limit,
                 phase="playground", max_receipts=10000):
        _text(episode_id, "episode_id")
        _count(native_action_limit, "native_action_limit")
        if native_action_limit == 0 or phase not in ("playground", "test"):
            raise ValueError("positive native budget and valid phase required")
        _count(max_receipts, "max_receipts")
        if max_receipts == 0:
            raise ValueError("positive receipt capacity required")
        self.backend, self.profile, self.episode_id = backend, profile, episode_id
        self.native_action_limit, self.phase, self.max_receipts = native_action_limit, phase, max_receipts
        self.version, self.reset_epoch = 0, 0
        self.receipts, self.faulted = {}, False
        self.last_frame = None
        self.session_id = uuid.uuid4().hex

    def observe(self):
        frame = self.backend.capture()
        if self.last_frame and frame.native_action_count < self.last_frame.native_action_count:
            self.faulted = True
            raise RuntimeError("native counter regressed; reset is not permitted through this worker")
        if self.last_frame and frame.native_action_count != self.last_frame.native_action_count:
            self.version += 1
        self.last_frame = frame
        source = ObservationSource(self.episode_id, uuid.uuid4().hex, self.reset_epoch,
                                   self.version, frame.native_action_count, time.monotonic())
        self.current = convert_frame(frame, source, self.profile)
        return self.current

    def describe(self):
        return {"session_id": self.session_id, "episode_id": self.episode_id, "phase": self.phase,
                "calibration_id": self.profile.calibration_id, "native_action_limit": self.native_action_limit,
                "faulted": self.faulted}

    def execute(self, action, action_id, session_id):
        _text(action_id, "action_id")
        if session_id != self.session_id:
            raise ValueError("worker session changed; reconcile before issuing an action")
        digest = hashlib.sha256(dumps(action)).hexdigest()
        if action_id in self.receipts:
            old_digest, receipt = self.receipts[action_id]
            if digest != old_digest:
                raise ValueError("action ID reused with a different command")
            return receipt
        if self.faulted:
            raise RuntimeError("uncertain previous action; worker requires operator reconciliation")
        if len(self.receipts) >= self.max_receipts:
            raise RuntimeError("receipt capacity reached")
        before = self.observe()
        if (self.last_frame.terminated or self.last_frame.truncated
                or before.source.native_action_count >= self.native_action_limit):
            raise RuntimeError("episode already terminal or native budget exhausted")
        self.profile.validate_action(action, before)
        native_action = {}
        for name, command in action.arms.items():
            if command.ee_target is not None:
                pose = command.ee_target
                native_action[name + "_ee_pose"] = np.r_[pose.position, pose.quaternion_wxyz]
            else:
                native_action[name + "_arm_joint_state"] = command.joint_targets
            native_action[name + "_ee_joint_state"] = np.array([command.gripper_position])
        self.faulted = True  # Any exception after dispatch leaves physical outcome uncertain.
        frame = self.backend.step(native_action)
        self.version += 1
        self.last_frame = frame
        source = ObservationSource(self.episode_id, uuid.uuid4().hex, self.reset_epoch,
                                   self.version, frame.native_action_count, time.monotonic())
        after = convert_frame(frame, source, self.profile)
        if frame.native_action_count < before.source.native_action_count:
            raise RuntimeError("native action counter regressed")
        self.current = after
        if frame.native_action_count > self.native_action_limit:
            raise RuntimeError("native driver exceeded action limit; worker remains faulted")
        receipt = StepReceipt(action_id, after, frame.terminated, frame.truncated,
                              frame.success and frame.terminated)
        self.receipts[action_id] = (digest, receipt)
        self.faulted = False
        return receipt

    def methods(self):
        return {"observe": self.observe, "execute": self.execute, "describe": self.describe,
                "get_receipt": self.get_receipt}

    def get_receipt(self, action_id, session_id):
        if session_id != self.session_id:
            raise ValueError("worker session changed")
        value = self.receipts.get(action_id)
        return value[1] if value else None


class RemoteRoboDojo:
    def __init__(self, rpc):
        self.rpc = rpc
        self.metadata = rpc.call("describe")

    def observe(self):
        value = self.rpc.call("observe")
        if not isinstance(value, Observation) or value.source.episode_id != self.metadata["episode_id"]:
            raise ValueError("invalid worker observation")
        return value

    def execute(self, action, action_id):
        value = self.rpc.call("execute", action=action, action_id=action_id,
                              session_id=self.metadata["session_id"])
        if not isinstance(value, StepReceipt) or value.action_id != action_id:
            raise ValueError("invalid worker receipt")
        return value
