"""Single-submission motion with measured-state hold for the other arm."""

import time
import uuid

import numpy as np

from embodied_harness.contracts import ArmCommand, Pose, RobotAction, StepReceipt
from embodied_harness.contracts.types import _number, _count


def pose_error(measured: Pose, target: Pose):
    if measured.frame != target.frame:
        raise ValueError("pose frame mismatch")
    dot = abs(float(measured.quaternion_wxyz @ target.quaternion_wxyz))
    return float(np.linalg.norm(target.position - measured.position)), float(2 * np.arccos(np.clip(dot, 0, 1)))


def servo_action(observation, arm, target, *, opening=None, translation_m=0.01, rotation_rad=0.1):
    for name, value, cap in (("translation_m", translation_m, 0.03), ("rotation_rad", rotation_rad, 0.15)):
        _number(value, name)
        if not 0 < value <= cap:
            raise ValueError(f"{name} outside per-submission cap")
    measured = observation.robot_state.arms[arm]
    distance, angle = pose_error(measured.ee_pose, target)
    xyz = measured.ee_pose.position + (target.position - measured.ee_pose.position) * min(1, translation_m / max(distance, 1e-15))
    a, b = measured.ee_pose.quaternion_wxyz, target.quaternion_wxyz
    if a @ b < 0:
        b = -b
    fraction = min(1, rotation_rad / max(angle, 1e-15))
    if angle < 1e-6:
        q = a + fraction * (b - a)
    else:
        half = angle / 2
        q = (np.sin((1 - fraction) * half) * a + np.sin(fraction * half) * b) / np.sin(half)
    q = q / np.linalg.norm(q)
    commands = {name: ArmCommand(state.gripper_position, ee_target=state.ee_pose)
                for name, state in observation.robot_state.arms.items()}
    commands[arm] = ArmCommand(measured.gripper_position if opening is None else opening,
                               ee_target=Pose(xyz, q, target.frame))
    return RobotAction(observation.source, commands)


class Executor:
    def __init__(self, adapter, *, max_submissions=1000):
        _count(max_submissions, "max_submissions")
        if max_submissions == 0:
            raise ValueError("positive submission budget required")
        self.adapter, self.max_submissions = adapter, max_submissions
        self.submissions, self.faulted, self.terminal = 0, False, False
        self.current, self.received_at = None, None
        self.last_action_id = None

    def observe(self):
        self.current = self.adapter.observe()
        self.received_at = time.monotonic()
        return self.current

    def execute(self, action, *, max_age_s=10, action_id=None):
        _number(max_age_s, "max_age_s")
        if self.faulted or self.terminal or self.submissions >= self.max_submissions:
            raise RuntimeError("executor blocked, terminal, or out of budget")
        if self.current is None or time.monotonic() - self.received_at > max_age_s:
            raise ValueError("observation too old")
        action.validate_against(self.current)
        self.submissions += 1
        self.faulted = True
        self.last_action_id = action_id or uuid.uuid4().hex
        receipt = self.adapter.execute(action, self.last_action_id)
        if not isinstance(receipt, StepReceipt):
            raise TypeError("adapter must return StepReceipt")
        if receipt.action_id != self.last_action_id:
            raise ValueError("receipt action ID mismatch")
        after = receipt.observation
        before = action.source
        if (after.source.episode_id != before.episode_id or after.source.reset_epoch != before.reset_epoch
                or after.source.state_version <= before.state_version
                or after.source.native_action_count < before.native_action_count):
            raise ValueError("receipt has inconsistent state provenance")
        self.current, self.received_at = after, time.monotonic()
        self.terminal = receipt.terminated or receipt.truncated
        self.faulted = False
        return receipt
