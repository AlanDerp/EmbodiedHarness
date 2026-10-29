"""Conservative geometric association and measured-following evidence, not grasp certification."""

from dataclasses import dataclass

import numpy as np

from embodied_harness.contracts import ErrorCode, HarnessError, Result
from embodied_harness.contracts.types import _number, _text
from embodied_harness.geometry import ObjectGeometry


@dataclass(frozen=True)
class Track:
    object_id: str
    geometry: ObjectGeometry

    def __post_init__(self):
        _text(self.object_id, "object_id")
        if not isinstance(self.geometry, ObjectGeometry):
            raise TypeError("track requires geometry")


def associate(track, candidates, *, max_center_distance_m=0.05, max_relative_size_change=0.3):
    """Require exactly one plausible candidate; no best-score shortcut on ambiguity."""
    for name, value in (("max_center_distance_m", max_center_distance_m),
                        ("max_relative_size_change", max_relative_size_change)):
        _number(value, name)
        if value == 0:
            raise ValueError("association gates must be positive")
    reference = track.geometry
    candidates = tuple(candidates)
    plausible = []
    for candidate in candidates:
        if not isinstance(candidate, ObjectGeometry):
            raise TypeError("association requires ObjectGeometry candidates")
        if (candidate.source.episode_id != reference.source.episode_id
                or candidate.source.reset_epoch != reference.source.reset_epoch
                or candidate.source.state_version < reference.source.state_version
                or candidate.source.native_action_count < reference.source.native_action_count
                or candidate.frame != reference.frame):
            return Result(error=HarnessError(ErrorCode.STALE_EVIDENCE, "association source/frame mismatch"))
        size_change = np.abs(np.sort(candidate.extents) - np.sort(reference.extents))
        denominator = np.maximum(np.sort(reference.extents), 0.005)
        if (np.linalg.norm(candidate.center - reference.center) <= max_center_distance_m
                and np.all(size_change / denominator <= max_relative_size_change)):
            plausible.append(candidate)
    if len(plausible) != 1:
        return Result(tuple(plausible), HarnessError(
            ErrorCode.AMBIGUOUS if plausible else ErrorCode.NO_CANDIDATES,
            "identity unresolved by geometric association"))
    return Result(Track(track.object_id, plausible[0]))


def following_evidence(reference, current, before_observation, after_observation, arm, *,
                       minimum_lift_m=0.02, following_tolerance_m=0.015):
    """Compare object and EE displacement; explicitly does not assert contact or no-slip."""
    # Imported lazily to keep the perception layer independent of runtime at import time.
    from embodied_harness.runtime.session import RelationEvidence
    for name, value in (("minimum_lift_m", minimum_lift_m), ("following_tolerance_m", following_tolerance_m)):
        _number(value, name)
        if value == 0:
            raise ValueError("following thresholds must be positive")
    source = after_observation.source
    valid = (reference.object_id == current.object_id
             and reference.geometry.source == before_observation.source
             and current.geometry.source == source
             and reference.geometry.frame == current.geometry.frame == "world"
             and source.episode_id == before_observation.source.episode_id
             and source.reset_epoch == before_observation.source.reset_epoch
             and source.native_action_count > before_observation.source.native_action_count
             and source.state_version > before_observation.source.state_version)
    if not valid:
        return RelationEvidence(source, "follows_gripper", "unknown", "identity or temporal evidence inconsistent")
    before = before_observation.robot_state.arms[arm].ee_pose
    after = after_observation.robot_state.arms[arm].ee_pose
    if before.frame != "world" or after.frame != "world":
        return RelationEvidence(source, "follows_gripper", "unknown", "EE frame not world")
    ee_delta = after.position - before.position
    object_delta = current.geometry.center - reference.geometry.center
    if ee_delta[2] < minimum_lift_m:
        return RelationEvidence(source, "follows_gripper", "unknown", "insufficient measured EE lift")
    follows = object_delta[2] >= minimum_lift_m and np.linalg.norm(ee_delta - object_delta) <= following_tolerance_m
    return RelationEvidence(source, "follows_gripper", "true" if follows else "false",
                            "measured object/EE displacement comparison; not grasp certification")
