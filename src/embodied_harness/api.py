"""Small typed public API composed over the session, as in ASPIRE's CapApi."""

import time

from embodied_harness.contracts import ErrorCode, HarnessError, Result
from embodied_harness.control import servo_action
from embodied_harness.geometry import estimate_geometry, mask_to_point_cloud
from embodied_harness.registry import ToolRegistry


class CapApi:
    def __init__(self, session):
        self.session = session
        self.registry = ToolRegistry()
        for name in ("get_observation", "get_instruction", "get_robot_state", "localize_object",
                     "segment_text", "segment_point", "mask_to_point_cloud", "estimate_geometry", "servo_action"):
            self.registry.register(name, getattr(self, name), description=getattr(self, name).__doc__ or "")

    def invoke(self, name, params):
        started = time.monotonic()
        try:
            result = self.registry.invoke(name, params)
        except Exception as exc:
            self.session.log.append("tool_call", tool=name, ok=False, error_type=type(exc).__name__)
            raise
        self.session.log.append("tool_call", tool=name, ok=True, seconds=time.monotonic()-started,
                                parameter_names=sorted(params), result=result)
        return result

    def methods(self):
        def wrap(name):
            return lambda **params: self.invoke(name, params)
        return {name: wrap(name) for name in self.registry.methods()}

    def get_observation(self):
        """Get and save public observation without advancing physics."""
        return self.session.observe()

    def get_instruction(self):
        observation = self.session.executor.current or self.session.observe()
        return observation.instruction

    def get_robot_state(self):
        return self.get_observation().robot_state

    def _snapshot(self, episode_id, snapshot_id):
        current = self.session.executor.current
        if current is None or current.source.episode_id != episode_id:
            raise ValueError("snapshot is outside the active episode")
        return self.session.store.get(episode_id, snapshot_id)

    def _segment(self, episode_id, snapshot_id, camera_id, prompt, point=False):
        observation = self._snapshot(episode_id, snapshot_id)
        provider = self.session.perception.provider if self.session.perception else None
        if provider is None:
            return Result(error=HarnessError(ErrorCode.NOT_CONFIGURED, "SAM3 not configured"))
        method = provider.segment_point if point else provider.segment_text
        return method(observation, camera_id, prompt)

    def segment_text(self, episode_id, snapshot_id, camera_id, query):
        return self._segment(episode_id, snapshot_id, camera_id, query)

    def segment_point(self, episode_id, snapshot_id, camera_id, point_xy):
        return self._segment(episode_id, snapshot_id, camera_id, point_xy, point=True)

    def localize_object(self, episode_id, snapshot_id, camera_id, query, candidate_index=None):
        return self.session.localize(episode_id, snapshot_id, camera_id, query, candidate_index)

    def mask_to_point_cloud(self, segmentation, target_frame="world"):
        source = segmentation.source
        observation = self._snapshot(source.episode_id, source.snapshot_id)
        return mask_to_point_cloud(observation, segmentation, target_frame=target_frame)

    def estimate_geometry(self, cloud):
        return estimate_geometry(cloud)

    def servo_action(self, observation, arm, target, opening=None, translation_m=.01, rotation_rad=.1):
        current = self.session.executor.current
        if current is None or not observation.source.same_state(current.source):
            raise ValueError("servo input is stale")
        # Never trust a program-modified robot state to define a hold or measured error.
        return servo_action(current, arm, target, opening=opening,
                            translation_m=translation_m, rotation_rad=rotation_rad)
