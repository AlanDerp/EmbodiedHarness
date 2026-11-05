"""Server-side acceptance against a captured public snapshot; no privileged scene state."""

import json
import re
import numpy as np

from embodied_harness.artifacts import SnapshotStore
from embodied_harness.artifacts.events import EventLog
from embodied_harness.providers.sam3 import Sam3Provider
from embodied_harness.providers.sam3.wire import MAX_BODY_BYTES
from embodied_harness.geometry import mask_to_point_cloud


def check_sam3(url, store_root, episode_id, snapshot_id, camera_id, query, checkpoint_sha256, log_path):
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("explicit checkpoint SHA256 required")
    provider = Sam3Provider(url)
    with provider._opener.open(url.rstrip("/") + "/ready", timeout=30) as response:
        ready = json.loads(response.read(MAX_BODY_BYTES))
    pattern = r"sam3:code:[0-9a-f]{64}:checkpoint:" + checkpoint_sha256
    if ready.get("ready") is not True or not re.fullmatch(pattern, ready.get("model_version", "")):
        raise ValueError("service not ready or checkpoint/code provenance differs")
    provider.expected_model_version = ready["model_version"]
    observation = SnapshotStore(store_root).get(episode_id, snapshot_id)
    candidates = provider.segment_text(observation, camera_id, query)
    projections = []
    if candidates.ok:
        projections = [mask_to_point_cloud(observation, candidate) for candidate in candidates.data]
    point_candidates = None
    if candidates.ok and candidates.data:
        pixels = np.argwhere(candidates.data[0].mask)
        y, x = pixels[len(pixels)//2]
        point_candidates = provider.segment_point(observation, camera_id, [int(x), int(y)])
    passed = bool(candidates.ok and candidates.data and all(item.ok for item in projections)
                  and point_candidates and point_candidates.ok and point_candidates.data)
    report = {"passed": passed, "source": observation.source, "camera_id": camera_id,
              "rgb_sha256": observation.cameras[camera_id].rgb_sha256,
              "model_version": ready["model_version"], "segmentation": candidates,
              "point_segmentation": point_candidates, "projections": projections,
              "scope": "service_snapshot_text_point_depth_not_identity_or_robot_success"}
    EventLog(log_path).append("sam3_acceptance", report=report)
    return report


def doctor(config):
    """Read-only service checks; no model generation or physics steps."""
    import os
    from embodied_harness.rpc import RpcClient
    harness = config["harness"]
    expected = harness["sam3_model_version"]
    if not re.fullmatch(r"sam3:code:[0-9a-f]{64}:checkpoint:[0-9a-f]{64}", expected):
        raise ValueError("explicit SAM3 code/checkpoint pin required")
    provider = Sam3Provider(harness["sam3_url"], expected_model_version=expected)
    with provider._opener.open(harness["sam3_url"].rstrip("/") + "/ready", timeout=30) as response:
        sam3 = json.loads(response.read(MAX_BODY_BYTES))
    if sam3 != {"ready": True, "model_version": expected}:
        raise ValueError("SAM3 readiness/version mismatch")
    def describe(service):
        return RpcClient(config[service + "_url"], os.environ[config[service + "_token_env"]]).call("describe")
    agent, native = describe("agent"), describe("provisioner")
    if (agent.get("protocol") != "aspire-role-agent-v1" or not agent.get("fresh_conversation_per_request")
            or not agent.get("credentials_configured", True)
            or native.get("protocol") != config.get("native_protocol", "robodojo-native-provisioner-v1")
            or not native.get("commissioned") or not native.get("runtime_present") or native.get("active_allocations") != 0):
        raise ValueError("agent/provisioner/profile not ready or previous allocation active")
    system1 = None
    if harness.get("selector") == "remote":
        system1 = RpcClient(harness["system1_url"], os.environ[harness["system1_token_env"]]).call("describe")
        if (system1.get("protocol") != "numeric-system1-v1" or not system1.get("credentials_configured")
                or system1.get("model") != harness.get("jev_model")):
            raise ValueError("System 1 service credentials/model not ready")
    return {"passed": True, "sam3": sam3, "agent": agent, "native": native, "system1": system1,
            "scope": "readiness_only_no_generation_segmentation_physics_or_calibration_measurement"}
