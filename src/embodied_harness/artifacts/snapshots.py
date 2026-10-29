"""Versioned JSON/NPZ snapshots, committed as a directory without overwriting evidence."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from embodied_harness.contracts import (
    ArmState, CameraObservation, Observation, ObservationSource, Pose, RobotState,
)


class SnapshotStore:
    def __init__(self, root, *, max_bytes=256 * 1024 * 1024, max_snapshots=1000):
        for name, value in (("max_bytes", max_bytes), ("max_snapshots", max_snapshots)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.max_snapshots = max_bytes, max_snapshots

    def _path(self, episode_id, snapshot_id):
        for value in (episode_id, snapshot_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("episode and snapshot IDs must be nonempty strings")
        key = json.dumps([episode_id, snapshot_id], ensure_ascii=True).encode()
        return self.root / hashlib.sha256(key).hexdigest()

    def put(self, observation: Observation) -> Path:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be Observation")
        destination = self._path(observation.source.episode_id, observation.source.snapshot_id)
        if destination.exists():
            raise FileExistsError("snapshot ID already stored; evidence cannot be overwritten")
        if sum(p.is_dir() and not p.name.startswith(".") for p in self.root.iterdir()) >= self.max_snapshots:
            raise ValueError("snapshot store capacity reached")
        arrays = {}

        def add(array):
            key = f"array_{len(arrays):04d}"
            arrays[key] = array
            return key

        cameras = []
        for camera in observation.cameras.values():
            cameras.append({"camera_id": camera.camera_id,
                            "captured_monotonic_s": camera.captured_monotonic_s,
                            **{name: add(getattr(camera, name)) for name in (
                                "rgb", "depth_m", "intrinsics", "T_world_from_camera_optical")}})
        arms = {}
        for name, arm in observation.robot_state.arms.items():
            arms[name] = {"joint_names": list(arm.joint_names), "joint_positions": add(arm.joint_positions),
                          "gripper_position": arm.gripper_position,
                          "ee_pose": {"position": add(arm.ee_pose.position),
                                      "quaternion_wxyz": add(arm.ee_pose.quaternion_wxyz),
                                      "frame": arm.ee_pose.frame}}
        if sum(a.nbytes for a in arrays.values()) > self.max_bytes:
            raise ValueError("snapshot arrays exceed byte limit")
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=self.root))
        try:
            payload = temporary / "arrays.npz"
            np.savez(payload, **arrays)
            manifest = {"schema_version": 1, "source": asdict(observation.source),
                        "instruction": observation.instruction, "cameras": cameras,
                        "robot_state": {"embodiment": observation.robot_state.embodiment, "arms": arms},
                        "arrays_sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            metadata = json.dumps(manifest, allow_nan=False, ensure_ascii=True).encode()
            if len(metadata) > self.max_bytes or payload.stat().st_size > self.max_bytes:
                raise ValueError("snapshot files exceed byte limit")
            (temporary / "manifest.json").write_bytes(metadata)
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def get(self, episode_id: str, snapshot_id: str) -> Observation:
        folder = self._path(episode_id, snapshot_id)
        if folder.is_symlink():
            raise ValueError("snapshot directory must not be a symlink")
        for name in ("manifest.json", "arrays.npz"):
            path = folder / name
            if path.is_symlink() or path.stat().st_size > self.max_bytes:
                raise ValueError("invalid snapshot file or byte limit exceeded")
        metadata = json.loads((folder / "manifest.json").read_text())
        if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
            raise ValueError("unsupported snapshot schema")
        source = ObservationSource(**metadata["source"])
        if (source.episode_id, source.snapshot_id) != (episode_id, snapshot_id):
            raise ValueError("snapshot identity mismatch")
        payload = folder / "arrays.npz"
        if hashlib.sha256(payload.read_bytes()).hexdigest() != metadata["arrays_sha256"]:
            raise ValueError("snapshot array checksum mismatch")
        with zipfile.ZipFile(payload) as archive:
            entries = archive.infolist()
            if sum(entry.file_size for entry in entries) > self.max_bytes:
                raise ValueError("expanded snapshot exceeds byte limit")
        with np.load(payload, allow_pickle=False) as arrays:
            cameras = {}
            for item in metadata["cameras"]:
                camera = CameraObservation(item["camera_id"],
                    **{name: arrays[item[name]] for name in (
                        "rgb", "depth_m", "intrinsics", "T_world_from_camera_optical")},
                    captured_monotonic_s=item["captured_monotonic_s"])
                if camera.camera_id in cameras:
                    raise ValueError("duplicate camera ID")
                cameras[camera.camera_id] = camera
            arms = {}
            for name, item in metadata["robot_state"]["arms"].items():
                pose = item["ee_pose"]
                arms[name] = ArmState(tuple(item["joint_names"]), arrays[item["joint_positions"]],
                    Pose(arrays[pose["position"]], arrays[pose["quaternion_wxyz"]], pose["frame"]),
                    item["gripper_position"])
        return Observation(source, cameras, RobotState(metadata["robot_state"]["embodiment"], arms),
                           metadata["instruction"])
