"""Host-only LIBERO-Plus RGB-D / Panda adapter. No object truth crosses the worker.

This is an RGB-D OSC_POSE harness condition, not the official RGB VLA protocol.
The sole arm is named right to preserve existing public servo contracts.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .robodojo import NativeFrame


def task_data(runtime_root, task):
    import libero.libero as package
    from libero.libero import benchmark, get_libero_path
    root = Path(runtime_root).resolve(strict=True)
    if not Path(package.__file__).resolve().is_relative_to(root):
        raise ValueError("LIBERO import does not match pinned runtime_root")
    suite, sep, ordinal = task.partition(":")
    if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"} or not sep or not ordinal.isdecimal():
        raise ValueError("expected LIBERO-Plus suite:task_index")
    with contextlib.redirect_stdout(io.StringIO()):
        bank = benchmark.get_benchmark_dict()[suite]()
    index = int(ordinal)
    if index >= bank.n_tasks:
        raise ValueError("task outside native suite")
    definition = bank.get_task(index)
    states = np.asarray(bank.get_task_init_states(index))
    if states.ndim != 2 or not len(states) or not np.isfinite(states).all():
        raise ValueError("invalid native initialization bank")
    bddl = Path(get_libero_path("bddl_files"))/definition.problem_folder/definition.bddl_file
    return definition, states, bddl


def layout_catalog(runtime_root, task):
    _, states, bddl = task_data(runtime_root, task)
    # State-content identity catches duplicate indices; BDDL distinguishes perturbations.
    # Upstream encodes camera/robot/noise perturbations in virtual BDDL names.
    # Its wrapper opens the base BDDL; preserve both that content and the variant.
    physical = bddl
    if not physical.is_file() and "_view_" in bddl.name and "_initstate_" in bddl.name:
        physical = bddl.with_name(bddl.name.split("_view_")[0] + ".bddl")
    prefix = bddl.name.encode() + b"\0" + physical.read_bytes()
    return {i: {"bank": 0, "ordinal": i,
                "layout_sha256": hashlib.sha256(prefix + np.asarray(state, dtype="<f8").tobytes()).hexdigest()}
            for i, state in enumerate(states)}


def pose_delta(target, measured_position, measured_xyzw):
    from scipy.spatial.transform import Rotation
    target = np.asarray(target, dtype=float)
    if target.shape != (7,) or not np.isfinite(target).all() or not np.isclose(np.linalg.norm(target[3:]), 1):
        raise ValueError("expected finite world position and unit WXYZ quaternion")
    rotation = Rotation.from_quat(target[[4, 5, 6, 3]]) * Rotation.from_quat(measured_xyzw).inv()
    return np.r_[target[:3] - measured_position, rotation.as_rotvec()]


class LiberoPlusNative:
    def __init__(self, runtime_root, task, scene_ref, run_dir, action_limit=4000):
        if type(action_limit) is not int or not 1 <= action_limit <= 4000:
            raise ValueError("bounded native action_limit required")
        definition, states, bddl = task_data(runtime_root, task)
        if type(scene_ref) is not int or not 0 <= scene_ref < len(states):
            raise ValueError("scene_ref outside native initialization bank; no modulo mapping")
        self.action_limit, self.count = action_limit, 0
        self.failed = self.terminated = self.truncated = self.success = False
        self.env = None
        self.directory = Path(run_dir)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.instruction = definition.language
        from libero.libero.envs import OffScreenRenderEnv
        try:
            self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), controller="OSC_POSE",
                horizon=action_limit + 10, control_freq=20, camera_widths=800,
                camera_heights=512, camera_depths=True)
            self.env.seed(scene_ref)
            self.env.reset()
            self.obs = self.env.set_init_state(states[scene_ref])
            for _ in range(10):
                self.obs, _, _, _ = self.env.step(np.r_[np.zeros(6), -1.])
            controller = self.env.env.robots[0].controller
            self.scale = (np.asarray(controller.output_max) - controller.output_min) / 2
            if self.scale.shape != (6,) or np.any(self.scale <= 0):
                raise ValueError("unexpected OSC action scaling")
            if not np.allclose(controller.input_min, -1) or not np.allclose(controller.input_max, 1):
                raise ValueError("unexpected OSC input bounds")
            (self.directory/"native.json").write_text(json.dumps({
                "protocol": "libero-plus-rgbd-osc-v1", "task": task, "scene_ref": scene_ref,
                "layout": layout_catalog(runtime_root, task)[scene_ref],
                "native_action_limit": action_limit, "settling_steps": 10,
                "controller": "OSC_POSE", "control_hz": 20,
                "translation_rotation_scale": self.scale.tolist()}))
        except BaseException:
            self.close()
            raise

    def capture(self):
        from robosuite.utils.camera_utils import (
            get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map)
        from robosuite import macros
        sim = self.env.env.sim
        vision = {}
        for name in ("agentview", "robot0_eye_in_hand"):
            rgb = np.asarray(self.obs[name + "_image"])
            depth = get_real_depth_map(sim, np.asarray(self.obs[name + "_depth"])).squeeze(-1)
            if macros.IMAGE_CONVENTION == "opengl":
                rgb, depth = rgb[::-1], depth[::-1]
            elif macros.IMAGE_CONVENTION != "opencv":
                raise ValueError("unrecognized image convention")
            vision[name] = {"color": rgb.copy(), "depth": depth.copy(),
                "intrinsic_matrix": get_camera_intrinsic_matrix(sim, name, *rgb.shape[:2]),
                "extrinsic_matrix": get_camera_extrinsic_matrix(sim, name)}
        q = np.asarray(self.obs["robot0_eef_quat"])
        public = {"instruction": self.instruction, "vision": vision, "state": {
            "right_ee_pose": np.r_[self.obs["robot0_eef_pos"], q[[3, 0, 1, 2]]],
            "right_arm_joint_state": np.asarray(self.obs["robot0_joint_pos"]).copy(),
            "right_ee_joint_state": np.array([np.clip(np.sum(np.abs(self.obs["robot0_gripper_qpos"]))/.08, 0, 1)])}}
        return NativeFrame(public, self.count, self.terminated, self.truncated, self.success)

    def step(self, action):
        if self.failed or self.terminated or self.truncated or self.count >= self.action_limit:
            raise RuntimeError("native episode terminal, faulted or exhausted")
        if set(action) != {"right_ee_pose", "right_ee_joint_state"}:
            raise ValueError("Panda adapter accepts world EE targets only; no joint-mode substitution")
        delta = pose_delta(action["right_ee_pose"], self.obs["robot0_eef_pos"], self.obs["robot0_eef_quat"])
        opening = np.asarray(action["right_ee_joint_state"])
        if opening.shape != (1,) or not np.isfinite(opening).all() or not 0 <= opening[0] <= 1:
            raise ValueError("gripper opening must be in [0,1]")
        # One host submission consumes exactly one native control tick, never a hidden servo loop.
        command = np.r_[np.clip(delta/self.scale, -1, 1), 1 - 2*opening[0]]
        self.failed = True
        self.obs, _, done, _ = self.env.step(command)
        self.count += 1
        self.success = bool(self.env.check_success())
        self.terminated = self.success
        self.truncated = bool(done or self.count >= self.action_limit) and not self.success
        frame = self.capture()
        self.failed = False
        return frame

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def from_config(path):
    return LiberoPlusNative(**json.loads(Path(path).read_text()))
