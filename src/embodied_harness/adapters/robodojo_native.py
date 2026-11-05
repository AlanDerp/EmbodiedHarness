"""Standalone RoboDojo backend, adapted from EmbodiedDualLoop's direct adapter.

Native imports are confined to the dedicated simulator process. No sibling checkout is imported.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np

from .robodojo import NativeFrame


def layout_catalog(runtime_root, task):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task):
        raise ValueError("invalid native task name")
    result = {}
    for bank in (0, 1, 2):
        folder = Path(runtime_root)/"Assets/Eval_Layout/RoboDojo/arx_x5"/str(bank)
        pattern = re.compile(re.escape(task)+r"_(\d+)\.json")
        files = sorted((p for p in folder.glob(f"{task}_*.json") if pattern.fullmatch(p.name)),
                       key=lambda p: int(pattern.fullmatch(p.name).group(1)))
        for ordinal, file in enumerate(files):
            result[bank*1_000_000+ordinal] = {"bank": bank, "ordinal": ordinal,
                "layout_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "filename": file.name}
    return result


class _NoPolicyServer:
    def __init__(self, **kwargs):
        pass

    def call(self, **kwargs):
        if kwargs.get("func_name") != "reset":
            raise RuntimeError("policy execution is owned by the harness")

    def close(self):
        pass


def configure_rgbd(cfg):
    vision = cfg.eval_cfg.observation.vision
    vision.depth, vision.approximate_depth = True, False
    vision.intrinsic_matrix, vision.extrinsic_matrix = True, True
    for capture in cfg.camera.annotator.values():
        if capture.get("enabled", False):
            capture["distance_to_image_plane_capture"] = {"type": "distance_to_image_plane", "device": "cpu"}


class RoboDojoNative:
    def __init__(self, runtime_root, task, scene_ref, run_dir):
        if type(scene_ref) is not int:
            raise ValueError("scene_ref must be an encoded bank/ordinal integer")
        self.root, self.task_name = Path(runtime_root).resolve(strict=True), task
        catalog = layout_catalog(self.root, task)
        if scene_ref not in catalog:
            raise ValueError("scene_ref outside native finite layout bank")
        self.layout = catalog[scene_ref]
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.native, self.app, self.monitor = None, None, None
        self.count, self.closed, self.failed = 0, False, False
        self.public = None
        try:
            self._start()
            self.action_limit = int(self.native.step_lim)
            if self.action_limit <= 0:
                raise ValueError("native action limit must be positive")
            with self._cwd():
                if self.monitor is not None:
                    self.monitor.reset()
                with self._restore_initial_robot():
                    self.native.reset(seed=[self.layout["ordinal"]])
                if self.native.episode_nums != 1 or bool(self.native.end_flag[0]):
                    raise RuntimeError("native layout initialization failed stability checks")
                self._check_robot_state(default=False)
                self.native.run_reward()
                if hasattr(self.native, "get_score"):
                    self.native.get_score()
                if getattr(self.native, "interact", False) and hasattr(self.native, "query_support_arm_traj"):
                    self.native.query_support_arm_traj(env_idx=0)
                self.public = copy.deepcopy(self.native.get_obs())
            identity = {"task": task, "scene_ref": scene_ref, "layout": self.layout,
                        "native_action_limit": self.action_limit,
                        "runtime_entry_sha256": hashlib.sha256((self.root/"src/eval_client/eval_env.py").read_bytes()).hexdigest()}
            (self.run_dir/"native.json").write_text(json.dumps(identity, sort_keys=True))
        except BaseException:
            self.close()
            raise

    @contextlib.contextmanager
    def _cwd(self):
        old = Path.cwd()
        os.chdir(self.root)
        try:
            yield
        finally:
            os.chdir(old)

    def _start(self):
        sys.path[:0] = [str(self.root), str(self.root/"XPolicyLab")]
        with self._cwd():
            from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
            from isaaclab.app import AppLauncher
            from utils.load_file import load_yaml
            task_cfg = load_yaml(str(self.root/"task/RoboDojo/config"/(self.task_name+".yml")))
            monitor_enabled = bool(task_cfg.get("Articulation"))
            if monitor_enabled:
                from src.eval_client.physx_warning_monitor import get_monitor
                self.monitor = get_monitor()
                self.monitor.start(enabled=True)
            parser = argparse.ArgumentParser()
            AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args([])
            args.headless, args.enable_cameras, args.device = True, True, "cuda:0"
            add_zero_delay_kit_args(args)
            self.app = AppLauncher(args).app
            from omegaconf import OmegaConf
            from utils.pipeline_utils import process_config, process_randomization
            import src.eval_client.eval_env as module
            base = self.root/"env_cfg"
            evaluation = load_yaml(str(base/"arx_x5.yml"))
            evaluation.update(task_name=self.task_name, num_envs=1, device_id=0, eval_batch=False,
                eval_num=1, policy_name="EmbodiedHarness", seed=self.layout["bank"],
                additional_info="aspire-fixloop-dualloop-robodojo-v1", physx_monitor_enabled=monitor_enabled)
            cfg = OmegaConf.create({"eval_cfg": evaluation,
                "deploy_cfg": {"policy_name": "EmbodiedHarness", "port": 1}, "task_env": task_cfg,
                **{name: load_yaml(str(base/name/(evaluation["config"][name]+".yml")))
                   for name in ("sim", "scene", "camera", "robot")}})
            cfg.sim.scene.num_envs, cfg.sim.seed, cfg.sim.device = 1, [self.layout["ordinal"]], "cpu"
            cfg, _ = process_config(process_randomization(cfg), self.task_name)
            cfg.camera.default_frequency = evaluation["observation"]["collect_freq"]
            configure_rgbd(cfg)
            with patch.object(module, "WsModelClient", _NoPolicyServer):
                self.native = module.create_eval_env(cfg, self.app,
                    resume_state={"save_dir": str(self.run_dir), "details": {}})
            configure = self.native.setup_sim_cfg
            def private_logs(config):
                configure(config)
                self.native.sim_cfg.sim.log_dir = str(self.run_dir/"isaaclab-logs")
            self.native.setup_sim_cfg = private_logs
            from .capture import install_capture_cleanup
            install_capture_cleanup(self.native.capture_manager)

    def _check_robot_state(self, *, default=True):
        manager = self.native.robot_manager
        for robot, articulation in zip(manager.robot_list, manager.robot_key, strict=True):
            if robot.robot_type != "arm":
                continue
            for name, tolerance in (("joint_pos", 1e-3), ("joint_vel", 1e-2)):
                actual = getattr(articulation.data, name).detach().cpu().numpy()
                if not np.isfinite(actual).all():
                    raise RuntimeError("nonfinite robot state after native initialization")
                if default:
                    expected = getattr(articulation.data, "default_"+name).detach().cpu().numpy()
                    if not np.allclose(actual, expected, atol=tolerance, rtol=0):
                        raise RuntimeError("native robot did not restore default state")

    @contextlib.contextmanager
    def _restore_initial_robot(self):
        manager = self.native.robot_manager
        original = manager.set_robot_init_pose
        def restore():
            original()
            for robot, articulation in zip(manager.robot_list, manager.robot_key, strict=True):
                if robot.robot_type == "arm":
                    articulation.reset()
                    articulation.write_joint_state_to_sim(articulation.data.default_joint_pos.clone(),
                                                         articulation.data.default_joint_vel.clone())
            self._check_robot_state()
        with patch.object(manager, "set_robot_init_pose", restore):
            yield

    def capture(self):
        if self.public is None:
            raise RuntimeError("native observation unavailable")
        ended = bool(self.native.end_flag[0])
        success = ended and bool(self.native.success[0])
        truncated = ended and not success and self.count >= self.action_limit
        return NativeFrame(copy.deepcopy(self.public), self.count, ended and not truncated, truncated, success)

    def step(self, action):
        if self.failed or self.capture().terminated or self.capture().truncated or self.count >= self.action_limit:
            raise RuntimeError("native episode ended, failed, or exhausted")
        action = {name: np.asarray(value, dtype=np.float64) for name, value in action.items()}
        if any(value.ndim != 1 or not np.isfinite(value).all() for value in action.values()):
            raise ValueError("native actions must contain finite vectors")
        self.native.validate_action_dict(action)
        self.failed = True
        with self._cwd():
            before = int(self.native.take_action_cnt[0])
            self.native.take_action(action)
            consumed = int(self.native.take_action_cnt[0])-before
            if consumed < 0:
                raise RuntimeError("native action counter regressed")
            self.count += consumed
            if 0 in self.native.unstable_envs:
                raise RuntimeError("native simulation became unstable")
            self.public = copy.deepcopy(self.native.get_obs())
        self.failed = False
        return self.capture()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.native is not None:
                with self._cwd():
                    self.native.close()
        finally:
            if self.app is not None:
                self.app.close()


def from_config(config_path):
    return RoboDojoNative(**json.loads(Path(config_path).read_text()))
