"""Concrete campaign driver for an independently provisioned simulator service."""

from pathlib import Path
import uuid

from embodied_harness.cli import build_session


class SessionEpisode:
    def __init__(self, session, workspace, close):
        self.session, self.workspace, self.cleanup = session, Path(workspace), close
        self.worker_session_id = session.executor.adapter.metadata["session_id"]
        self.agent_session_id = uuid.uuid4().hex

    def context(self):
        return {"observation": self.session.observe(), "public_api": self.session.api.registry.describe(),
                "parameter_selection": {"mode": self.session.parameter_selection_mode,
                                        "jev_override_threshold": self.session.jev_override_threshold}}

    def run(self, plan):
        return self.session.run_task(plan.goal, plan, allow_system2=False)

    def close(self):
        self.cleanup()


class ProvisionedEpisodes:
    """RPC provisioner owns one fresh native worker per job and verifies layout hash.

    Agent RPC never receives this client or host job metadata. No SSH, Docker or
    automatic replay is hidden here. Provisioning errors remain terminal trials.
    """
    def __init__(self, provisioner, harness_config):
        self.provisioner, self.config = provisioner, dict(harness_config)

    def __call__(self, job, workspace):
        allocation = self.provisioner.call("open_episode", job=job)
        if set(allocation) != {"allocation_id", "worker_url", "layout_sha256"}:
            raise ValueError("invalid provisioner response")
        allocation_id = allocation["allocation_id"]
        def close():
            self.provisioner.call("close_episode", allocation_id=allocation_id)
        try:
            if allocation["layout_sha256"] != job["layout_sha256"]:
                raise ValueError("provisioned native layout differs from partition")
            config = {**self.config, "worker_url": allocation["worker_url"], "run_dir": str(Path(workspace).parent / "harness")}
            session = build_session(config)
            if session.executor.adapter.metadata["phase"] != "test":
                raise ValueError("campaign trials require strict worker phase")
            return SessionEpisode(session, workspace, close)
        except BaseException:
            close()
            raise


class NativeProvisioner:
    """Run on the simulator machine; one native process at a time, ordinary Python."""
    def __init__(self, *, runtime_root, profile, root, python_command, worker_token_env,
                 startup_timeout_s=300, backend="robodojo", record_video=False):
        if backend not in ("robodojo", "libero_plus"):
            raise ValueError("unsupported native backend")
        self.backend = backend
        if type(record_video) is not bool or (record_video and backend != "libero_plus"):
            raise ValueError("video recording is supported for LIBERO-Plus only")
        self.record_video = record_video
        import os
        if not isinstance(python_command, list) or not python_command or not all(isinstance(x, str) and x for x in python_command):
            raise ValueError("python_command requires an argv list for the Isaac Python runtime")
        if not os.environ.get(worker_token_env) or not 0 < startup_timeout_s <= 900:
            raise ValueError("worker token and bounded startup timeout required")
        self.runtime_root, self.profile = str(Path(runtime_root).resolve()), str(Path(profile).resolve())
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.command, self.token_env, self.timeout = python_command, worker_token_env, startup_timeout_s
        self.active = {}

    def open_episode(self, job):
        import json
        import os
        import socket
        import subprocess
        import time
        if self.backend == "libero_plus":
            from embodied_harness.adapters.libero_plus import layout_catalog
        else:
            from embodied_harness.adapters.robodojo_native import layout_catalog
        from embodied_harness.rpc import RpcClient
        if self.active:
            raise RuntimeError("close previous native allocation first")
        if set(job) != {"job_id", "task", "scene_ref", "layout_sha256", "phase"} or job["phase"] != "test":
            raise ValueError("strict campaign allocation required")
        catalog = layout_catalog(self.runtime_root, job["task"])
        if job["scene_ref"] not in catalog or catalog[job["scene_ref"]]["layout_sha256"] != job["layout_sha256"]:
            raise ValueError("native layout hash differs from locked partition")
        identity = uuid.uuid4().hex
        directory = self.root / identity
        directory.mkdir(exist_ok=False)
        (directory / "job.json").write_text(json.dumps(job))
        config = {"runtime_root": self.runtime_root, "task": job["task"], "scene_ref": job["scene_ref"],
                  "run_dir": str(directory / "native")}
        if self.backend == "libero_plus":
            config["record_video"] = self.record_video
        path = directory / "native-config.json"
        path.write_text(json.dumps(config))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [*self.command, "-m", "embodied_harness.cli", "native-worker", "--config", str(path),
                   "--backend", self.backend, "--profile", self.profile, "--episode-id", identity, "--phase", "test",
                   "--port", str(port), "--token-env", self.token_env]
        with (directory / "process.log").open("ab") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        self.active[identity] = process
        url = f"http://127.0.0.1:{port}"
        rpc = RpcClient(url, os.environ[self.token_env], timeout_s=2)
        deadline = time.monotonic() + self.timeout
        try:
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    metadata = rpc.call("describe")
                except (OSError, RuntimeError):
                    time.sleep(.2)
                    continue
                if metadata["episode_id"] != identity or metadata["phase"] != "test":
                    raise ValueError("native worker identity mismatch")
                return {"allocation_id": identity, "worker_url": url, "layout_sha256": job["layout_sha256"]}
            raise RuntimeError("native startup failed/timed out; inspect private process.log")
        except BaseException:
            self.close_episode(identity)
            raise

    def close_episode(self, allocation_id):
        import os
        import signal
        import subprocess
        process = self.active.pop(allocation_id)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                raise RuntimeError("native process required forced cleanup")
        return {"closed": True}

    def close(self):
        for identity in list(self.active):
            self.close_episode(identity)

    def methods(self):
        return {"open_episode": self.open_episode, "close_episode": self.close_episode, "describe": self.describe}

    def describe(self):
        from embodied_harness.embodiments import RobotProfile
        profile = RobotProfile.from_file(self.profile)
        return {"protocol": self.backend.replace("_", "-") + "-native-provisioner-v1", "commissioned": profile.commissioned,
                "calibration_id": profile.calibration_id, "active_allocations": len(self.active),
                "runtime_present": (Path(self.runtime_root) / ("libero/libero/envs/env_wrapper.py" if self.backend == "libero_plus" else "src/eval_client/eval_env.py")).is_file()}
