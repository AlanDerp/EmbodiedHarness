from dataclasses import replace
import io
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np

from embodied_harness import wire
from embodied_harness.adapters.robodojo import NativeFrame, RemoteRoboDojo, RoboDojoWorker, convert_frame
from embodied_harness.artifacts import SnapshotStore
from embodied_harness.artifacts.events import EventLog
from embodied_harness.contracts import Pose, RobotAction
from embodied_harness.control import Executor, pose_error, servo_action
from embodied_harness.embodiments import ArmProfile, RobotProfile
from embodied_harness.evaluation import freeze_knowledge, stage_knowledge, verify_knowledge
from embodied_harness.rpc import RpcClient, RpcServer
from embodied_harness.runtime import Session
from embodied_harness.runtime.selectors import FixedSelector, JevSelector, Selection
from embodied_harness.runtime.session import RelationEvidence
from test_contracts import scene


def profile(commissioned=True, convention="optical_verified"):
    arm = ArmProfile(("j1", "j2"), [-2, -2], [2, 2], [-1, -1, -1], [1, 1, 1], np.eye(4))
    return RobotProfile("arx_x5", "test-fixture-only", convention, commissioned,
                        {"left": arm, "right": arm})


class NativeFixture:
    def __init__(self, *, stuck=False, success_at=None):
        self.observation = scene()
        self.count, self.calls = 0, 0
        self.stuck, self.success_at = stuck, success_at

    def capture(self):
        obs = self.observation
        state = {}
        for name, arm in obs.robot_state.arms.items():
            state[name + "_ee_pose"] = np.r_[arm.ee_pose.position, arm.ee_pose.quaternion_wxyz]
            state[name + "_ee_joint_state"] = np.array([arm.gripper_position])
            state[name + "_arm_joint_state"] = arm.joint_positions
        c = obs.cameras["head"]
        ended = self.success_at is not None and self.calls >= self.success_at
        return NativeFrame({"instruction": obs.instruction, "state": state, "vision": {"head": {
            "color": c.rgb, "depth": c.depth_m, "intrinsic_matrix": c.intrinsics,
            "extrinsic_matrix": c.T_world_from_camera_optical}}}, self.count,
            terminated=ended, success=True)  # Upstream's initial true must not mean success.

    def step(self, action):
        self.calls += 1
        self.count += 2  # One submission need not equal one native action count.
        if not self.stuck:
            arms = {}
            for name, arm in self.observation.robot_state.arms.items():
                pose = action.get(name + "_ee_pose")
                arms[name] = replace(arm, ee_pose=Pose(pose[:3], pose[3:], "world") if pose is not None else arm.ee_pose,
                                     gripper_position=float(action[name + "_ee_joint_state"][0]))
            self.observation = replace(self.observation, robot_state=replace(self.observation.robot_state, arms=arms))
        return self.capture()

    def close(self):
        pass


class DirectRemote:
    def __init__(self, worker):
        self.worker, self.metadata = worker, worker.describe()

    def observe(self):
        return self.worker.observe()

    def execute(self, action, action_id):
        return self.worker.execute(action, action_id, self.metadata["session_id"])


def manifest(x=0.025, revision=1, **kwargs):
    return {"task_id": "move", "goal": "move robot left EE", "revision": revision,
            "stages": [{"stage_id": "move-left", "kind": "servo", "arm": "left",
                        "target": {"position": [x, 0, 0], "quaternion_wxyz": [1, 0, 0, 0], "frame": "world"}}],
            **kwargs}


class AdapterControlTests(unittest.TestCase):
    def worker(self, **kwargs):
        self.backend = NativeFixture(**kwargs)
        return RoboDojoWorker(self.backend, profile(), "episode", native_action_limit=100)

    def test_conversion_optical_and_verified_opengl(self):
        frame = NativeFixture().capture()
        source = scene().source
        optical = convert_frame(frame, source, profile())
        gl = convert_frame(frame, source, profile(convention="opengl_verified"))
        np.testing.assert_equal(optical.cameras["head"].T_world_from_camera_optical, np.eye(4))
        np.testing.assert_equal(gl.cameras["head"].T_world_from_camera_optical, np.diag([1, -1, -1, 1]))
        self.assertEqual(optical.robot_state.arms["left"].joint_names, ("j1", "j2"))

    def test_duplicate_action_does_not_move_twice(self):
        worker = self.worker()
        obs = worker.observe()
        action = servo_action(obs, "left", Pose([0.02, 0, 0], [1, 0, 0, 0], "world"))
        receipt = worker.execute(action, "same", worker.session_id)
        duplicate = worker.execute(action, "same", worker.session_id)
        self.assertIs(receipt, duplicate)
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(receipt.observation.source.native_action_count, 2)
        self.assertIs(worker.get_receipt("same", worker.session_id), receipt)
        self.assertIsNone(worker.get_receipt("missing", worker.session_id))
        self.assertFalse(receipt.native_task_success)
        with self.assertRaises(ValueError):
            worker.execute(replace(action, source=replace(action.source, snapshot_id="changed")), "same", worker.session_id)
        self.backend.count = 0
        with self.assertRaises(RuntimeError):
            worker.observe()
        self.assertTrue(worker.faulted)

    def test_stale_state_session_and_terminal_reject_motion(self):
        worker = self.worker(success_at=1)
        obs = worker.observe()
        action = servo_action(obs, "left", Pose([0.02, 0, 0], [1, 0, 0, 0], "world"))
        with self.assertRaises(ValueError):
            worker.execute(action, "a", "old-session")
        receipt = worker.execute(action, "a", worker.session_id)
        self.assertTrue(receipt.native_task_success)
        with self.assertRaises(RuntimeError):
            worker.execute(action, "b", worker.session_id)
        self.assertEqual(self.backend.calls, 1)

    def test_uncommissioned_profile_and_limits(self):
        worker = self.worker()
        obs = worker.observe()
        action = servo_action(obs, "left", Pose([0.1, 0, 0], [1, 0, 0, 0], "world"))
        with self.assertRaises(PermissionError):
            profile(False).validate_action(action, obs)
        commands = dict(action.arms)
        commands["left"] = replace(commands["left"], ee_target=Pose([2, 0, 0], [1, 0, 0, 0], "world"))
        with self.assertRaises(ValueError):
            profile().validate_action(RobotAction(obs.source, commands), obs)

    def test_dispatch_exception_latches_worker_fault(self):
        worker = self.worker()
        obs = worker.observe()
        action = servo_action(obs, "left", Pose([0.1, 0, 0], [1, 0, 0, 0], "world"))
        with patch.object(self.backend, "step", side_effect=TimeoutError()):
            with self.assertRaises(TimeoutError):
                worker.execute(action, "a", worker.session_id)
        self.assertTrue(worker.faulted)
        with self.assertRaises(RuntimeError):
            worker.execute(action, "b", worker.session_id)
        worker = self.worker()
        worker.native_action_limit = 1
        observation = worker.observe()
        action = servo_action(observation, "left", Pose([.02, 0, 0], [1, 0, 0, 0], "world"))
        with self.assertRaises(RuntimeError):
            worker.execute(action, "overshoot", worker.session_id)
        self.assertTrue(worker.faulted)

    def test_servo_is_bounded_and_other_arm_is_held(self):
        obs = scene()
        target = Pose([1, 0, 0], [0, 0, 0, 1], "world")
        action = servo_action(obs, "left", target)
        self.assertAlmostEqual(np.linalg.norm(action.arms["left"].ee_target.position), 0.01)
        self.assertAlmostEqual(pose_error(obs.robot_state.arms["left"].ee_pose, action.arms["left"].ee_target)[1], 0.1)
        np.testing.assert_equal(action.arms["right"].ee_target.position, [0, 0, 0])
        self.assertAlmostEqual(pose_error(target, Pose([1, 0, 0], [0, 0, 0, -1], "world"))[1], 0)

    def test_executor_uses_local_age_and_submission_budget(self):
        worker = self.worker()
        executor = Executor(DirectRemote(worker), max_submissions=1)
        obs = executor.observe()
        action = servo_action(obs, "left", Pose([0.02, 0, 0], [1, 0, 0, 0], "world"))
        executor.received_at -= 20
        with self.assertRaises(ValueError):
            executor.execute(action)
        self.assertEqual(self.backend.calls, 0)
        executor.observe()
        executor.execute(action)
        with self.assertRaises(RuntimeError):
            executor.execute(action)
        self.assertEqual(self.backend.calls, 1)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = NativeFixture()
        self.worker = RoboDojoWorker(self.backend, profile(), "episode", native_action_limit=100)
        self.session = Session(Executor(DirectRemote(self.worker)), SnapshotStore(Path(self.directory.name) / "snapshots"),
                               EventLog(Path(self.directory.name) / "events.jsonl"))

    def test_full_ordered_robot_pose_program_and_report(self):
        result = self.session.delegate(manifest())
        self.assertEqual(result["status"], "program_completed")
        self.assertEqual(result["completed_stage_count"], 1)
        self.assertEqual(result["submissions"], 3)
        self.assertEqual(result["native_action_delta"], 6)
        events = [wire.loads(line) for line in (Path(self.directory.name) / "events.jsonl").read_bytes().splitlines()]
        self.assertIn("action_intent", [e["kind"] for e in events])
        self.assertEqual(events[-1]["kind"], "handoff")

    def test_native_success_requires_terminal_native_evidence(self):
        self.backend.success_at = 1
        result = self.session.delegate(manifest())
        self.assertEqual(result["status"], "native_task_succeeded")
        self.assertEqual(result["submissions"], 1)

    def test_low_confidence_performs_no_action(self):
        class LowConfidence:
            def select(self, state, options):
                return Selection("normal", 0.1)
        self.session.selector = LowConfidence()
        result = self.session.delegate(manifest())
        self.assertEqual(result["reason"], "low_confidence")
        self.assertEqual(self.backend.calls, 0)

    def test_stop_during_selection_prevents_next_action(self):
        session = self.session
        class StopSelector:
            def select(self, state, options):
                session.stop()
                return Selection("normal", 1)
        session.selector = StopSelector()
        report = session.delegate(manifest())
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(self.backend.calls, 0)
        with self.assertRaises(RuntimeError):
            session.delegate(manifest(x=.03, revision=2))

    def test_rename_only_cannot_restart_unchanged_program(self):
        self.session.delegate(manifest(max_actions=1))
        value = manifest(revision=2, max_actions=1)
        value["stages"][0]["stage_id"] = "new-label"
        with self.assertRaises(ValueError):
            self.session.delegate(value)

    def test_action_budget_and_unchanged_repair_rejection(self):
        result = self.session.delegate(manifest(max_actions=1))
        self.assertEqual(result["status"], "needs_system2")
        self.assertEqual(self.backend.calls, 1)
        with self.assertRaises(ValueError):
            self.session.delegate(manifest(revision=2, max_actions=50))
        with self.assertRaises(ValueError):
            self.session.delegate(manifest(x=0.04, revision=2))
        repaired = self.session.delegate(manifest(x=0.02, revision=2, max_actions=1))
        self.assertEqual(repaired["status"], "program_completed")

    def test_no_progress_hands_back_to_system2(self):
        self.backend.stuck = True
        report = self.session.delegate(manifest(max_no_progress=2))
        self.assertEqual(report["reason"], "no_measured_progress")
        self.assertEqual(self.backend.calls, 3)

    def test_missing_unknown_and_stale_relation_never_verify(self):
        value = manifest(x=0)
        value["stages"].append({"stage_id": "verify", "kind": "relation", "relation": "lifted"})
        first = self.session.delegate(value)
        self.assertEqual(first["reason"], "relation_evaluator_unavailable")
        self.assertEqual(first["completed_stage_count"], 1)
        self.assertEqual(self.backend.calls, 0)
        self.session.evaluators["lifted"] = lambda obs: RelationEvidence(
            replace(obs.source, snapshot_id="stale"), "lifted", "true", "fixture")
        value["revision"] = 2
        value["stages"][1]["relation"] = "lifted2"
        self.session.evaluators["lifted2"] = lambda obs: RelationEvidence(
            replace(obs.source, snapshot_id="stale"), "lifted2", "true", "fixture")
        second = self.session.delegate(value)
        self.assertEqual(second["reason"], "invalid_or_stale_relation_evidence")
        self.assertEqual(second["completed_stage_count"], 1)

    def test_repair_cannot_change_completed_prefix(self):
        value = manifest(x=0)
        value["stages"].append({"stage_id": "verify", "kind": "relation", "relation": "lifted"})
        self.session.delegate(value)
        value["revision"] = 2
        value["stages"][0]["target"]["position"] = [0.01, 0, 0]
        with self.assertRaises(ValueError):
            self.session.delegate(value)

    def test_required_evidence_write_failure_blocks_motion(self):
        with patch.object(self.session.store, "put", side_effect=OSError("disk full")):
            report = self.session.delegate(manifest())
        self.assertEqual(report["status"], "needs_system2")
        self.assertTrue(self.session.executor.faulted)
        self.assertEqual(self.backend.calls, 0)


class WireSelectorFreezeTests(unittest.TestCase):
    def test_wire_rejects_oversized_npy_header_without_allocating(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False, "shape": (10**9,)})
        raw = json.dumps({"$array": base64.b64encode(stream.getvalue()).decode()}).encode()
        with self.assertRaises(ValueError):
            wire.loads(raw)

    def test_rpc_url_cannot_embed_credentials(self):
        with self.assertRaises(ValueError):
            RpcClient("http://user:password@example.com", "token")

    def test_contract_wire_roundtrip_and_unknown_type_rejection(self):
        obs = wire.loads(wire.dumps(scene()))
        self.assertEqual(obs.source, scene().source)
        np.testing.assert_equal(obs.cameras["head"].rgb, scene().cameras["head"].rgb)
        with self.assertRaises(ValueError):
            wire.loads(b'{"$type":"os.system","fields":{}}')
        with self.assertRaises(ValueError):
            wire.loads(b'{"x":NaN}')

    def test_jev_distribution_without_live_api_call(self):
        answer = {"answers": {"execution_parameter": {"type": "choice", "choice": "fine", "confidence": 0.8,
                  "probabilities": {"fine": 0.8, "normal": 0.2}}}}
        selector = JevSelector("test-model", transport=lambda payload: answer)
        self.assertEqual(selector.select({}, {"fine": "1mm", "normal": "10mm"}).option, "fine")
        answer["answers"]["execution_parameter"]["probabilities"]["fine"] = -1
        with self.assertRaises(ValueError):
            selector.select({}, {"fine": "1mm", "normal": "10mm"})

    def test_knowledge_freeze_transfer_and_tamper_detection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source/skills").mkdir(parents=True)
            (root / "source/skills/helper.py").write_text("x = 1")
            (root / "source/secret.txt").write_text("not transferred")
            digest = freeze_knowledge(root / "source", root / "frozen")
            self.assertEqual(stage_knowledge(root / "frozen", root / "test"), digest)
            self.assertFalse((root / "test/secret.txt").exists())
            self.assertEqual((root / "test/skills/helper.py").stat().st_mode & 0o222, 0)
            (root / "frozen/skills/helper.py").write_text("tampered")
            with self.assertRaises(ValueError):
                verify_knowledge(root / "frozen")

    def test_knowledge_symlink_cannot_escape_workspace(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "skills").mkdir()
            (root / "private.txt").write_text("not knowledge")
            (root / "skills/escape").symlink_to(root / "private.txt")
            with self.assertRaises(ValueError):
                freeze_knowledge(root, root / "frozen")


class RpcIntegrationTests(unittest.TestCase):
    def test_remote_stop_can_interrupt_a_waiting_selector(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingSelector:
            def select(self, state, options):
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test selector was not released")
                return Selection("normal", 1)
        backend = NativeFixture()
        worker = RoboDojoWorker(backend, profile(), "episode", native_action_limit=100)
        with TemporaryDirectory() as folder:
            session = Session(Executor(DirectRemote(worker)), SnapshotStore(Path(folder) / "snapshots"),
                              EventLog(Path(folder) / "events.jsonl"), selector=WaitingSelector())
            server = RpcServer(("127.0.0.1", 0), "token", session.tools(), concurrent_methods=("stop",))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            reports = []
            caller = threading.Thread(target=lambda: reports.append(RpcClient(url, "token").call("delegate", manifest=manifest())))
            try:
                caller.start()
                self.assertTrue(entered.wait(timeout=3))
                self.assertTrue(RpcClient(url, "token").call("stop")["stop_requested"])
                release.set()
                caller.join(timeout=3)
                self.assertFalse(caller.is_alive())
                self.assertEqual(reports[0]["status"], "stopped")
                self.assertEqual(backend.calls, 0)
            finally:
                release.set()
                caller.join(timeout=5)
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_remote_system2_to_harness_to_worker_with_mock_native(self):
        backend = NativeFixture()
        worker = RoboDojoWorker(backend, profile(), "episode", native_action_limit=100)
        server = RpcServer(("127.0.0.1", 0), "worker-token", worker.methods())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host_server = None
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            remote = RemoteRoboDojo(RpcClient(url, "worker-token"))
            with TemporaryDirectory() as folder:
                session = Session(Executor(remote), SnapshotStore(Path(folder) / "snapshots"),
                                  EventLog(Path(folder) / "events.jsonl"))
                host_server = RpcServer(("127.0.0.1", 0), "system2-token", session.tools())
                host_thread = threading.Thread(target=host_server.serve_forever, daemon=True)
                host_thread.start()
                system2 = RpcClient(f"http://127.0.0.1:{host_server.server_port}", "system2-token")
                self.assertEqual(system2.call("capabilities")["selector"], "FixedSelector")
                report = system2.call("delegate", manifest=manifest())
                self.assertEqual(report["status"], "program_completed")
                self.assertEqual(report["native_action_delta"], 6)
                host_server.shutdown()
                host_server.server_close()
                host_thread.join(timeout=2)
                host_server = None
            with self.assertRaises(urllib.error.HTTPError) as caught:
                RpcClient(url, "wrong-token").call("describe")
            self.assertEqual(caught.exception.code, 401)
            with self.assertRaises(urllib.error.HTTPError):
                RpcClient(url, "worker-token").call("reset")
        finally:
            if host_server:
                host_server.shutdown()
                host_server.server_close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
