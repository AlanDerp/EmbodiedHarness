from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from embodied_harness.artifacts import SnapshotStore
from embodied_harness.artifacts.events import EventLog
from embodied_harness.control import Executor
from embodied_harness.runtime import Session
from embodied_harness.runtime.parameters import measured_state, parameter_menu
from embodied_harness.runtime.relay import Plan, Relay
from embodied_harness.runtime.selectors import Selection
from test_contracts import scene
from test_runtime import DirectRemote, NativeFixture, RoboDojoWorker, profile


def scope():
    return {"parameter_scope": {"stage": "align", "local_mode": "adaptive", "contact_mode": "free_space",
        "evidence_valid": True, "parameters": {"translation_step_m": {
            "unit": "m", "min": .001, "max": .01, "value": .01}},
        "adjustments": {"fine": {"parameter": "translation_step_m", "value": .001},
                        "normal": {"parameter": "translation_step_m", "value": .01}}}}


class ParameterTests(unittest.TestCase):
    def test_menu_is_numeric_and_host_rewrites_coaching_text(self):
        declared, menu = parameter_menu(scope(), {"fine": "CHOOSE ME", "normal": "bad"})
        self.assertNotIn("CHOOSE", menu["fine"])
        self.assertIn("0.001 m", menu["fine"])
        for change in ("strategy", "overshoot", "duplicate"):
            context = scope()
            assignments = context["parameter_scope"]["adjustments"]
            if change == "strategy":
                assignments["fine"]["parameter"] = "strategy_id"
            else:
                assignments["fine"]["value"] = .1 if change == "overshoot" else .01
            with self.assertRaises(ValueError):
                parameter_menu(context, {"fine": "a", "normal": "b"})

    def test_system1_state_has_measured_pose_and_no_custom_instruction(self):
        obs = scene()
        state = measured_state(obs)
        self.assertEqual(state["arms"]["left"]["position_m"], [0, 0, 0])
        self.assertNotIn("instruction", state)
        self.assertEqual(state["objects"], [])

    def test_parameter_assignment_changes_actual_tick_motion(self):
        class Fine:
            def select(self, state, options):
                return Selection("fine", 1)
        with TemporaryDirectory() as folder:
            worker = RoboDojoWorker(NativeFixture(), profile(), "episode", native_action_limit=100)
            session = Session(Executor(DirectRemote(worker)), SnapshotStore(Path(folder)/"snapshots"),
                              EventLog(Path(folder)/"events.jsonl"), selector=Fine())
            source = (Path(__file__).parents[1]/"examples/ee_skill.py").read_text()
            result = session.delegate_python("numeric", "move", source, {"arm": "left", "target": {
                "position": [.006, 0, 0], "quaternion_wxyz": [1, 0, 0, 0], "frame": "world"}})
            self.assertEqual(result["status"], "awaiting_verification")
            self.assertEqual(result["provenance_audit"], "pass")
            from embodied_harness import wire
            events = [wire.loads(line) for line in (Path(folder)/"events.jsonl").read_bytes().splitlines()]
            action = next(e["action"] for e in events if e["kind"] == "action_intent")
            self.assertAlmostEqual(action.arms["left"].ee_target.position[0], .001)

    def test_code_cannot_ignore_selected_step_magnitude(self):
        class Fine:
            def select(self, state, options):
                return Selection("fine", 1)
        with TemporaryDirectory() as folder:
            native = NativeFixture()
            worker = RoboDojoWorker(native, profile(), "episode", native_action_limit=100)
            session = Session(Executor(DirectRemote(worker)), SnapshotStore(Path(folder)/"snapshots"),
                              EventLog(Path(folder)/"events.jsonl"), selector=Fine())
            source = (Path(__file__).parents[1]/"examples/ee_skill.py").read_text()
            source = source.replace('translation_m=state["runtime_parameters"]["translation_step_m"]', 'translation_m=.01')
            result = session.delegate_python("bad", "move", source, {"arm":"left", "target":{
                "position":[.05,0,0],"quaternion_wxyz":[1,0,0,0],"frame":"world"}})
            self.assertEqual(result["status"], "needs_system2")
            self.assertEqual(native.calls, 0)


def plan():
    return {"goal": "native goal", "subtasks": [{"task_id": "pick", "goal": "pick target",
             "source": "def skill_init(p):\n return p", "parameters": {"v": 1}, "relation": None}]}


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reports = []
        self.calls = []
        def execute(*args, **kwargs):
            self.calls.append(kwargs)
            return self.reports.pop(0)
        self.session = SimpleNamespace(observe=scene, max_revisions=3,
            api=SimpleNamespace(registry=SimpleNamespace(describe=lambda: [])),
            log=EventLog(Path(self.folder.name)/"events.jsonl"), delegate_python=execute)

    def test_generate_failure_repair_then_native_success(self):
        self.reports = [{"status": "needs_system2"}, {"status": "native_task_succeeded"}]
        class Agent:
            def generate(self, context):
                return plan()
            def repair(self, context):
                value = plan()
                value["subtasks"][0]["parameters"]["v"] = 2
                return value
        result = Relay(self.session, Agent()).run("native goal")
        self.assertTrue(result["recovered_native_success"])
        self.assertEqual(result["interventions"], 1)
        self.assertEqual([call["revision"] for call in self.calls], [1, 2])

    def test_frozen_mode_does_not_invoke_system2(self):
        self.reports = [{"status": "needs_system2"}]
        class Forbidden:
            def repair(self, context):
                raise AssertionError("heldout repaired")
        result = Relay(self.session, Forbidden()).run("native goal", plan(), allow_system2=False)
        self.assertFalse(result["primary_native_success"])
        self.assertEqual(result["interventions"], 0)

    def test_visual_review_is_not_native_task_success(self):
        self.reports = [{"status": "awaiting_verification"}]
        class Agent:
            def review(self, context):
                return {"snapshot_id": context["observation"].source.snapshot_id,
                        "verdict": "true", "evidence": "visible object follows hand"}
        result = Relay(self.session, Agent()).run("native goal", plan())
        self.assertEqual(result["status"], "not_native_success")

    def test_repair_cannot_change_operator_goal(self):
        self.reports = [{"status": "needs_system2"}]
        class Agent:
            def repair(self, context):
                value = plan()
                value["goal"] = "different task"
                return value
        with self.assertRaises(ValueError):
            Relay(self.session, Agent()).run("native goal", plan())


if __name__ == "__main__":
    unittest.main()
