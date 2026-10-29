from pathlib import Path
from tempfile import TemporaryDirectory
import os
import sys
import unittest
from unittest.mock import patch

from embodied_harness.artifacts import SnapshotStore
from embodied_harness.artifacts.events import EventLog
from embodied_harness.control import Executor
from embodied_harness.programs import PersistentProgram, PythonSkill
from embodied_harness.runtime import Session
from test_contracts import scene
from test_runtime import NativeFixture, RoboDojoWorker, DirectRemote, profile




class ProgramTests(unittest.TestCase):
    def program(self, methods=None, timeout_s=3):
        program = PersistentProgram(methods or {"get_observation": scene}, timeout_s=timeout_s)
        self.addCleanup(program.close)
        return program

    def test_namespace_persists_and_observation_contract_crosses_pipe(self):
        program = self.program()
        self.assertTrue(program.execute("x = 10")["ok"])
        result = program.execute("x += 2\nobs = get_observation()", exports=("x", "obs"))
        self.assertEqual(result["exports"]["x"], 12)
        self.assertEqual(result["exports"]["obs"].source, scene().source)
        failure = program.execute("bad = (x for x in range(2))", exports=("bad",))
        self.assertFalse(failure["ok"])
        self.assertEqual(failure["exports"], {})
        self.assertIn("TypeError", failure["traceback"])

    def test_zero_action_gate_cannot_be_bypassed_by_catching_error_or_alias(self):
        calls = []
        program = self.program({"step": lambda action: calls.append(action)})
        result = program.execute("saved_step=step\ntry:\n saved_step(None)\nexcept Exception:\n pass")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "ActionBudgetViolation")
        self.assertEqual(calls, [])

    def test_tick_cannot_execute_two_actions(self):
        calls = []
        program = self.program({"step": lambda action: calls.append(action)})
        result = program.execute("step(None)\ntry:\n step(None)\nexcept Exception:\n pass", max_actions=1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["action_calls"], 1)
        self.assertEqual(len(calls), 1)

    def test_output_bounded_and_infinite_loop_terminated(self):
        program = self.program(timeout_s=1)
        result = program.execute("print('x' * 100000)")
        self.assertLessEqual(len(result["stdout"]), 8192)
        with self.assertRaises(TimeoutError):
            program.execute("while True: pass")
        self.assertIsNotNone(program.process.poll())

    def test_real_python_subprocess_uses_current_interpreter_and_does_not_inherit_api_tokens(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "fixture-secret"}):
            program = self.program()
        result = program.execute("import os, sys\nchild_pid=os.getpid()\n"
                                 "interpreter=sys.executable\nhas_token='TYPESAFE_API_KEY' in os.environ",
                                 exports=("child_pid", "interpreter", "has_token"))
        self.assertNotEqual(result["exports"]["child_pid"], os.getpid())
        self.assertEqual(result["exports"]["interpreter"], sys.executable)
        self.assertFalse(result["exports"]["has_token"])
        self.assertEqual(program.process.args[0], sys.executable)

    def test_forged_completion_cannot_hang_next_pipe_write(self):
        program = self.program(timeout_s=1)
        program.execute("import sys\nfrom embodied_harness import wire\n"
                        "sys.__stdout__.buffer.write(wire.dumps({'kind':'done','ok':True,'exports':{}})+b'\\n')\n"
                        "sys.__stdout__.buffer.flush()\nwhile True: pass")
        with self.assertRaises(TimeoutError):
            program.execute("x = '" + "x" * 100000 + "'")

    def test_python_hooks_integrate_with_session_and_cannot_claim_success(self):
        backend = NativeFixture()
        worker = RoboDojoWorker(backend, profile(), "episode", native_action_limit=100)
        source = '''
def skill_init(parameters):
    return {"status":"active"}
def skill_observe(state):
    return {"observation":get_observation()}
def skill_context(state):
    return {"status":state["status"]}
def skill_options(state):
    return {"normal":"one bounded motion"} if state["status"] == "active" else {}
def skill_tick(state, option):
    from embodied_harness.control import servo_action
    from embodied_harness.contracts import Pose
    obs = get_observation()
    result = step(servo_action(obs,"left",Pose([.01,0,0],[1,0,0,0],"world")))
    state["status"]="done"
    return result
'''
        with TemporaryDirectory() as folder:
            session = Session(Executor(DirectRemote(worker)), SnapshotStore(Path(folder) / "snapshots"),
                              EventLog(Path(folder) / "events.jsonl"))
            report = session.delegate_python("python-task", "move", source, {})
            self.assertEqual(report["status"], "awaiting_verification")
            self.assertEqual(backend.calls, 1)
            self.assertEqual(session.executor.submissions, 1)


if __name__ == "__main__":
    unittest.main()
