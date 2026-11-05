from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embodied_harness.api import CapApi
from embodied_harness.artifacts import SnapshotStore
from embodied_harness.artifacts.events import EventLog
from embodied_harness.control import Executor
from embodied_harness.programs.policy import validate
from embodied_harness.registry import ToolRegistry
from embodied_harness.runtime import Session
from test_runtime import DirectRemote, NativeFixture, RoboDojoWorker, profile


class PolicyTests(unittest.TestCase):
    def test_reject_privileged_imports_files_reflection_and_native_state(self):
        for code in ("import os", "import numpy", "from math import sqrt", "open('x')", "eval('1')",
                     "a.__class__", "obs.env", "obs.reward", "image.tofile('x')", "getattr(a,'env')"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate(code)
        validate("import math\ndef f(x):\n return math.sqrt(x)\nx=f(4)")

    def test_registry_rejects_duplicates_and_unknown_arguments(self):
        registry = ToolRegistry().register("read", lambda x: x)
        with self.assertRaises(ValueError):
            registry.register("read", lambda: None)
        with self.assertRaises(TypeError):
            registry.invoke("read", {"unknown": 2})
        self.assertEqual(registry.invoke("read", {"x": 4}), 4)

    def test_public_api_is_discoverable_and_traced(self):
        with TemporaryDirectory() as folder:
            worker = RoboDojoWorker(NativeFixture(), profile(), "episode", native_action_limit=10)
            session = Session(Executor(DirectRemote(worker)), SnapshotStore(Path(folder)/"snapshots"),
                              EventLog(Path(folder)/"events.jsonl"))
            obs = session.api.invoke("get_observation", {})
            self.assertEqual(obs.source.episode_id, "episode")
            self.assertIn("segment_point", session.tools())
            self.assertIn("tool_call", (Path(folder)/"events.jsonl").read_text())
            self.assertTrue(session.tools()["capabilities"]()["public_tools"])


if __name__ == "__main__":
    unittest.main()
