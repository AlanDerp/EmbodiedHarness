import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from embodied_harness.artifacts import SnapshotStore
from embodied_harness.diagnostics import check_sam3, doctor
from embodied_harness.rpc import RpcServer
from embodied_harness.services.sam3.server import Sam3Server
from test_contracts import scene


class DiagnosticsTests(unittest.TestCase):
    def serve(self, server):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.addCleanup(close)
        return "http://127.0.0.1:" + str(server.server_port)

    def test_snapshot_acceptance_calls_text_and_point_and_rejects_wrong_weight_pin(self):
        class Backend:
            model_version = "sam3:code:" + "a"*64 + ":checkpoint:" + "b"*64
            calls = []
            def segment(self, rgb, prompt, point=False):
                self.calls.append(point)
                return [(np.ones(rgb.shape[:2], bool), .9)]
        backend = Backend()
        url = self.serve(Sam3Server(("127.0.0.1", 0), backend))
        with tempfile.TemporaryDirectory() as folder:
            store = SnapshotStore(Path(folder) / "snapshots")
            store.put(scene())
            report = check_sam3(url, store.root, "episode", "snapshot", "head", "fixture", "b"*64, Path(folder)/"audit.jsonl")
            self.assertTrue(report["passed"])
            self.assertEqual(backend.calls, [False, True])
            with self.assertRaises(ValueError):
                check_sam3(url, store.root, "episode", "snapshot", "head", "fixture", "c"*64, Path(folder)/"audit.jsonl")

    def test_doctor_is_read_only_and_requires_idle_commissioned_native_service(self):
        class Backend:
            model_version = "sam3:code:" + "a"*64 + ":checkpoint:" + "b"*64
        sam3 = self.serve(Sam3Server(("127.0.0.1", 0), Backend()))
        agent = self.serve(RpcServer(("127.0.0.1", 0), "token", {"describe": lambda: {
            "protocol": "aspire-role-agent-v1", "fresh_conversation_per_request": True}}))
        native = {"protocol": "robodojo-native-provisioner-v1", "commissioned": True,
                  "runtime_present": True, "active_allocations": 0}
        provisioner = self.serve(RpcServer(("127.0.0.1", 0), "token", {"describe": lambda: native}))
        config = {"harness": {"sam3_url": sam3, "sam3_model_version": Backend.model_version},
                  "agent_url": agent, "agent_token_env": "CHECK_TOKEN", "provisioner_url": provisioner,
                  "provisioner_token_env": "CHECK_TOKEN"}
        with patch.dict(os.environ, {"CHECK_TOKEN": "token"}):
            self.assertTrue(doctor(config)["passed"])
            native["active_allocations"] = 1
            with self.assertRaises(ValueError):
                doctor(config)
