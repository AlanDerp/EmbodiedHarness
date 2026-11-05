import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from embodied_harness.deployment import NativeProvisioner


class ProvisionerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        layout = self.root / "Assets/Eval_Layout/RoboDojo/arx_x5/0/task_11.json"
        layout.parent.mkdir(parents=True)
        layout.write_text('{"fixture": true}')
        self.digest = hashlib.sha256(layout.read_bytes()).hexdigest()
        self.job = {"job_id": "dev_0", "task": "task", "scene_ref": 0, "layout_sha256": self.digest, "phase": "test"}
        self.token = patch.dict(os.environ, {"NATIVE_TEST_TOKEN": "secret"})
        self.token.start()
        self.addCleanup(self.token.stop)

    def provisioner(self):
        return NativeProvisioner(runtime_root=self.root, profile=self.root / "profile.json", root=self.root / "runs",
                                 python_command=["/fixture/isaaclab.sh", "-p"], worker_token_env="NATIVE_TEST_TOKEN")

    def test_verified_allocation_and_graceful_process_group_cleanup(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        rpc = Mock()
        rpc.call.return_value = {"episode_id": "allocation", "phase": "test"}
        with patch("uuid.uuid4", return_value=Mock(hex="allocation")), patch("subprocess.Popen", return_value=process) as popen, \
                patch("embodied_harness.rpc.RpcClient", return_value=rpc), patch("os.killpg") as kill:
            provisioner = self.provisioner()
            result = provisioner.open_episode(self.job)
            self.assertEqual(result["layout_sha256"], self.digest)
            command = popen.call_args.args[0]
            self.assertEqual(command[:2], ["/fixture/isaaclab.sh", "-p"])
            self.assertNotIn("secret", command)
            config = json.loads((self.root / "runs/allocation/native-config.json").read_text())
            self.assertEqual(config["scene_ref"], 0)
            with self.assertRaises(RuntimeError):
                provisioner.open_episode(self.job)
            self.assertTrue(provisioner.close_episode("allocation")["closed"])
            kill.assert_called_once()
            process.wait.assert_called_once_with(timeout=30)
            self.assertFalse(provisioner.active)

    def test_layout_mismatch_is_rejected_before_process_start(self):
        with patch("subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                self.provisioner().open_episode({**self.job, "layout_sha256": "wrong"})
            popen.assert_not_called()

    def test_startup_exit_is_terminal_and_allocation_is_released(self):
        process = Mock()
        process.poll.return_value = 1
        with patch("subprocess.Popen", return_value=process):
            provisioner = self.provisioner()
            with self.assertRaises(RuntimeError):
                provisioner.open_episode(self.job)
            self.assertFalse(provisioner.active)
