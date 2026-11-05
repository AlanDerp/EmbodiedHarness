from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from embodied_harness.adapters.robodojo_native import RoboDojoNative, layout_catalog
from embodied_harness.adapters.robodojo import RoboDojoWorker
from test_runtime import NativeFixture, profile


class NativeTests(unittest.TestCase):
    def files(self, root):
        folder = root/"Assets/Eval_Layout/RoboDojo/arx_x5/0"
        folder.mkdir(parents=True)
        (folder/"stack_bowls_20.json").write_text("{}")
        (folder/"stack_bowls_3.json").write_text("{}")
        (root/"src/eval_client").mkdir(parents=True)
        (root/"src/eval_client/eval_env.py").write_text("# fixture")

    def test_native_catalog_is_numeric_ordinal_not_filename_suffix(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.files(root)
            catalog = layout_catalog(root, "stack_bowls")
            self.assertEqual(catalog[0]["filename"], "stack_bowls_3.json")
            self.assertEqual(catalog[1]["filename"], "stack_bowls_20.json")
            self.assertEqual(len(catalog[0]["layout_sha256"]), 64)
            with self.assertRaises(ValueError):
                layout_catalog(root, "../other")

    def test_lifecycle_reads_native_limit_and_never_invents_initial_success(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.files(root)
            fixture = NativeFixture()
            native = Mock()
            native.step_lim, native.episode_nums = 4, 1
            native.end_flag, native.success, native.take_action_cnt = [False], [True], [0]
            native.unstable_envs, native.interact = set(), False
            native.get_obs.side_effect = lambda: fixture.capture().public
            def take(action):
                fixture.step(action)
                native.take_action_cnt[0] = fixture.count
                native.end_flag[0] = fixture.count >= native.step_lim
                native.success[0] = False
            native.take_action.side_effect = take
            def start(instance):
                instance.native, instance.app = native, Mock()
            with patch.object(RoboDojoNative, "_start", start), \
                 patch.object(RoboDojoNative, "_restore_initial_robot", return_value=nullcontext()), \
                 patch.object(RoboDojoNative, "_check_robot_state"):
                backend = RoboDojoNative(root, "stack_bowls", 0, root/"run")
            self.assertFalse(backend.capture().success)
            self.assertEqual(backend.action_limit, 4)
            native.reset.assert_called_once_with(seed=[0])
            with self.assertRaises(ValueError):
                RoboDojoWorker(backend, profile(), "episode", native_action_limit=100)
            from embodied_harness.control import servo_action
            from embodied_harness.contracts import Pose
            worker = RoboDojoWorker(backend, profile(), "episode", native_action_limit=4)
            for i in range(2):
                obs = worker.observe()
                action = servo_action(obs, "left", Pose([.02,0,0],[1,0,0,0],"world"))
                receipt = worker.execute(action, str(i), worker.session_id)
            self.assertTrue(receipt.truncated)
            self.assertEqual(receipt.observation.source.native_action_count, 4)
            self.assertFalse(receipt.native_task_success)
            backend.close()
            backend.close()
            native.close.assert_called_once()

    def test_missing_layout_rejected_before_native_import(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                RoboDojoNative(folder, "stack_bowls", 0, Path(folder)/"run")


if __name__ == "__main__":
    unittest.main()
