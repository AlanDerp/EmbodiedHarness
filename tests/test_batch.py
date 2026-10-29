import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embodied_harness.batch import BatchRunner
from embodied_harness.evaluation import freeze_knowledge


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "source/skills").mkdir(parents=True)
        (self.root / "source/skills/helper.py").write_text("pass")
        freeze_knowledge(self.root / "source", self.root / "frozen")
        self.jobs = [{"job_id": f"scene_{i}", "task": "fixture", "scene_ref": i} for i in range(2)]

    def runner(self):
        return BatchRunner(self.root / "batch", self.jobs, self.root / "frozen")

    def factory(self, same_agent=False, fail=False):
        def open_episode(job, workspace):
            self.assertTrue((workspace / "skills/helper.py").is_file())
            class Driver:
                worker_session_id = "worker-" + job["job_id"]
                agent_session_id = "same" if same_agent else "agent-" + job["job_id"]
                def run(self):
                    if fail:
                        raise RuntimeError("fixture failure")
                    return {"primary_native_success": True, "recovered_native_success": False, "interventions": 0}
                def close(self):
                    pass
            return Driver()
        return open_episode

    def test_fresh_scenes_keep_complete_denominator(self):
        summary = self.runner().run(self.factory())
        self.assertEqual(summary["denominator"], 2)
        self.assertEqual(summary["primary_native_successes"], 2)

    def test_cross_scene_agent_reuse_is_rejected(self):
        summary = self.runner().run(self.factory(same_agent=True))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["primary_native_successes"], 1)

    def test_failed_attempts_are_not_rerun_on_resume(self):
        summary = self.runner().run(self.factory(fail=True))
        self.assertEqual(summary["failed"], 2)
        def forbidden(*args):
            self.fail("failed scene was retried")
        summary = self.runner().run(forbidden)
        self.assertEqual(summary["failed"], 2)

    def test_interrupted_attempt_is_preserved(self):
        runner = self.runner()
        runner.ledger["results"]["scene_0"]["status"] = "running"
        runner.path.write_text(json.dumps(runner.ledger))
        summary = self.runner().run(self.factory())
        self.assertEqual(summary["interrupted"], 1)
        self.assertEqual(summary["primary_native_successes"], 1)

    def test_resume_cannot_change_scene_plan(self):
        self.runner()
        changed = [dict(job) for job in self.jobs]
        changed[0]["scene_ref"] = 999
        with self.assertRaises(ValueError):
            BatchRunner(self.root / "batch", changed, self.root / "frozen")
