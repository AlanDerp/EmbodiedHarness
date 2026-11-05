import unittest

from embodied_harness.curriculum import Curriculum
import test_campaign as fixtures
from test_campaign import Agent, Episode, plan


class CurriculumTests(unittest.TestCase):
    setUp = fixtures.CampaignTests.setUp
    # Reuse the fixture setup only, without inheriting the campaign test methods.
    def test_final_library_sweep_keeps_original_results_and_resumes_without_reexecution(self):
        tasks = [{"task_id": key, "task": key, "goal": "move object", "partitions": self.partitions} for key in ("first", "second")]
        class SuiteAgent(Agent):
            def request(self, role, context):
                if role == "generate":
                    reply = super().request(role, context)
                    reply["reply"] = plan(2)
                    return reply
                return super().request(role, context)
        agent = SuiteAgent()
        original = agent.request
        proposals = []
        def request(role, context):
            if role == "propose":
                agent.proposal = len(proposals) == 1
                proposals.append(role)
            return original(role, context)
        agent.request = request
        def factory(job, workspace):
            episode = Episode(job, self.calls)
            run = episode.run
            def execute(selected):
                result = run(selected)
                if job["task"] == "first" and "attempt_01" in str(workspace) and job["job_id"].startswith("heldout"):
                    result.update(status="not_native_success", primary_native_success=False)
                    result["reports"][-1]["status"] = "needs_system2"
                return result
            episode.run = execute
            return episode
        root = self.root / "suite"
        report = Curriculum(root, tasks, self.frozen).run(agent, factory)
        self.assertEqual(report["attempt_01"]["first"]["heldout"]["native_successes"], 0)
        self.assertEqual(set(report["rerun_02"]), {"first"})
        self.assertEqual(report["rerun_02"]["first"]["heldout"]["native_successes"], 2)
        self.assertEqual(Curriculum(root, tasks, self.frozen).run(None, lambda *a: self.fail("replay")), report)
        with self.assertRaises(ValueError):
            Curriculum(root, list(reversed(tasks)), self.frozen)
