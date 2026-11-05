import json
from pathlib import Path
import tempfile
import unittest
import uuid

from embodied_harness.agents import AgentService
from embodied_harness.campaign import Campaign, partition
from embodied_harness.evaluation import freeze_knowledge
from embodied_harness.runtime.relay import Plan


def plan(version):
    return {"goal": "move object", "subtasks": [{"task_id": "move", "goal": "move object",
            "source": f"value = {version}", "parameters": {}, "relation": None}]}


class Agent:
    def __init__(self, proposal=True):
        self.calls, self.proposal = [], proposal

    def request(self, role, context):
        self.calls.append((role, context))
        if role == "propose":
            reply = {"skills": {"skills/grasp.md": "Observe with SAM3 before approaching; retry only with fresh geometry."} if self.proposal else {}, "skill_report": [{"skill": key, "verdict": "helpful", "evidence": "Used on development"} for key in context["library"]]}
        else:
            reply = plan(1 if role == "generate" else 2)
        return {"agent_session_id": uuid.uuid4().hex, "reply": reply}


class Episode:
    def __init__(self, job, calls, *, bad=False):
        self.job, self.calls, self.bad = job, calls, bad
        self.worker_session_id, self.agent_session_id = uuid.uuid4().hex, uuid.uuid4().hex

    def context(self):
        return {"observation": "public RGB-D fixture", "public_api": []}

    def run(self, selected):
        self.calls.append((self.job["job_id"], selected.fingerprint()))
        success = "value = 2" in selected.subtasks[0].source
        if self.job["job_id"].startswith("unseen"):
            success = False  # This must never trigger repair or change the selected version.
        return {"status": "native_task_succeeded" if success else "not_native_success",
                "primary_native_success": success, "recovered_native_success": False,
                "interventions": 0, "system2_enabled": False, "plan_sha256": selected.fingerprint(),
                "selected_plan": selected.to_dict(), "reports": [{"provenance_audit": "rejected" if self.bad else "pass",
                "status": "native_task_succeeded" if success else "needs_system2"}]}

    def close(self):
        pass


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "library"
        self.source.mkdir()
        self.frozen = self.root / "frozen"
        freeze_knowledge(self.source, self.frozen)
        catalog = {i: {"layout_sha256": f"{i:064x}"} for i in range(5)}
        self.partitions = partition(catalog, [0, 1], [2], [3, 4])
        self.calls = []

    def campaign(self):
        return Campaign(self.root / "campaign", task="task", goal="move object",
                        partitions=self.partitions, frozen_knowledge=self.frozen)

    def factory(self, job, workspace):
        return Episode(job, self.calls)

    def test_full_matrix_selection_unseen_once_frozen_evaluation_and_independent_promotion(self):
        agent = Agent()
        runner = self.campaign()
        result = runner.run(agent, self.factory)
        self.assertTrue(result["complete"])
        self.assertEqual(result["selected"]["development_scores"], [0, 2])
        self.assertEqual(result["unseen"], {"denominator": 1, "native_successes": 0})
        self.assertEqual(result["heldout"]["native_successes"], 2)
        self.assertEqual(result["promotion"]["status"], "verified")
        self.assertEqual([key for key, _ in self.calls], ["dev_v0_0", "dev_v0_1", "dev_v1_0", "dev_v1_1",
                                                        "unseen_0", "heldout_0", "heldout_1", "verify_0_0", "verify_0_1"])
        self.assertEqual([role for role, _ in agent.calls], ["generate", "repair", "propose", "verify"])
        verifier = agent.calls[-1][1]
        self.assertEqual(set(verifier), {"goal", "public", "library"})
        self.assertNotIn("selected_plan", verifier)
        self.assertIn("skills/grasp.md", verifier["library"])
        # Resume must use recorded agent replies/results, even after all evaluation completes.
        class NoCalls:
            def request(self, *args):
                raise AssertionError("replayed agent")
        self.assertEqual(self.campaign().run(NoCalls(), lambda *a: self.fail("replayed episode")), result)

    def test_overlap_and_duplicate_layout_content_rejected(self):
        catalog = {0: {"layout_sha256": "a"}, 1: {"layout_sha256": "b"}, 2: {"layout_sha256": "a"}}
        with self.assertRaises(ValueError):
            partition(catalog, [0], [1], [0])
        with self.assertRaises(ValueError):
            partition(catalog, [0], [1], [2])

    def test_interrupted_or_failed_trial_never_replayed(self):
        runner = self.campaign()
        def fail(job, workspace):
            raise RuntimeError("provisioner disconnected")
        with self.assertRaises(RuntimeError):
            runner.run(Agent(), fail)
        with self.assertRaises(RuntimeError):
            self.campaign().run(Agent(), lambda *a: self.fail("must not reprovision"))
        runner.state["trials"]["preview"]["status"] = "running"
        runner.save()
        restored = self.campaign()
        self.assertEqual(restored.state["trials"]["preview"]["status"], "interrupted")

    def test_rejected_provenance_stops_before_unseen(self):
        runner = self.campaign()
        with self.assertRaises(ValueError):
            runner.run(Agent(), lambda job, ws: Episode(job, self.calls, bad=True))
        self.assertIsNone(runner.state["selected"])
        self.assertFalse(any(key.startswith("unseen") for key in runner.state["trials"]))

    def test_reused_worker_rejected(self):
        def reuse(job, workspace):
            episode = Episode(job, self.calls)
            episode.worker_session_id = "reused"
            return episode
        with self.assertRaises(ValueError):
            self.campaign().run(Agent(), reuse)

    def test_skill_path_escape_is_rejected(self):
        class BadProposal(Agent):
            def request(self, role, context):
                if role == "propose":
                    return {"agent_session_id": uuid.uuid4().hex, "reply": {"skills": {"skills/../../escape.md": "bad"}, "skill_report": []}}
                return super().request(role, context)
        with self.assertRaises(ValueError):
            self.campaign().run(BadProposal(), self.factory)
        self.assertFalse((self.root / "campaign" / "promoted").exists())

    def test_no_promotion_when_verifier_cannot_reconstruct_success(self):
        class FailedVerifier(Agent):
            def request(self, role, context):
                if role == "verify":
                    self.calls.append((role, context))
                    return {"agent_session_id": uuid.uuid4().hex, "reply": plan(1)}
                return super().request(role, context)
        agent = FailedVerifier()
        result = self.campaign().run(agent, self.factory)
        self.assertEqual(result["promotion"]["status"], "rejected")
        self.assertEqual(sum(role == "verify" for role, _ in agent.calls), 2)
        self.assertFalse((self.root / "campaign" / "promoted").exists())

    def test_verifier_service_has_fresh_driver_and_rejects_solution_context(self):
        created = []
        class Driver:
            def __init__(self):
                created.append(self)
            def respond(self, role, prompt, context):
                self.prompt = prompt
                return plan(2)
        service = AgentService(Driver)
        context = {"goal": "move object", "public": {}, "library": {}}
        first = service.respond("verify", context)
        second = service.respond("verify", context)
        self.assertNotEqual(first["agent_session_id"], second["agent_session_id"])
        self.assertEqual(len(created), 2)
        self.assertIn("Exactly one bounded native submission", created[0].prompt)
        with self.assertRaises(ValueError):
            service.respond("verify", {**context, "selected_plan": plan(2)})
