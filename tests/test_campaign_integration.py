"""Real HTTP, real generated Python child, fixture model and fixture native physics."""

from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from embodied_harness.adapters.robodojo import RoboDojoWorker
from embodied_harness.agents import AgentService, RemoteCampaignAgent
from embodied_harness.campaign import Campaign, partition
from embodied_harness.deployment import ProvisionedEpisodes
from embodied_harness.evaluation import freeze_knowledge
from embodied_harness.rpc import RpcClient, RpcServer
from embodied_harness.services.sam3.server import Sam3Server
from test_runtime import NativeFixture, profile


class CampaignIntegrationTests(unittest.TestCase):
    def test_agent_harness_sam3_python_child_worker_campaign(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.dict(os.environ, {"WORKER_TOKEN": "worker"}))
            def serve(server):
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                def close():
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)
                stack.callback(close)
                return "http://127.0.0.1:" + str(server.server_port)

            class SegmentationFixture:
                model_version = "fixture-only-not-SAM3"
                calls = 0
                def segment(self, rgb, prompt, point=False):
                    self.calls += 1
                    return [(np.ones(rgb.shape[:2], bool), .99)]
            model = SegmentationFixture()
            sam3_url = serve(Sam3Server(("127.0.0.1", 0), model))
            allocations, natives = {}, []
            def open_episode(job):
                identity = uuid.uuid4().hex
                backend = NativeFixture(success_at=1)
                natives.append(backend)
                worker = RoboDojoWorker(backend, profile(), identity, native_action_limit=100, phase="test")
                server = RpcServer(("127.0.0.1", 0), "worker", worker.methods())
                url = serve(server)
                allocations[identity] = worker
                return {"allocation_id": identity, "worker_url": url, "layout_sha256": job["layout_sha256"]}
            def close_episode(allocation_id):
                allocations.pop(allocation_id)
                return {"closed": True}
            provisioner_url = serve(RpcServer(("127.0.0.1", 0), "host", {"open_episode": open_episode, "close_episode": close_episode}))
            source = Path("examples/ee_skill.py").read_text()
            source = source.replace('state["observation"] = get_observation()', '''state["observation"] = get_observation()
    obs = state["observation"]
    masks = segment_text(obs.source.episode_id, obs.source.snapshot_id, "head", "fixture object")
    if not masks.ok or not masks.data:
        raise RuntimeError("no segmentation")
    projection = mask_to_point_cloud(masks.data[0])
    if not projection.ok:
        raise RuntimeError("no depth")''')
            roles = []
            class ModelDriver:
                def respond(self, role, instructions, context):
                    roles.append(role)
                    if role == "propose":
                        return {"skills": {}, "skill_report": []}
                    program = source
                    if role == "generate":
                        program = source.replace('return step(action)', 'raise RuntimeError("fixture common failure")')
                    return {"goal": "move left EE", "subtasks": [{"task_id": "move", "goal": "move left EE", "source": program,
                            "parameters": {"arm": "left", "target": {"position": [.01, 0, 0], "quaternion_wxyz": [1, 0, 0, 0], "frame": "world"}}, "relation": None}]}
            agent_url = serve(RpcServer(("127.0.0.1", 0), "agent", AgentService(ModelDriver).methods()))
            episodes = ProvisionedEpisodes(RpcClient(provisioner_url, "host"), {"worker_token_env": "WORKER_TOKEN",
                    "sam3_url": sam3_url, "sam3_model_version": model.model_version, "selector": "fixed"})
            library = root / "library"
            library.mkdir()
            freeze_knowledge(library, root / "frozen")
            partitions = partition({i: {"layout_sha256": str(i)} for i in range(3)}, [0], [1], [2])
            runner = Campaign(root / "run", task="fixture", goal="move left EE", partitions=partitions,
                              frozen_knowledge=root / "frozen")
            result = runner.run(RemoteCampaignAgent(RpcClient(agent_url, "agent")), episodes)
            self.assertTrue(result["complete"])
            self.assertEqual(result["selected"]["development_scores"], [0, 1])
            self.assertEqual(result["heldout"]["native_successes"], 1)
            self.assertEqual(roles, ["generate", "repair", "propose"])
            self.assertGreater(model.calls, 0)
            self.assertFalse(allocations)
            self.assertEqual([native.calls for native in natives], [0, 0, 1, 1, 1])
