import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from embodied_harness.categories import categories, select_category
from embodied_harness.cli import main
from embodied_harness.codex_driver import CodexDriver
from embodied_harness.evaluation import freeze_knowledge
from test_contracts import scene


class CategoryTests(unittest.TestCase):
    def setup_registry(self, root):
        source = root / "empty"
        (source / "skills").mkdir(parents=True)
        (source / "lessons").mkdir()
        freeze_knowledge(source, root / "initial")
        manifest = {"task": "PickBowl", "partitions": {
            name: [{"scene_ref": index, "layout_sha256": str(index) * 64}]
            for index, name in enumerate(("development", "unseen", "heldout"))}}
        (root / "partition.json").write_text(json.dumps(manifest))
        config = {"goal": "Pick up the bowl", "partition_manifest": "partition.json",
                  "frozen_knowledge": "initial", "run_dir": "run",
                  "agent_url": "http://127.0.0.1:8123", "agent_token_env": "AGENT_TOKEN",
                  "provisioner_url": "http://127.0.0.1:8122", "provisioner_token_env": "NATIVE_TOKEN",
                  "harness": {"sam3_url": "http://127.0.0.1:8114", "worker_token_env": "WORKER_TOKEN",
                              "sam3_model_version": "sam3:code:" + "a" * 64 + ":checkpoint:" + "b" * 64}}
        (root / "campaign.json").write_text(json.dumps(config))
        registry = root / "categories.json"
        registry.write_text(json.dumps({"categories": {"bowls": {
            "task": "PickBowl", "config": "campaign.json"}}}))
        return registry, manifest

    def test_dry_run_resolves_paths_without_writes_or_services(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry, _ = self.setup_registry(root)
            config, report = select_category(registry, "bowls")
            self.assertEqual(config["run_dir"], str((root / "run").resolve()))
            self.assertEqual(report["partitions"], {"development": 1, "unseen": 1, "heldout": 1})
            with patch.object(sys, "argv", ["embodied-harness", "test-category", "--registry",
                                           str(registry), "--category", "bowls", "--dry-run"]), \
                    patch("embodied_harness.diagnostics.doctor") as doctor, \
                    patch("embodied_harness.cli.run_campaign") as run, patch("sys.stdout", new_callable=io.StringIO):
                main()
                doctor.assert_not_called()
                run.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_task_overlap_unknown_category_and_modified_library_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry, manifest = self.setup_registry(root)
            self.assertEqual(list(categories(registry)), ["bowls"])
            with self.assertRaisesRegex(ValueError, "unknown category"):
                select_category(registry, "missing")
            manifest["task"] = "OtherTask"
            (root / "partition.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "differs"):
                select_category(registry, "bowls")
            manifest["task"] = "PickBowl"
            manifest["partitions"]["unseen"] = manifest["partitions"]["development"]
            (root / "partition.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlap"):
                select_category(registry, "bowls")
            manifest["partitions"]["unseen"] = [{"scene_ref": 1, "layout_sha256": "1" * 64}]
            (root / "partition.json").write_text(json.dumps(manifest))
            (root / "initial/skills").mkdir(exist_ok=True)
            (root / "initial/skills/injected.md").write_text("changed")
            with self.assertRaises(ValueError):
                select_category(registry, "bowls")

    def test_launch_preflight_failure_never_enters_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            registry, _ = self.setup_registry(Path(folder))
            with patch.object(sys, "argv", ["embodied-harness", "test-category", "--registry",
                                           str(registry), "--category", "bowls"]), \
                    patch("embodied_harness.diagnostics.doctor", side_effect=ValueError("not ready")), \
                    patch("embodied_harness.cli.run_campaign") as run:
                with self.assertRaisesRegex(ValueError, "not ready"):
                    main()
                run.assert_not_called()

    def test_launch_preflight_then_unchanged_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            registry, _ = self.setup_registry(Path(folder))
            calls = []
            with patch.object(sys, "argv", ["embodied-harness", "test-category", "--registry",
                                           str(registry), "--category", "bowls"]), \
                    patch("embodied_harness.diagnostics.doctor", side_effect=lambda cfg: calls.append("doctor")), \
                    patch("embodied_harness.cli.run_campaign", side_effect=lambda cfg: calls.append("campaign")), \
                    patch("sys.stdout", new_callable=io.StringIO):
                main()
            self.assertEqual(calls, ["doctor", "campaign"])


class CodexDriverTests(unittest.TestCase):
    def test_fresh_multimodal_requests_and_no_service_credentials(self):
        roots = []

        def run(command, **kwargs):
            root = Path(kwargs["cwd"])
            roots.append(root)
            self.assertIn("--ephemeral", command)
            self.assertIn("--ignore-user-config", command)
            self.assertEqual(command[-1], "-")
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertNotIn("EMBODIED_AGENT_TOKEN", kwargs["env"])
            self.assertNotIn("TYPESAFE_API_KEY", kwargs["env"])
            image = Path(command[command.index("--image") + 1])
            self.assertTrue(image.exists())
            self.assertIn('"image_attachment": "rgb_0.png"', kwargs["input"])
            self.assertNotIn("$array", kwargs["input"])
            self.assertTrue(np.array_equal(np.load(root / "array_0.npy"), np.ones((10, 10))))
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text('{"skills": {}, "skill_report": []}')

        with patch.dict(os.environ, {"EMBODIED_AGENT_TOKEN": "private", "TYPESAFE_API_KEY": "private"}), \
                patch("embodied_harness.codex_driver.subprocess.run", side_effect=run):
            driver = CodexDriver(model="pinned-model")
            for _ in range(2):
                reply = driver.respond("propose", "contract", {"observation": scene(), "depth": np.ones((10, 10))})
                self.assertEqual(reply, {"skills": {}, "skill_report": []})
        self.assertNotEqual(*roots)
        self.assertTrue(all(not root.exists() for root in roots))

    def test_model_error_and_invalid_json_are_not_retried(self):
        driver = CodexDriver(model="pinned-model")
        with patch("embodied_harness.codex_driver.subprocess.run", side_effect=subprocess.TimeoutExpired("codex", 1)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                driver.respond("generate", "contract", {})
            self.assertEqual(run.call_count, 1)

        def malformed(command, **kwargs):
            Path(command[command.index("--output-last-message") + 1]).write_text("```json\n{}\n```")
        with patch("embodied_harness.codex_driver.subprocess.run", side_effect=malformed):
            with self.assertRaises(json.JSONDecodeError):
                driver.respond("generate", "contract", {})
        with self.assertRaises(ValueError):
            CodexDriver(model=None)
