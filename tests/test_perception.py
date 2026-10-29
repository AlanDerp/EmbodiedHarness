from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from embodied_harness.artifacts import SnapshotStore
from embodied_harness.contracts import ErrorCode, HarnessError, Result
from embodied_harness.perception import Perception
from test_contracts import scene, segment


class FakeProvider:
    """Protocol fixture, not model inference."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def segment_text(self, observation, camera_id, query):
        self.calls.append((observation.source, camera_id, query))
        return self.answer


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = SnapshotStore(self.temporary.name)

    def test_roundtrip_preserves_all_public_state(self):
        obs = scene()
        depth = np.ones((3, 4))
        depth[0, 0] = np.nan
        obs = replace(obs, cameras={"head": replace(obs.cameras["head"], depth_m=depth)})
        self.store.put(obs)
        restored = self.store.get("episode", "snapshot")
        self.assertEqual(restored.source, obs.source)
        self.assertEqual(restored.instruction, obs.instruction)
        self.assertEqual(restored.cameras["head"].rgb_sha256, obs.cameras["head"].rgb_sha256)
        np.testing.assert_equal(restored.cameras["head"].depth_m, depth)
        np.testing.assert_equal(restored.cameras["head"].intrinsics, obs.cameras["head"].intrinsics)
        np.testing.assert_equal(restored.cameras["head"].T_world_from_camera_optical,
                                obs.cameras["head"].T_world_from_camera_optical)
        for name, arm in restored.robot_state.arms.items():
            self.assertEqual(arm.joint_names, obs.robot_state.arms[name].joint_names)
            np.testing.assert_equal(arm.joint_positions, obs.robot_state.arms[name].joint_positions)
            np.testing.assert_equal(arm.ee_pose.quaternion_wxyz, [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            restored.cameras["head"].rgb.setflags(write=True)

    def test_duplicate_never_overwrites_evidence(self):
        folder = self.store.put(scene())
        original = (folder / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.store.put(replace(scene(), instruction="different goal"))
        self.assertEqual((folder / "manifest.json").read_bytes(), original)

    def test_ids_are_not_paths_and_episodes_are_separate(self):
        obs = scene()
        a = replace(obs, source=replace(obs.source, episode_id="../outside", snapshot_id="../../a"))
        folder = self.store.put(a)
        self.assertEqual(folder.parent, Path(self.temporary.name))
        self.assertEqual(self.store.get("../outside", "../../a").source, a.source)
        self.store.put(obs)
        with self.assertRaises(FileNotFoundError):
            self.store.get("other", "snapshot")

    def test_corruption_and_identity_mismatch_are_rejected(self):
        folder = self.store.put(scene())
        payload = folder / "arrays.npz"
        payload.write_bytes(payload.read_bytes() + b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.get("episode", "snapshot")

    def test_manifest_schema_and_identity_are_checked(self):
        folder = self.store.put(scene())
        path = folder / "manifest.json"
        original = json.loads(path.read_text())
        wrong = dict(original, schema_version=99)
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "schema"):
            self.store.get("episode", "snapshot")
        original["source"]["episode_id"] = "other"
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.store.get("episode", "snapshot")

    def test_capacity_and_read_limits(self):
        small = SnapshotStore(self.temporary.name, max_bytes=1)
        with self.assertRaises(ValueError):
            small.put(scene())
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])
        limited = SnapshotStore(self.temporary.name, max_snapshots=1)
        limited.put(scene())
        with self.assertRaises(ValueError):
            limited.put(replace(scene(), source=replace(scene().source, snapshot_id="second")))
        with self.assertRaises(ValueError):
            small.get("episode", "snapshot")

    def test_failed_write_leaves_no_committed_or_temporary_snapshot(self):
        with patch("embodied_harness.artifacts.snapshots.np.savez", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.put(scene())
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = SnapshotStore(self.temporary.name)
        self.obs = scene()
        self.store.put(self.obs)

    def localize(self, response, **kwargs):
        provider = FakeProvider(response)
        result = Perception(self.store, provider).localize_object(
            "episode", "snapshot", "head", "bowl", **kwargs)
        self.assertEqual(provider.calls, [(self.obs.source, "head", "bowl")])
        return result

    def test_saved_snapshot_to_geometry(self):
        result = self.localize(Result((segment(self.obs),)))
        self.assertTrue(result.ok)
        self.assertEqual(result.data.selected_index, 0)
        self.assertEqual(result.data.geometry.point_count, 12)
        self.assertEqual(result.data.geometry.source, self.obs.source)
        self.assertTrue(result.data.geometry.orientation_ambiguous)  # visible plane

    def test_multiple_candidates_require_explicit_selection(self):
        seg = segment(self.obs)
        candidates = (seg, replace(seg, score=0.1))
        result = self.localize(Result(candidates))
        self.assertEqual(result.error.code, ErrorCode.AMBIGUOUS)
        self.assertEqual(len(result.data.candidates), 2)
        self.assertIsNone(result.data.projection)
        chosen = self.localize(Result(candidates), candidate_index=1)
        self.assertTrue(chosen.ok)
        self.assertEqual(chosen.data.selected_index, 1)

    def test_no_candidates_and_missing_provider(self):
        self.assertEqual(self.localize(Result(())).error.code, ErrorCode.NO_CANDIDATES)
        result = Perception(self.store, None).localize_object("episode", "snapshot", "head", "bowl")
        self.assertEqual(result.error.code, ErrorCode.NOT_CONFIGURED)

    def test_provider_failure_keeps_partial_candidates(self):
        error = HarnessError(ErrorCode.TIMEOUT, "provider timeout")
        result = self.localize(Result((segment(self.obs),), error))
        self.assertEqual(result.error, error)
        self.assertEqual(len(result.data.candidates), 1)
        self.assertIsNone(result.data.geometry)

    def test_invalid_provider_response_is_rejected(self):
        seg = segment(self.obs)
        for response in ("bad", Result([seg]), Result((object(),)),
                         Result((replace(seg, source=replace(seg.source, snapshot_id="old")),)),
                         Result((replace(seg, camera_id="wrist"),)),
                         Result((replace(seg, prompt="different object"),)),
                         Result((replace(seg, input_rgb_sha256="0" * 64),))):
            with self.subTest(response=type(response)):
                self.assertEqual(self.localize(response).error.code, ErrorCode.INVALID_RESPONSE)

    def test_low_depth_coverage_and_zero_depth_preserve_evidence(self):
        for count, code in ((0, ErrorCode.NO_VALID_DEPTH), (3, ErrorCode.INSUFFICIENT_DEPTH)):
            with TemporaryDirectory() as folder:
                depth = np.zeros((3, 4))
                depth.ravel()[:count] = 1
                obs = replace(self.obs, cameras={"head": replace(self.obs.cameras["head"], depth_m=depth)})
                store = SnapshotStore(folder)
                store.put(obs)
                result = Perception(store, FakeProvider(Result((segment(obs),)))).localize_object(
                    "episode", "snapshot", "head", "bowl")
                self.assertEqual(result.error.code, code)
                self.assertEqual(result.data.projection.valid_depth_count, count)
                self.assertIsNone(result.data.geometry)

    def test_degenerate_geometry_failure_keeps_point_cloud(self):
        mask = np.zeros((3, 4), bool)
        mask[0, :] = True
        result = self.localize(Result((replace(segment(self.obs), mask=mask),)))
        self.assertEqual(result.error.code, ErrorCode.GEOMETRY_FAILED)
        self.assertEqual(result.data.projection.valid_depth_count, 4)
        self.assertIsNone(result.data.geometry)

    def test_invalid_call_does_not_invoke_provider(self):
        provider = FakeProvider(Result((segment(self.obs),)))
        pipeline = Perception(self.store, provider)
        for kwargs in ({"candidate_index": -1}, {"min_valid_depth_fraction": 2}):
            with self.assertRaises(ValueError):
                pipeline.localize_object("episode", "snapshot", "head", "bowl", **kwargs)
        with self.assertRaises(ValueError):
            pipeline.localize_object("episode", "snapshot", "missing", "bowl")
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
