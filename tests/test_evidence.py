from dataclasses import replace
import unittest

import numpy as np

from embodied_harness.contracts import ErrorCode, PointCloud, Pose
from embodied_harness.geometry import estimate_geometry, matrix_to_pose, pose_to_matrix
from embodied_harness.perception.tracking import Track, associate, following_evidence
from test_contracts import scene
from test_runtime import profile


class EvidenceTests(unittest.TestCase):
    def geometry(self, obs, delta=(0, 0, 0)):
        points = np.array([[x, y, z] for x in (-.02, .02) for y in (-.01, .01) for z in (-.005, .005)])
        return estimate_geometry(PointCloud(obs.source, "head", "world", points + delta)).data

    def test_association_unique_ambiguous_missing_and_cross_episode(self):
        obs = scene()
        reference = Track("bowl", self.geometry(obs))
        candidate = self.geometry(obs, [.01, 0, 0])
        self.assertTrue(associate(reference, [candidate]).ok)
        self.assertEqual(associate(reference, [candidate, candidate]).error.code, ErrorCode.AMBIGUOUS)
        self.assertEqual(associate(reference, []).error.code, ErrorCode.NO_CANDIDATES)
        wrong = replace(candidate, source=replace(candidate.source, episode_id="other"))
        self.assertEqual(associate(reference, [wrong]).error.code, ErrorCode.STALE_EVIDENCE)

    def test_visible_following_stationary_and_stale_are_distinguished(self):
        before = scene()
        source = replace(before.source, snapshot_id="after", state_version=2, native_action_count=2)
        arms = dict(before.robot_state.arms)
        arms["left"] = replace(arms["left"], ee_pose=Pose([0, 0, .03], [1, 0, 0, 0], "world"))
        after = replace(before, source=source, robot_state=replace(before.robot_state, arms=arms))
        reference = Track("bowl", self.geometry(before))
        moved = Track("bowl", self.geometry(after, [0, 0, .03]))
        stationary = Track("bowl", self.geometry(after))
        self.assertEqual(following_evidence(reference, moved, before, after, "left").verdict, "true")
        self.assertEqual(following_evidence(reference, stationary, before, after, "left").verdict, "false")
        self.assertEqual(following_evidence(reference, reference, before, after, "left").verdict, "unknown")

    def test_pose_matrix_roundtrip_including_half_turn(self):
        for q in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [.5, .5, .5, .5]):
            pose = Pose([1, 2, 3], q, "world")
            recovered = matrix_to_pose(pose_to_matrix(pose), frame="world")
            self.assertAlmostEqual(abs(recovered.quaternion_wxyz @ pose.quaternion_wxyz), 1)
            np.testing.assert_allclose(recovered.position, pose.position)

    def test_tcp_target_conversion_uses_explicit_ee_offset(self):
        p = profile()
        transform = np.eye(4)
        transform[0, 3] = .15
        arms = dict(p.arms)
        arms["left"] = replace(arms["left"], T_ee_from_tcp=transform)
        p = replace(p, arms=arms)
        result = p.tcp_target_to_ee("left", Pose([.25, 0, 0], [1, 0, 0, 0], "world"))
        np.testing.assert_allclose(result.position, [.1, 0, 0])


if __name__ == "__main__":
    unittest.main()
