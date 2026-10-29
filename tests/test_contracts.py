from dataclasses import replace
import subprocess
import sys
import unittest

import numpy as np

from embodied_harness.contracts import (
    ArmCommand, ArmState, CameraObservation, ErrorCode, HarnessError, Observation,
    ObservationSource, PointCloud, Pose, Result, RobotAction, RobotState, Segmentation,
)


def scene():
    source = ObservationSource("episode", "snapshot", 0, 1, 1, 12.0)
    camera = CameraObservation("head", np.zeros((3, 4, 3), dtype=np.uint8),
                               np.ones((3, 4)), np.eye(3), np.eye(4))
    pose = Pose([0, 0, 0], [1, 0, 0, 0], "world")
    arm = ArmState(("j1", "j2"), [0, 0], pose, 1.0)
    robot = RobotState("arx_x5", {"left": arm, "right": arm})
    return Observation(source, {"head": camera}, robot, "pick up the bowl")


def segment(obs):
    return Segmentation(obs.source, "head", np.ones((3, 4), dtype=bool),
                        [0, 0, 4, 3], 0.9, "bowl", "sam3", "checkpoint-1",
                        obs.cameras["head"].rgb_sha256)


class ContractTests(unittest.TestCase):
    def test_segmentation_binds_exact_snapshot_and_rgb(self):
        obs = scene()
        seg = segment(obs)
        seg.validate_against(obs)
        with self.assertRaises(ValueError):
            seg.validate_against(replace(obs, source=replace(obs.source, snapshot_id="other")))
        rgb = np.ones((3, 4, 3), dtype=np.uint8)
        changed = replace(obs.cameras["head"], rgb=rgb)
        with self.assertRaises(ValueError):
            seg.validate_against(replace(obs, cameras={"head": changed}))

    def test_state_identity_is_not_request_identity(self):
        source = scene().source
        self.assertTrue(source.same_state(replace(source, snapshot_id="new", captured_monotonic_s=20)))
        for field in ("episode_id", "reset_epoch", "state_version", "native_action_count"):
            value = "other" if field == "episode_id" else 2
            self.assertFalse(source.same_state(replace(source, **{field: value})))

    def test_arrays_and_mappings_are_defensively_immutable(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        camera = replace(scene().cameras["head"], rgb=rgb)
        rgb[:] = 255
        self.assertFalse(camera.rgb.any())
        with self.assertRaises(ValueError):
            camera.rgb.setflags(write=True)
        cameras = {"head": camera}
        obs = replace(scene(), cameras=cameras)
        cameras.clear()
        self.assertIn("head", obs.cameras)
        with self.assertRaises(TypeError):
            obs.cameras["new"] = camera

    def test_camera_rejects_invalid_calibration(self):
        camera = scene().cameras["head"]
        reflection = np.diag([1, 1, -1, 1])
        for changes in ({"T_world_from_camera_optical": reflection},
                        {"intrinsics": np.zeros((3, 3))},
                        {"depth_m": np.ones((2, 4))},
                        {"rgb": np.zeros((3, 4, 3), dtype=float)}):
            with self.subTest(changes=changes.keys()), self.assertRaises(ValueError):
                replace(camera, **changes)

    def test_invalid_depth_is_preserved_for_perception_filtering(self):
        depth = np.ones((3, 4))
        depth[0] = [np.nan, np.inf, 0, -1]
        camera = replace(scene().cameras["head"], depth_m=depth)
        self.assertTrue(np.isnan(camera.depth_m[0, 0]))

    def test_invalid_segmentation_response_is_rejected(self):
        seg = segment(scene())
        for changes in ({"score": 2}, {"score": float("nan")},
                        {"mask": np.ones((3, 4))}, {"mask": np.zeros((3, 4), bool)},
                        {"box_xyxy": [0, 0, 5, 3]}, {"input_rgb_sha256": "bad"}):
            with self.subTest(changes=changes.keys()), self.assertRaises(ValueError):
                replace(seg, **changes)

    def test_pose_requires_unit_quaternion_and_explicit_frame(self):
        for q in ([0, 0, 0, 0], [2, 0, 0, 0], [float("nan"), 0, 0, 0]):
            with self.assertRaises(ValueError):
                Pose([0, 0, 0], q, "world")
        with self.assertRaises(ValueError):
            Pose([0, 0, 0], [1, 0, 0, 0], "")

    def test_action_is_complete_and_state_bound(self):
        obs = scene()
        cmd = ArmCommand(1, ee_target=obs.robot_state.arms["left"].ee_pose)
        action = RobotAction(obs.source, {"left": cmd, "right": cmd})
        action.validate_against(obs)
        with self.assertRaises(ValueError):
            replace(action, arms={"left": cmd}).validate_against(obs)
        with self.assertRaises(ValueError):
            action.validate_against(replace(obs, source=replace(obs.source, state_version=2)))
        with self.assertRaises(ValueError):
            ArmCommand(1, ee_target=cmd.ee_target, joint_targets=[0, 0])

    def test_joint_dimensions_come_from_robot_state(self):
        obs = scene()
        cmd = ArmCommand(0.5, joint_targets=[0, 0])
        action = RobotAction(obs.source, {"left": cmd, "right": cmd})
        action.validate_against(obs)
        bad = ArmCommand(0.5, joint_targets=[0] * 7)
        with self.assertRaises(ValueError):
            replace(action, arms={"left": bad, "right": bad}).validate_against(obs)

    def test_result_keeps_partial_evidence(self):
        seg = segment(scene())
        result = Result(data=(seg,), error=HarnessError(ErrorCode.AMBIGUOUS, "two candidates"))
        self.assertFalse(result.ok)
        self.assertIs(result.data[0], seg)
        self.assertTrue(Result(data=()).ok)
        with self.assertRaises(ValueError):
            Result()

    def test_counts_and_timestamps_reject_invalid_values(self):
        source = scene().source
        for changes in ({"native_action_count": True}, {"state_version": -1},
                        {"captured_monotonic_s": float("inf")}, {"sim_time_s": -1}):
            with self.assertRaises(ValueError):
                replace(source, **changes)

    def test_point_cloud_rejects_nonfinite_points(self):
        source = scene().source
        with self.assertRaises(ValueError):
            PointCloud(source, "head", "world", [[np.nan, 0, 0]])
        self.assertEqual(PointCloud(source, "head", "world", np.empty((0, 3))).points.shape, (0, 3))

    def test_import_does_not_load_model_or_simulator(self):
        subprocess.run([sys.executable, "-c", "import sys; import embodied_harness.contracts; "
                        "assert not ({'torch', 'sam3', 'isaacsim', 'requests'} & sys.modules.keys())"],
                       check=True)


if __name__ == "__main__":
    unittest.main()
