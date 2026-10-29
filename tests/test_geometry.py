from dataclasses import replace
from itertools import product
import unittest

import numpy as np

from embodied_harness.contracts import ErrorCode, PointCloud
from embodied_harness.geometry import estimate_geometry, mask_to_point_cloud, transform_point_cloud
from test_contracts import scene, segment


class ProjectionTests(unittest.TestCase):
    def calibrated_scene(self):
        obs = scene()
        k = np.array([[2, 0, 1], [0, 4, 1], [0, 0, 1]])
        camera = replace(obs.cameras["head"], depth_m=np.full((3, 4), 2.0), intrinsics=k)
        return replace(obs, cameras={"head": camera})

    def test_known_pixels_use_optical_axis_depth(self):
        obs = self.calibrated_scene()
        result = mask_to_point_cloud(obs, segment(obs), target_frame="camera/head/optical")
        self.assertTrue(result.ok)
        # Pixel (0,0), principal point (1,1), and a point right/below it.
        np.testing.assert_allclose(result.data.cloud.points[[0, 5, 11]],
                                   [[-1, -0.5, 2], [0, 0, 2], [2, 0.5, 2]])
        self.assertEqual(result.data.valid_depth_fraction, 1)
        self.assertEqual(result.data.cloud.source, obs.source)

    def test_rotation_translation_and_inverse(self):
        obs = self.calibrated_scene()
        t = np.array([[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        camera = replace(obs.cameras["head"], T_world_from_camera_optical=t)
        obs = replace(obs, cameras={"head": camera})
        cloud = mask_to_point_cloud(obs, segment(obs)).data.cloud
        np.testing.assert_allclose(cloud.points[0], [3.5, 3, 7])
        back = transform_point_cloud(cloud, np.linalg.inv(t), source_frame="world",
                                     target_frame="camera/head/optical")
        expected = mask_to_point_cloud(obs, segment(obs), target_frame=back.frame).data.cloud
        np.testing.assert_allclose(back.points, expected.points)

    def test_projection_roundtrip_recovers_pixel_grid(self):
        obs = self.calibrated_scene()
        p = mask_to_point_cloud(obs, segment(obs), target_frame="camera/head/optical").data.cloud.points
        k = obs.cameras["head"].intrinsics
        uv = p[:, :2] / p[:, 2:] * [k[0, 0], k[1, 1]] + k[:2, 2]
        rows, cols = np.indices((3, 4))
        np.testing.assert_allclose(uv, np.column_stack((cols.ravel(), rows.ravel())))

    def test_mask_and_depth_filters_keep_only_matching_pixels(self):
        obs = self.calibrated_scene()
        depth = np.array([[np.nan, np.inf, 0, -1], [1, 2, 3, 4], [5, 6, 7, 8]])
        obs = replace(obs, cameras={"head": replace(obs.cameras["head"], depth_m=depth)})
        mask = np.ones((3, 4), bool)
        mask[2] = False
        result = mask_to_point_cloud(obs, replace(segment(obs), mask=mask),
                                     min_depth_m=1, max_depth_m=3)
        self.assertEqual(result.data.mask_pixel_count, 8)
        self.assertEqual(result.data.valid_depth_count, 2)
        self.assertEqual(result.data.valid_depth_fraction, 0.25)
        np.testing.assert_allclose(result.data.cloud.points[:, 2], [2, 3])

    def test_no_depth_returns_failure_with_partial_evidence(self):
        obs = scene()
        obs = replace(obs, cameras={"head": replace(obs.cameras["head"], depth_m=np.zeros((3, 4)))})
        result = mask_to_point_cloud(obs, segment(obs))
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.NO_VALID_DEPTH)
        self.assertEqual(result.data.valid_depth_fraction, 0)
        self.assertEqual(result.data.cloud.points.shape, (0, 3))

    def test_different_snapshot_or_reset_is_rejected(self):
        obs = scene()
        seg = segment(obs)
        for field, value in (("snapshot_id", "other"), ("reset_epoch", 1)):
            with self.assertRaises(ValueError):
                mask_to_point_cloud(replace(obs, source=replace(obs.source, **{field: value})), seg)

    def test_camera_transform_comes_from_current_snapshot(self):
        obs = scene()
        original = mask_to_point_cloud(obs, segment(obs)).data.cloud.points
        t = np.eye(4)
        t[0, 3] = 0.25
        current = replace(obs, source=replace(obs.source, snapshot_id="moved", state_version=2),
                          cameras={"head": replace(obs.cameras["head"], T_world_from_camera_optical=t)})
        moved = mask_to_point_cloud(current, segment(current)).data.cloud.points
        np.testing.assert_allclose(moved - original, np.tile([0.25, 0, 0], (12, 1)))

    def test_invalid_frames_transforms_and_ranges_are_rejected(self):
        obs = scene()
        cloud = mask_to_point_cloud(obs, segment(obs)).data.cloud
        for options in ({"target_frame": "robot_base"}, {"min_depth_m": -1},
                        {"min_depth_m": 2, "max_depth_m": 1}, {"max_depth_m": np.inf}):
            with self.assertRaises(ValueError):
                mask_to_point_cloud(obs, segment(obs), **options)
        with self.assertRaises(ValueError):
            transform_point_cloud(cloud, np.eye(4), source_frame="wrong", target_frame="base")
        for t in (np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
            with self.assertRaises(ValueError):
                transform_point_cloud(cloud, t, source_frame="world", target_frame="base")


class GeometryTests(unittest.TestCase):
    def cloud(self, points):
        return PointCloud(scene().source, "head", "world", points)

    def test_rotated_box_recovers_center_extents_and_contains_points(self):
        corners = np.array(list(product([-2, 2], [-1, 1], [-0.5, 0.5])))
        a = 0.43
        r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        points = corners @ r.T + [3, 4, 5]
        result = estimate_geometry(self.cloud(points))
        self.assertTrue(result.ok)
        g = result.data
        np.testing.assert_allclose(g.center, [3, 4, 5])
        np.testing.assert_allclose(g.extents, [4, 2, 1])
        np.testing.assert_allclose(g.axes.T @ g.axes, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(g.axes), 1)
        self.assertTrue(np.all(np.abs((points - g.center) @ g.axes) <= g.extents / 2 + 1e-12))
        self.assertFalse(g.orientation_ambiguous)
        self.assertEqual(g.source, self.cloud(points).source)

    def test_symmetry_and_planarity_are_marked_ambiguous(self):
        cube = np.array(list(product([-1, 1], repeat=3)))
        plane = np.array([[x, y, 0] for x, y in product([-2, 2], [-1, 1])])
        for points in (cube, plane):
            result = estimate_geometry(self.cloud(points))
            self.assertTrue(result.ok)
            self.assertTrue(result.data.orientation_ambiguous)

    def test_empty_single_point_and_line_fail_geometry(self):
        for points in (np.empty((0, 3)), [[1, 1, 1]], np.ones((5, 3)),
                       [[0, 0, 0], [1, 0, 0], [2, 0, 0]]):
            result = estimate_geometry(self.cloud(points))
            self.assertFalse(result.ok)
            self.assertEqual(result.error.code, ErrorCode.GEOMETRY_FAILED)


if __name__ == "__main__":
    unittest.main()
