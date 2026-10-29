"""Snapshot-bound optical geometry; no environment or model calls."""

from dataclasses import dataclass

import numpy as np

from embodied_harness.contracts import (
    ErrorCode, HarnessError, Observation, ObservationSource, PointCloud, Pose, Result, Segmentation,
)
from embodied_harness.contracts.types import _array, _count, _number, _text


def _rigid_transform(value):
    t = _array(value, (4, 4), "transform")
    r = t[:3, :3]
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)):
        raise ValueError("transform must be a proper rigid transform")
    return t


def pose_to_matrix(pose: Pose):
    w, x, y, z = pose.quaternion_wxyz
    t = np.eye(4)
    t[:3, :3] = [[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]]
    t[:3, 3] = pose.position
    return t


def matrix_to_pose(matrix, *, frame):
    t = _rigid_transform(matrix)
    r = t[:3, :3]
    # Symmetric eigenproblem avoids unstable trace divisions near 180 degrees.
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                  [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
                  [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    if q[0] < 0:
        q = -q
    return Pose(t[:3, 3], q, frame)


@dataclass(frozen=True)
class Projection:
    cloud: PointCloud
    mask_pixel_count: int
    valid_depth_count: int

    def __post_init__(self):
        if not isinstance(self.cloud, PointCloud):
            raise TypeError("cloud must be PointCloud")
        _count(self.mask_pixel_count, "mask_pixel_count")
        _count(self.valid_depth_count, "valid_depth_count")
        if not 0 <= self.valid_depth_count <= self.mask_pixel_count or self.mask_pixel_count == 0:
            raise ValueError("projection counts must describe a nonempty mask")
        if self.valid_depth_count != len(self.cloud.points):
            raise ValueError("valid depth count must match point count")

    @property
    def valid_depth_fraction(self) -> float:
        return self.valid_depth_count / self.mask_pixel_count


@dataclass(frozen=True, eq=False)
class ObjectGeometry:
    """PCA box of visible surfaces, not an identity-resolved full object pose."""

    source: ObservationSource
    camera_id: str
    frame: str
    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    point_count: int
    orientation_ambiguous: bool

    def __post_init__(self):
        if not isinstance(self.source, ObservationSource):
            raise TypeError("source must be ObservationSource")
        _text(self.camera_id, "camera_id")
        _text(self.frame, "frame")
        for name, shape in (("center", (3,)), ("axes", (3, 3)), ("extents", (3,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        t = np.eye(4)
        t[:3, :3] = self.axes
        _rigid_transform(t)
        if np.any(self.extents < 0):
            raise ValueError("extents must be nonnegative")
        _count(self.point_count, "point_count")
        if self.point_count < 3 or type(self.orientation_ambiguous) is not bool:
            raise ValueError("geometry requires >=3 points and a boolean ambiguity flag")


def transform_point_cloud(
    cloud: PointCloud, T_target_from_source, *, source_frame: str, target_frame: str,
) -> PointCloud:
    """Apply an explicitly directed transform, preserving observation provenance."""
    if cloud.frame != source_frame:
        raise ValueError("cloud frame does not match transform source frame")
    _text(target_frame, "target_frame")
    t = _rigid_transform(T_target_from_source)
    return PointCloud(cloud.source, cloud.camera_id, target_frame,
                      cloud.points @ t[:3, :3].T + t[:3, 3])


def mask_to_point_cloud(
    observation: Observation, segmentation: Segmentation, *, target_frame: str = "world",
    min_depth_m: float = 0.0, max_depth_m: float | None = None,
) -> Result[Projection]:
    """Use optical-axis depth from exactly the segmentation's snapshot.

    Invalid contract/configuration raises ValueError. No usable measured depth
    returns NO_VALID_DEPTH with an empty cloud and coverage as partial evidence.
    """
    segmentation.validate_against(observation)
    camera = observation.cameras[segmentation.camera_id]
    optical_frame = f"camera/{camera.camera_id}/optical"
    if target_frame not in ("world", optical_frame):
        raise ValueError("target frame must be world or the source camera optical frame")
    _number(min_depth_m, "min_depth_m")
    if max_depth_m is not None:
        _number(max_depth_m, "max_depth_m")
        if max_depth_m <= min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")
    rows, cols = np.nonzero(segmentation.mask)
    depth = camera.depth_m[rows, cols]
    valid = np.isfinite(depth) & (depth > min_depth_m)
    if max_depth_m is not None:
        valid &= depth <= max_depth_m
    z = depth[valid]
    k = camera.intrinsics
    points = np.column_stack(((cols[valid] - k[0, 2]) * z / k[0, 0],
                              (rows[valid] - k[1, 2]) * z / k[1, 1], z))
    cloud = PointCloud(observation.source, camera.camera_id, optical_frame, points)
    if target_frame == "world":
        cloud = transform_point_cloud(cloud, camera.T_world_from_camera_optical,
                                      source_frame=optical_frame, target_frame="world")
    projection = Projection(cloud, len(depth), len(z))
    if not len(z):
        return Result(projection, HarnessError(ErrorCode.NO_VALID_DEPTH,
                                               "mask contains no usable positive depth"))
    return Result(projection)


def estimate_geometry(cloud: PointCloud) -> Result[ObjectGeometry]:
    """Estimate a visible-surface box; report PCA symmetry/planarity ambiguity."""
    points = cloud.points
    if len(points) < 3:
        return Result(error=HarnessError(ErrorCode.GEOMETRY_FAILED, "at least three points required"))
    mean = points.mean(axis=0)
    centered = points - mean
    try:
        eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(points))
    except np.linalg.LinAlgError:
        return Result(error=HarnessError(ErrorCode.GEOMETRY_FAILED, "PCA decomposition failed"))
    eigenvalues, axes = eigenvalues[::-1], axes[:, ::-1].copy()
    if not np.isfinite(eigenvalues).all() or eigenvalues[1] <= max(eigenvalues[0] * 1e-10, 1e-18):
        return Result(error=HarnessError(ErrorCode.GEOMETRY_FAILED, "point cloud is point/line degenerate"))
    for col in range(3):
        if axes[np.argmax(np.abs(axes[:, col])), col] < 0:
            axes[:, col] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    local = centered @ axes
    lower, upper = local.min(axis=0), local.max(axis=0)
    ambiguous = bool(eigenvalues[2] <= eigenvalues[0] * 1e-6
                     or np.any(np.abs(np.diff(eigenvalues)) <= eigenvalues[0] * 0.05))
    geometry = ObjectGeometry(cloud.source, cloud.camera_id, cloud.frame,
                              mean + axes @ ((lower + upper) / 2), axes, upper - lower,
                              len(points), ambiguous)
    return Result(geometry)


__all__ = ["ObjectGeometry", "Projection", "estimate_geometry", "mask_to_point_cloud",
           "transform_point_cloud", "pose_to_matrix", "matrix_to_pose"]
