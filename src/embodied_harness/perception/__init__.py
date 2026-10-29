"""Read-only snapshot-bound localization orchestration."""

from dataclasses import dataclass

from embodied_harness.artifacts import SnapshotStore
from embodied_harness.contracts import ErrorCode, HarnessError, Result, Segmentation
from embodied_harness.contracts.protocols import SegmentationProvider
from embodied_harness.contracts.types import _number, _text
from embodied_harness.geometry import ObjectGeometry, Projection, estimate_geometry, mask_to_point_cloud


@dataclass(frozen=True)
class Localization:
    candidates: tuple[Segmentation, ...]
    selected_index: int | None = None
    projection: Projection | None = None
    geometry: ObjectGeometry | None = None


class Perception:
    def __init__(self, store: SnapshotStore, provider: SegmentationProvider | None):
        self.store, self.provider = store, provider

    def localize_object(
        self, episode_id: str, snapshot_id: str, camera_id: str, query: str, *,
        candidate_index: int | None = None, min_valid_depth_fraction: float = 0.5,
    ) -> Result[Localization]:
        """Never recapture or move; multiple candidates require an explicit choice."""
        _text(query, "query")
        _number(min_valid_depth_fraction, "min_valid_depth_fraction")
        if min_valid_depth_fraction > 1:
            raise ValueError("min_valid_depth_fraction must be in [0, 1]")
        if candidate_index is not None and (type(candidate_index) is not int or candidate_index < 0):
            raise ValueError("candidate_index must be a nonnegative integer")
        observation = self.store.get(episode_id, snapshot_id)
        if camera_id not in observation.cameras:
            raise ValueError("camera unavailable in snapshot")
        if self.provider is None:
            return Result(error=HarnessError(ErrorCode.NOT_CONFIGURED, "segmentation provider not configured"))
        response = self.provider.segment_text(observation, camera_id, query)
        try:
            if not isinstance(response, Result):
                raise ValueError("provider must return Result")
            candidates = () if response.data is None else response.data
            if not isinstance(candidates, tuple):
                raise ValueError("provider candidates must be a tuple")
            for candidate in candidates:
                if not isinstance(candidate, Segmentation) or candidate.camera_id != camera_id:
                    raise ValueError("invalid candidate or wrong camera")
                candidate.validate_against(observation)
                if candidate.prompt != query:
                    raise ValueError("candidate prompt does not match the request")
        except (ValueError, TypeError) as exc:
            return Result(error=HarnessError(ErrorCode.INVALID_RESPONSE, str(exc)))
        evidence = Localization(candidates)
        if not response.ok:
            return Result(evidence, response.error)
        if not candidates:
            return Result(evidence, HarnessError(ErrorCode.NO_CANDIDATES, "no matching instance"))
        if candidate_index is None and len(candidates) > 1:
            return Result(evidence, HarnessError(ErrorCode.AMBIGUOUS, "explicit candidate selection required"))
        index = 0 if candidate_index is None else candidate_index
        if index >= len(candidates):
            raise ValueError("candidate_index is out of range")
        projection = mask_to_point_cloud(observation, candidates[index])
        evidence = Localization(candidates, index, projection.data)
        if not projection.ok:
            return Result(evidence, projection.error)
        if projection.data.valid_depth_fraction < min_valid_depth_fraction:
            return Result(evidence, HarnessError(ErrorCode.INSUFFICIENT_DEPTH, "valid depth coverage below threshold"))
        geometry = estimate_geometry(projection.data.cloud)
        evidence = Localization(candidates, index, projection.data, geometry.data)
        return Result(evidence, geometry.error)


__all__ = ["Localization", "Perception"]
