"""First perception interfaces; providers must not recapture the observation."""

from typing import Protocol

from .types import Observation, Result, Segmentation


class ObservationProvider(Protocol):
    def observe(self) -> Observation:
        """Read public state without submitting a native action."""
        ...


class SegmentationProvider(Protocol):
    def segment_text(
        self, observation: Observation, camera_id: str, query: str,
    ) -> Result[tuple[Segmentation, ...]]:
        ...

    def segment_point(
        self, observation: Observation, camera_id: str, point_xy: tuple[float, float],
    ) -> Result[tuple[Segmentation, ...]]:
        """Exactly one positive pixel prompt in the supplied snapshot."""
        ...
