"""Declarative stages submitted by System 2; no wire-supplied Python execution."""

from dataclasses import dataclass
import hashlib
import json

from embodied_harness.contracts import Pose
from embodied_harness.contracts.types import _count, _number, _text


@dataclass(frozen=True)
class Stage:
    stage_id: str
    kind: str
    arm: str | None = None
    target: Pose | None = None
    opening: float | None = None
    relation: str | None = None

    def __post_init__(self):
        _text(self.stage_id, "stage_id")
        if self.kind == "servo":
            if self.arm not in ("left", "right") or not isinstance(self.target, Pose) or self.relation is not None:
                raise ValueError("servo stage requires arm/target and no relation")
            if self.opening is not None:
                _number(self.opening, "opening")
                if self.opening > 1:
                    raise ValueError("opening must be normalized")
        elif self.kind == "relation":
            _text(self.relation, "relation")
            if self.arm is not None or self.target is not None or self.opening is not None:
                raise ValueError("relation stage has no motion fields")
        else:
            raise ValueError("unsupported stage kind")


@dataclass(frozen=True)
class Program:
    task_id: str
    goal: str
    revision: int
    stages: tuple[Stage, ...]
    max_actions: int = 100
    max_decisions: int = 150
    max_seconds: float = 180
    max_state_age_s: float = 10
    min_confidence: float = 0.5
    position_tolerance_m: float = 0.003
    rotation_tolerance_rad: float = 0.01
    max_no_progress: int = 5

    def __post_init__(self):
        _text(self.task_id, "task_id")
        _text(self.goal, "goal")
        for name in ("revision", "max_actions", "max_decisions", "max_no_progress"):
            _count(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_seconds", "max_state_age_s", "position_tolerance_m", "rotation_tolerance_rad"):
            _number(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        _number(self.min_confidence, "min_confidence")
        if self.min_confidence > 1:
            raise ValueError("min_confidence must be <=1")
        stages = tuple(self.stages)
        if not stages or len(stages) > 100 or any(not isinstance(s, Stage) for s in stages):
            raise ValueError("program requires 1..100 stages")
        if len({s.stage_id for s in stages}) != len(stages):
            raise ValueError("stage IDs must be unique")
        object.__setattr__(self, "stages", stages)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        stages = []
        for item in data.pop("stages"):
            stage = dict(item)
            if stage.get("target") is not None:
                stage["target"] = Pose(**stage["target"])
            stages.append(Stage(**stage))
        return cls(stages=tuple(stages), **data)

    def fingerprint(self):
        from embodied_harness.wire import dumps
        return hashlib.sha256(dumps({"goal": self.goal, "stages": [
            {"kind": s.kind, "arm": s.arm, "target": s.target,
             "opening": s.opening, "relation": s.relation} for s in self.stages]})).hexdigest()
