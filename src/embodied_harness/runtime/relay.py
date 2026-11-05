"""System 2 planning/repair around System 1 execution, with a frozen evaluation mode."""

from dataclasses import asdict, dataclass
import hashlib
import json

from embodied_harness.programs.policy import validate
from embodied_harness.contracts.types import _count, _text


@dataclass(frozen=True)
class Subtask:
    task_id: str
    goal: str
    source: str
    parameters: dict
    relation: str | None = None

    def __post_init__(self):
        for name in ("task_id", "goal", "source"):
            _text(getattr(self, name), name)
        validate(self.source)
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a JSON object")
        json.dumps(self.parameters, allow_nan=False)


@dataclass(frozen=True)
class Plan:
    goal: str
    subtasks: tuple[Subtask, ...]

    def __post_init__(self):
        _text(self.goal, "goal")
        tasks = tuple(self.subtasks)
        if not 1 <= len(tasks) <= 100 or any(not isinstance(task, Subtask) for task in tasks):
            raise ValueError("plan requires ordered subtasks")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("subtask IDs must be unique")
        object.__setattr__(self, "subtasks", tasks)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"goal", "subtasks"}:
            raise ValueError("plan fields must be goal/subtasks")
        return cls(value["goal"], tuple(Subtask(**task) for task in value["subtasks"]))

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()


class RemoteSystem2:
    """Adapter to an independently deployed agent; no simulator/credentials sent as context."""

    def __init__(self, rpc):
        self.rpc = rpc

    def generate(self, context):
        return self.rpc.call("generate", context=context)

    def repair(self, context):
        return self.rpc.call("repair", context=context)

    def review(self, context):
        return self.rpc.call("review", context=context)


class Relay:
    def __init__(self, session, system2=None, *, max_interventions=2):
        _count(max_interventions, "max_interventions")
        self.session, self.system2, self.max_interventions = session, system2, max_interventions
        if max_interventions >= session.max_revisions:
            raise ValueError("relay repairs must fit the session revision budget")

    def _context(self, goal, report=None, plan=None, index=0):
        observation = self.session.observe()
        return {"goal": goal, "observation": observation, "report": report,
                "plan": plan.to_dict() if plan else None, "active_index": index,
                "public_api": self.session.api.registry.describe()}

    def _request(self, method, context):
        if self.system2 is None:
            raise RuntimeError("System 2 agent not configured")
        self.session.log.append("system2_request", role=method, context=context)
        reply = getattr(self.system2, method)(context)
        self.session.log.append("system2_reply", role=method, reply=reply)
        return reply

    def run(self, goal, plan=None, *, allow_system2=True):
        _text(goal, "goal")
        if plan is None:
            if not allow_system2:
                raise ValueError("frozen execution requires a selected plan")
            plan = Plan.from_dict(self._request("generate", self._context(goal)))
        elif isinstance(plan, dict):
            plan = Plan.from_dict(plan)
        if plan.goal != goal:
            raise ValueError("System 2 changed the operator goal")
        initial_hash = plan.fingerprint()
        index, interventions, revision = 0, 0, 1
        reports = []
        status = "needs_system2"
        while index < len(plan.subtasks):
            task = plan.subtasks[index]
            report = self.session.delegate_python(task.task_id, task.goal, task.source,
                                                  task.parameters, revision=revision, relation=task.relation)
            reports.append(report)
            status = report["status"]
            if status == "native_task_succeeded":
                break
            if status == "verified_subtask":
                index += 1
                revision = 1
                continue
            if status == "stopped" or not allow_system2:
                break
            if status == "awaiting_verification":
                context = self._context(goal, report, plan, index)
                review = self._request("review", context)
                if (not isinstance(review, dict) or set(review) != {"snapshot_id", "verdict", "evidence"}
                        or review["snapshot_id"] != context["observation"].source.snapshot_id
                        or review["verdict"] not in ("true", "false", "unknown")
                        or not isinstance(review["evidence"], str) or not review["evidence"].strip()):
                    raise ValueError("System 2 review lacks fresh visible evidence")
                if review["verdict"] == "true":
                    index += 1
                    revision = 1
                    continue
            if interventions >= self.max_interventions:
                break
            replacement = Plan.from_dict(self._request("repair", self._context(goal, report, plan, index)))
            if replacement.goal != goal or replacement.subtasks[:index] != plan.subtasks[:index]:
                raise ValueError("repair changed goal or completed prefix")
            if len(replacement.subtasks) <= index or replacement.subtasks[index].task_id != task.task_id:
                raise ValueError("repair must retain current subtask identity")
            if replacement.subtasks[index].goal != task.goal or replacement.fingerprint() == plan.fingerprint():
                raise ValueError("repair changed subgoal or did not change the program")
            interventions += 1
            revision += 1
            plan = replacement
        # Completion/review of a plan is not native task success.
        result = {"status": "native_task_succeeded" if status == "native_task_succeeded" else "not_native_success",
                  "primary_native_success": status == "native_task_succeeded" and interventions == 0,
                  "recovered_native_success": status == "native_task_succeeded" and interventions > 0,
                  "interventions": interventions, "reports": reports, "selected_plan": plan.to_dict(),
                  "initial_plan_sha256": initial_hash, "plan_sha256": plan.fingerprint(),
                  "system2_enabled": allow_system2}
        self.session.log.append("relay_result", result=result)
        return result
