"""Host-owned ordered execution, evidence gates and bounded System 2 repair."""

from dataclasses import dataclass
import hashlib
import threading
import time
import uuid

from embodied_harness.control import Executor, pose_error, servo_action
from embodied_harness.contracts import ObservationSource
from embodied_harness.contracts.types import _count, _text
from embodied_harness.runtime.program import Program
from embodied_harness.runtime.selectors import FixedSelector
from embodied_harness.wire import dumps


@dataclass(frozen=True)
class RelationEvidence:
    source: ObservationSource
    relation: str
    verdict: str
    reason: str

    def __post_init__(self):
        if not isinstance(self.source, ObservationSource):
            raise TypeError("relation evidence requires observation source")
        _text(self.relation, "relation")
        _text(self.reason, "reason")
        if self.verdict not in ("true", "false", "unknown"):
            raise ValueError("relation verdict must be true/false/unknown")


class Session:
    def __init__(self, executor: Executor, store, event_log, *, selector=None, perception=None,
                 evaluators=None, max_revisions=3, skill_python=None):
        _count(max_revisions, "max_revisions")
        if not max_revisions:
            raise ValueError("max_revisions must be positive")
        self.executor, self.store, self.log = executor, store, event_log
        self.selector = selector or FixedSelector()
        self.perception, self.evaluators = perception, dict(evaluators or {})
        self.max_revisions = max_revisions
        self.tasks, self.lock = {}, threading.RLock()
        self.last_report = None
        self.stop_requested = threading.Event()
        self.skill_python, self.python_tasks = skill_python, {}

    def stop(self):
        """Latch stop; an in-flight native action must still return its physical receipt."""
        self.stop_requested.set()
        return {"stop_requested": True}

    def _save(self, observation):
        try:
            self.store.put(observation)
        except Exception:
            self.executor.faulted = True
            raise
        return observation

    def observe(self):
        with self.lock:
            return self._save(self.executor.observe())

    def localize(self, episode_id, snapshot_id, camera_id, query, candidate_index=None):
        with self.lock:
            if self.perception is None:
                raise ValueError("perception provider not configured")
            if self.executor.current is None or episode_id != self.executor.current.source.episode_id:
                raise ValueError("snapshot episode is outside current session")
            result = self.perception.localize_object(episode_id, snapshot_id, camera_id, query,
                                                     candidate_index=candidate_index)
            self.log.append("localization", result=result)
            return result

    def tools(self):
        return {"observe": self.observe, "localize_object": self.localize,
                "delegate": self.delegate, "delegate_python": self.delegate_python,
                "get_report": lambda: self.last_report, "stop": self.stop,
                "capabilities": lambda: {"protocol": "embodied-harness-v1",
                    "stage_kinds": ["servo", "relation"], "selector": type(self.selector).__name__,
                    "relations": sorted(self.evaluators), "phase": self.executor.adapter.metadata["phase"]}}

    def delegate_python(self, task_id, goal, source, parameters, revision=1,
                        max_actions=100, max_decisions=150, max_seconds=180, relation=None):
        """Persistent Python hook loop; source completion alone never proves a physical goal."""
        from embodied_harness.programs import PersistentProgram, PythonSkill
        import ast
        with self.lock:
            if self.stop_requested.is_set():
                raise RuntimeError("session stopped")
            _text(task_id, "task_id")
            _text(goal, "goal")
            _text(source, "source")
            if task_id in self.tasks:
                raise ValueError("task cannot switch execution profiles")
            for name, value, cap in (("max_actions", max_actions, 100), ("max_decisions", max_decisions, 150)):
                _count(value, name)
                if not 0 < value <= cap:
                    raise ValueError("Python task budget outside host limits")
            from embodied_harness.contracts.types import _number
            _number(max_seconds, "max_seconds")
            _count(revision, "revision")
            if not 0 < max_seconds <= 180 or revision == 0 or not isinstance(parameters, dict):
                raise ValueError("invalid Python task parameters")
            digest = hashlib.sha256(dumps({"ast": ast.dump(ast.parse(source)), "parameters": parameters})).hexdigest()
            record = self.python_tasks.get(task_id)
            if record:
                if (record["goal"] != goal or revision <= record["revision"]
                        or record["attempts"] >= self.max_revisions or digest in record["hashes"]
                        or max_actions > record["max_actions"] or max_decisions > record["max_decisions"]
                        or max_seconds > record["max_seconds"]):
                    raise ValueError("invalid Python task repair")
            else:
                record = {"goal": goal, "revision": 0, "attempts": 0, "hashes": set(),
                          "max_actions": max_actions, "max_decisions": max_decisions, "max_seconds": max_seconds}
                self.python_tasks[task_id] = record
            record["revision"], record["attempts"] = revision, record["attempts"] + 1
            record["hashes"].add(digest)
            started, initial_actions, decisions = time.monotonic(), self.executor.submissions, 0
            native_success = False
            diagnostics = {}

            def report(status, reason):
                result = {"status": status, "reason": reason, "task_id": task_id, "revision": revision,
                          "submissions": self.executor.submissions - initial_actions,
                          "decisions": decisions, "source": self.executor.current.source if self.executor.current else None,
                          "profile": "python-hooks-v1", "diagnostics": dict(diagnostics)}
                self.log.append("python_handoff", report=result)
                self.last_report = result
                return result

            def step(action):
                nonlocal native_success
                if (self.stop_requested.is_set() or time.monotonic() - started > max_seconds
                        or self.executor.submissions - initial_actions >= max_actions):
                    raise RuntimeError("Python task stopped or out of action/time budget")
                action_id = uuid.uuid4().hex
                self.log.append("action_intent", action=action, action_id=action_id)
                try:
                    receipt = self.executor.execute(action, action_id=action_id)
                    self._save(receipt.observation)
                    self.log.append("action_result", receipt=receipt)
                except BaseException:
                    self.executor.faulted = True
                    raise
                native_success = receipt.native_task_success
                return receipt

            process = None
            try:
                self.log.append("python_delegate", task_id=task_id, revision=revision, source=source,
                                parameters=parameters, fingerprint=digest)
                self.observe()
                process = PersistentProgram(
                    {"get_observation": self.observe, "step": step, "localize_object": self.localize},
                    python_executable=self.skill_python,
                    timeout_s=min(30, max_seconds))
                skill = PythonSkill(process)
                initialized = skill.initialize(source, parameters)
                diagnostics.update({key: initialized[key] for key in ("error_type", "traceback", "stdout", "stderr") if key in initialized})
                if not initialized["ok"]:
                    return report("needs_system2", "python_init_failed")
                while time.monotonic() - started <= max_seconds:
                    if self.stop_requested.is_set():
                        return report("stopped", "operator_stop_requested")
                    observation_result = skill.refresh()
                    diagnostics.clear()
                    diagnostics.update({key: observation_result[key] for key in ("error_type", "traceback", "stdout", "stderr") if key in observation_result})
                    self.log.append("python_observe", result=observation_result)
                    if not observation_result["ok"]:
                        return report("needs_system2", "python_observe_failed")
                    context = observation_result["exports"]["context"]
                    options = observation_result["exports"]["options"]
                    if not isinstance(context, dict) or not isinstance(options, dict):
                        return report("needs_system2", "invalid_python_snapshot")
                    status = context.get("status")
                    if status == "done":
                        evaluator = self.evaluators.get(relation)
                        if evaluator:
                            current = self.executor.current
                            evidence = evaluator(current)
                            if (isinstance(evidence, RelationEvidence) and evidence.source == current.source
                                    and evidence.relation == relation and evidence.verdict == "true"):
                                return report("verified_subtask", "host_relation_verified")
                        return report("awaiting_verification", "code_done_is_not_physical_success")
                    if not 1 <= len(options) <= 12 or any(not isinstance(k, str) or not isinstance(v, str) for k, v in options.items()):
                        return report("needs_system2", "invalid_python_menu")
                    if status != "active" or decisions >= max_decisions:
                        return report("needs_system2", "python_status_or_decision_budget")
                    self.log.append("selection_request", state=context, options=options)
                    selected = self.selector.select(context, options)
                    decisions += 1
                    selected.validate(options)
                    self.log.append("selection", option=selected.option, confidence=selected.confidence,
                                    raw_response=getattr(self.selector, "last_response", None))
                    if selected.confidence < .5:
                        return report("needs_system2", "low_confidence")
                    if self.stop_requested.is_set():
                        return report("stopped", "operator_stop_requested")
                    tick = skill.tick(selected.option)
                    diagnostics.clear()
                    diagnostics.update({key: tick[key] for key in ("error_type", "traceback", "stdout", "stderr") if key in tick})
                    self.log.append("python_tick", result=tick)
                    if self.executor.faulted:
                        return report("needs_system2", "uncertain_action_outcome")
                    if native_success:
                        return report("native_task_succeeded", "native_terminal_success")
                    if self.executor.terminal or not tick["ok"]:
                        return report("needs_system2", "python_tick_failed_or_native_ended")
                return report("needs_system2", "time_budget")
            except Exception as exc:
                return report("needs_system2", "python_error:" + type(exc).__name__)
            finally:
                if process is not None:
                    process.close()

    @staticmethod
    def _stage_hash(stage):
        return hashlib.sha256(dumps({"kind": stage.kind, "arm": stage.arm, "target": stage.target,
                                    "opening": stage.opening, "relation": stage.relation})).hexdigest()

    def delegate(self, manifest):
        with self.lock:
            return self._delegate(Program.from_dict(manifest))

    def _delegate(self, program):
        if program.task_id in self.python_tasks:
            raise ValueError("task cannot switch execution profiles")
        if self.stop_requested.is_set():
            raise RuntimeError("session stopped; start a reconciled host session to continue")
        upper_limits = {"max_actions": 100, "max_decisions": 150, "max_seconds": 180,
                        "max_state_age_s": 10, "position_tolerance_m": 0.003,
                        "rotation_tolerance_rad": 0.01, "max_no_progress": 5}
        if any(getattr(program, name) > bound for name, bound in upper_limits.items()) or program.min_confidence < 0.5:
            raise ValueError("program exceeds host policy limits")
        record = self.tasks.get(program.task_id)
        fingerprint = program.fingerprint()
        if record:
            if record["done"]:
                raise ValueError("task already completed")
            if (program.goal != record["goal"] or program.revision <= record["revision"]
                    or record["attempts"] >= self.max_revisions or fingerprint in record["fingerprints"]):
                raise ValueError("repair must be bounded, newer, changed, and preserve the task goal")
            if any(getattr(program, name) > bound for name, bound in record["limits"].items()):
                raise ValueError("repair cannot enlarge budgets or weaken evidence limits")
            if program.min_confidence < record["min_confidence"]:
                raise ValueError("repair cannot lower confidence threshold")
            prefix = [self._stage_hash(s) for s in program.stages[:len(record["completed"])]]
            if prefix != record["completed"]:
                raise ValueError("repair cannot rewrite completed stages")
        else:
            record = {"goal": program.goal, "revision": 0, "attempts": 0,
                      "fingerprints": set(), "completed": [], "done": False,
                      "limits": {name: getattr(program, name) for name in upper_limits},
                      "min_confidence": program.min_confidence}
            self.tasks[program.task_id] = record
        record["revision"] = program.revision
        record["attempts"] += 1
        record["fingerprints"].add(fingerprint)
        started, initial_actions = time.monotonic(), self.executor.submissions
        decisions, index = 0, len(record["completed"])
        initial_native = None
        self.log.append("delegate", task_id=program.task_id, revision=program.revision,
                        fingerprint=fingerprint, max_actions=program.max_actions)

        def finish(status, reason):
            current = self.executor.current
            report = {"status": status, "reason": reason, "task_id": program.task_id,
                      "revision": program.revision, "stage_index": index,
                      "completed_stage_count": len(record["completed"]),
                      "submissions": self.executor.submissions - initial_actions,
                      "decisions": decisions, "elapsed_s": time.monotonic() - started,
                      "last_action_id": self.executor.last_action_id,
                      "source": current.source if current else None,
                      "native_action_delta": (current.source.native_action_count - initial_native
                                               if current and initial_native is not None else 0)}
            self.log.append("handoff", report=report)
            self.last_report = report
            return report

        previous_error, no_progress = None, 0
        try:
            observation = self.observe()
            initial_native = observation.source.native_action_count
            while index < len(program.stages):
                if self.stop_requested.is_set():
                    return finish("stopped", "operator_stop_requested")
                if time.monotonic() - started > program.max_seconds:
                    return finish("needs_system2", "time_budget")
                stage = program.stages[index]
                if stage.kind == "relation":
                    evaluator = self.evaluators.get(stage.relation)
                    if evaluator is None:
                        return finish("needs_system2", "relation_evaluator_unavailable")
                    evidence = evaluator(observation)
                    if (not isinstance(evidence, RelationEvidence) or evidence.source != observation.source
                            or evidence.relation != stage.relation):
                        return finish("needs_system2", "invalid_or_stale_relation_evidence")
                    self.log.append("relation", source=evidence.source, relation=evidence.relation,
                                    verdict=evidence.verdict, reason=evidence.reason)
                    if evidence.verdict != "true":
                        return finish("needs_system2", "relation_" + evidence.verdict)
                    arrived = True
                else:
                    measured = observation.robot_state.arms[stage.arm]
                    position, rotation = pose_error(measured.ee_pose, stage.target)
                    grip_ok = stage.opening is None or abs(measured.gripper_position - stage.opening) <= 0.01
                    arrived = (position <= program.position_tolerance_m
                               and rotation <= program.rotation_tolerance_rad and grip_ok)
                if arrived:
                    record["completed"].append(self._stage_hash(stage))
                    self.log.append("stage_verified", stage_id=stage.stage_id, stage_kind=stage.kind,
                                    source=observation.source)
                    index += 1
                    previous_error, no_progress = None, 0
                    continue
                if self.executor.terminal:
                    return finish("needs_system2", "native_episode_ended")
                if (decisions >= program.max_decisions
                        or self.executor.submissions - initial_actions >= program.max_actions):
                    return finish("needs_system2", "decision_or_action_budget")
                state = {"stage": stage.stage_id, "position_error_m": position,
                         "rotation_error_rad": rotation, "gripper_position": measured.gripper_position}
                options = {"fine": "translation 1mm, rotation 0.01rad, jaw change <=0.02",
                           "normal": "translation 10mm, rotation 0.10rad, jaw change <=0.02"}
                self.log.append("selection_request", state=state, options=options, source=observation.source)
                decision_started = time.monotonic()
                selected = self.selector.select(state, options)
                decisions += 1
                selected.validate(options)
                self.log.append("selection", option=selected.option, confidence=selected.confidence,
                                seconds=time.monotonic() - decision_started,
                                raw_response=getattr(self.selector, "last_response", None))
                if self.stop_requested.is_set():
                    return finish("stopped", "operator_stop_requested")
                if selected.confidence < program.min_confidence:
                    return finish("needs_system2", "low_confidence")
                if time.monotonic() - started > program.max_seconds:
                    return finish("needs_system2", "time_budget_after_selection")
                scale = 0.1 if selected.option == "fine" else 1
                opening = (None if stage.opening is None else measured.gripper_position +
                           max(-0.02, min(0.02, stage.opening - measured.gripper_position)))
                action = servo_action(observation, stage.arm, stage.target, opening=opening,
                                      translation_m=0.01 * scale, rotation_rad=0.1 * scale)
                action_id = uuid.uuid4().hex
                self.log.append("action_intent", action=action, action_id=action_id)
                receipt = self.executor.execute(action, max_age_s=program.max_state_age_s, action_id=action_id)
                observation = self._save(receipt.observation)
                self.log.append("action_result", receipt=receipt)
                if receipt.native_task_success:
                    record["done"] = True
                    return finish("native_task_succeeded", "native_terminal_success")
                error = sum(pose_error(observation.robot_state.arms[stage.arm].ee_pose, stage.target))
                error += 0 if stage.opening is None else abs(observation.robot_state.arms[stage.arm].gripper_position - stage.opening)
                no_progress = no_progress + 1 if previous_error is not None and error >= previous_error - 1e-6 else 0
                previous_error = error
                if no_progress >= program.max_no_progress:
                    return finish("needs_system2", "no_measured_progress")
            record["done"] = True
            return finish("program_completed", "configured_stage_predicates_verified")
        except Exception as exc:
            # A failed required evidence write is not converted into a reviewable success report.
            if isinstance(exc, OSError):
                self.executor.faulted = True
            return finish("needs_system2", "execution_error:" + type(exc).__name__)
