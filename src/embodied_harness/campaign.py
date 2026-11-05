"""ASPIRE tabletop fix-loop over an explicit, private RoboDojo layout partition."""

import json
from pathlib import Path

from embodied_harness import wire
from embodied_harness.batch import _commit
from embodied_harness.evaluation import freeze_knowledge, stage_knowledge, verify_staged_knowledge, verify_knowledge
from embodied_harness.runtime.relay import Plan


def partition(catalog, development, unseen, heldout):
    groups = {"development": list(development), "unseen": list(unseen), "heldout": list(heldout)}
    refs = [ref for group in groups.values() for ref in group]
    if any(not group for group in groups.values()) or any(type(ref) is not int or ref not in catalog for ref in refs):
        raise ValueError("nonempty partitions must reference actual RoboDojo layouts")
    if len(set(refs)) != len(refs):
        raise ValueError("development/unseen/heldout overlap")
    hashes = [catalog[ref]["layout_sha256"] for ref in refs]
    if len(set(hashes)) != len(hashes):
        raise ValueError("identical layout contents across partition entries")
    return {name: [{"scene_ref": ref, "layout_sha256": catalog[ref]["layout_sha256"]}
                   for ref in group] for name, group in groups.items()}


class Campaign:
    """Coordinator state is host-only. Agent contexts never contain evaluation layouts.

    open_episode(job, workspace) returns context(), run(Plan), close() and fresh
    worker_session_id/agent_session_id. All trials execute with in-episode S2 off;
    repairs happen only between complete development matrices.
    """

    def __init__(self, root, *, task, goal, partitions, frozen_knowledge, max_versions=3):
        if type(max_versions) is not int or not 1 <= max_versions <= 10:
            raise ValueError("max_versions must be 1..10")
        # Revalidate even a hand-written manifest, without needing native imports.
        catalog = {entry["scene_ref"]: entry for group in partitions.values() for entry in group}
        checked = partition(catalog, **partitions_to_refs(partitions))
        self.root, self.frozen = Path(root), Path(frozen_knowledge)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "campaign.json"
        spec = {"protocol": "aspire-fixloop-dualloop-robodojo-v1", "task": task, "goal": goal,
                "partitions": checked, "knowledge_sha256": verify_knowledge(self.frozen),
                "max_versions": max_versions}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["spec"] != spec:
                raise ValueError("campaign resume specification changed")
        else:
            self.state = {"spec": spec, "operations": {}, "trials": {}, "versions": [], "selected": None}
            self.save()
        for record in (*self.state["operations"].values(), *self.state["trials"].values()):
            if record["status"] == "running":
                record.update(status="interrupted", reason="no automatic rerun after uncertain execution")
        self.save()

    def save(self):
        _commit(self.path, self.state)

    def operation(self, key, action):
        record = self.state["operations"].get(key)
        if record:
            if record["status"] != "completed":
                raise RuntimeError(f"operation {key} is {record['status']}; preserve evidence, start a new campaign")
            return wire.decode(record["value"])
        record = self.state["operations"][key] = {"status": "running"}
        self.save()
        try:
            value = action()
            record.update(status="completed", value=wire.encode(value))
        except BaseException as exc:
            record.update(status="failed", reason=type(exc).__name__)
            raise
        finally:
            self.save()
        return value

    def ask(self, agent, role, context, key):
        def request():
            reply = agent.request(role, context)
            if set(reply) != {"agent_session_id", "reply"} or not reply["agent_session_id"]:
                raise ValueError("agent reply requires fresh session identity")
            identities = [v.get("value", {}).get("agent_session_id") for v in self.state["operations"].values() if isinstance(v.get("value"), dict)]
            if reply["agent_session_id"] in identities:
                raise ValueError("agent context/session reused between roles")
            return reply
        return self.operation(key, request)["reply"]

    def episode(self, key, scene, plan, open_episode, *, knowledge=None, preview=False):
        record = self.state["trials"].get(key)
        if record:
            if record["status"] != "completed":
                raise RuntimeError(f"trial {key} is {record['status']}; no automatic rerun")
            return wire.decode(record["result"])
        frozen = Path(knowledge or self.frozen)
        if knowledge is None and verify_knowledge(frozen) != self.state["spec"]["knowledge_sha256"]:
            raise ValueError("initial skill library changed")
        record = self.state["trials"][key] = {"status": "running", "plan_sha256": plan.fingerprint() if plan else None}
        self.save()
        driver = None
        try:
            workspace = self.root / key / "workspace"
            workspace.mkdir(parents=True, exist_ok=False)
            stage_knowledge(frozen, workspace)
            job = {"job_id": key, "task": self.state["spec"]["task"], "scene_ref": scene["scene_ref"],
                   "layout_sha256": scene["layout_sha256"], "phase": "test"}
            driver = open_episode(job, workspace)
            ids = {"worker_session_id": driver.worker_session_id, "agent_session_id": driver.agent_session_id}
            for field, identity in ids.items():
                if not isinstance(identity, str) or not identity or any(
                        old is not record and old.get(field) == identity for old in self.state["trials"].values()):
                    raise ValueError("fresh worker/agent sessions required per trial")
            record.update(ids)
            self.save()
            result = driver.context() if preview else driver.run(plan)
            verify_staged_knowledge(frozen, workspace)
            if not preview:
                reports = result.get("reports", ())
                if (result.get("system2_enabled") is not False or result.get("interventions") != 0
                        or result.get("plan_sha256") != plan.fingerprint()
                        or result.get("selected_plan") != plan.to_dict()
                        or not reports or any(r.get("provenance_audit") != "pass" for r in reports)):
                    raise ValueError("trial failed fixed-program/provenance audit")
                success = result.get("status") == "native_task_succeeded"
                if (type(result.get("primary_native_success")) is not bool
                        or result["primary_native_success"] != success
                        or result.get("recovered_native_success") is not False
                        or success != (reports[-1].get("status") == "native_task_succeeded")):
                    raise ValueError("inconsistent native outcome")
            record.update(status="completed", result=wire.encode(result))
            return result
        except BaseException as exc:
            record.update(status="failed", reason=type(exc).__name__)
            raise
        finally:
            if driver is not None:
                try:
                    driver.close()
                except Exception as exc:
                    record.update(status="failed", cleanup_error=type(exc).__name__)
            self.save()
            if record["status"] == "failed" and "cleanup_error" in record:
                raise RuntimeError("episode cleanup failed")

    def library(self, frozen=None):
        root = Path(frozen or self.frozen)
        verify_knowledge(root)
        manifest = json.loads((root / "manifest.json").read_text())
        return {name: (root / name).read_text() for name in manifest["files"]}

    def run(self, agent, open_episode):
        spec = self.state["spec"]
        development = spec["partitions"]["development"]
        public = self.episode("preview", development[0], None, open_episode, preview=True)
        base = {"goal": spec["goal"], "public": public, "library": self.library()}
        versions = self.state["versions"]
        if not versions:
            plan = Plan.from_dict(self.ask(agent, "generate", base, "generate"))
            if plan.goal != spec["goal"]:
                raise ValueError("generated program changed task goal")
            versions.append(plan.to_dict())
            self.save()
        scores = []
        for version in range(spec["max_versions"]):
            if version >= len(versions):
                break
            plan = Plan.from_dict(versions[version])
            outcomes = [self.episode(f"dev_v{version}_{index}", scene, plan, open_episode)
                        for index, scene in enumerate(development)]
            score = sum(result["primary_native_success"] for result in outcomes)
            scores.append(score)
            if score == len(development) or version + 1 == spec["max_versions"]:
                break
            if version + 1 == len(versions):
                context = {**base, "plan": plan.to_dict(), "development_reports": outcomes,
                           "instruction": "Repair the common failure mode, then retest every development scene."}
                replacement = Plan.from_dict(self.ask(agent, "repair", context, f"repair_{version}"))
                if replacement.goal != spec["goal"] or replacement.fingerprint() in [Plan.from_dict(p).fingerprint() for p in versions]:
                    raise ValueError("repair changed goal or repeated program")
                versions.append(replacement.to_dict())
                self.save()
        # Deterministic best dev score, earlier version wins ties. Unseen cannot affect selection.
        selected_index = max(range(len(scores)), key=lambda index: scores[index])
        selected = Plan.from_dict(versions[selected_index])
        selection = {"version": selected_index, "plan": selected.to_dict(), "sha256": selected.fingerprint(),
                     "development_scores": scores, "knowledge_sha256": spec["knowledge_sha256"]}
        selection = json.loads(json.dumps(selection))
        if self.state["selected"] is not None and self.state["selected"] != selection:
            raise ValueError("selected artifact changed after freeze")
        self.state["selected"] = selection
        self.save()
        _commit(self.root / "selected.json", selection)
        for phase in ("unseen", "heldout"):
            for index, scene in enumerate(spec["partitions"][phase]):
                self.episode(f"{phase}_{index}", scene, selected, open_episode)
        self.promote(agent, open_episode, selected, base, development)
        self.state["complete"] = True
        self.save()
        return self.summary()

    def promote(self, agent, open_episode, selected, base, development):
        proposal = self.ask(agent, "propose", {**base, "selected_plan": selected.to_dict(),
                            "development_reports": [wire.decode(record["result"]) for key, record in self.state["trials"].items()
                                                    if key.startswith("dev_")]}, "propose")
        if set(proposal) != {"skills", "skill_report"} or not isinstance(proposal["skills"], dict) or not isinstance(proposal["skill_report"], (list, tuple)):
            raise ValueError("proposal requires skills mapping and skill_report")
        for grade in proposal["skill_report"]:
            if (set(grade) != {"skill", "verdict", "evidence"} or grade["skill"] not in base["library"]
                    or grade["verdict"] not in ("helpful", "unhelpful", "unused")
                    or not isinstance(grade["evidence"], str) or not grade["evidence"].strip()):
                raise ValueError("skill grades must cite existing library entries and development evidence")
        if {grade["skill"] for grade in proposal["skill_report"]} != set(base["library"]):
            raise ValueError("grade every existing library entry, including unused skills")
        _commit(self.root / "skill_report.json", list(proposal["skill_report"]))
        if not proposal["skills"]:
            self.state["promotion"] = {"status": "no_proposal"}
            return
        workspace = self.root / "candidate_library"
        def candidate():
            workspace.mkdir(exist_ok=False)
            stage_knowledge(self.frozen, workspace)
            for name, content in proposal["skills"].items():
                path = Path(name)
                if (len(path.parts) != 2 or path.parts[0] != "skills" or path.suffix not in (".md", ".py")
                        or not path.stem.isidentifier() or not isinstance(content, str) or not 0 < len(content) <= 100_000):
                    raise ValueError("invalid skill proposal path/content")
                if path.suffix == ".py":
                    from embodied_harness.programs.policy import validate
                    validate(content)
                target = workspace / path
                if target.exists():
                    raise ValueError("proposal may not overwrite existing skills")
                target.parent.mkdir(exist_ok=True)
                target.write_text(content)
            return freeze_knowledge(workspace, self.root / "candidate_frozen")
        self.operation("candidate_library", candidate)
        verified = False
        # Independent reconstruction: verifier receives library+public goal, never proposer solution/reports.
        for attempt in range(2):
            context = {"goal": base["goal"], "public": base["public"], "library": self.library(self.root / "candidate_frozen")}
            plan = Plan.from_dict(self.ask(agent, "verify", context, f"verify_{attempt}"))
            if plan.goal != base["goal"]:
                raise ValueError("verifier changed task goal")
            outcomes = [self.episode(f"verify_{attempt}_{index}", scene, plan, open_episode,
                                    knowledge=self.root / "candidate_frozen")
                        for index, scene in enumerate(development[:2])]
            if all(result["primary_native_success"] for result in outcomes):
                verified = True
                break
        self.state["promotion"] = {"status": "verified" if verified else "rejected",
                                   "library_sha256": verify_knowledge(self.root / "candidate_frozen")}
        if verified:
            self.operation("promotion", lambda: freeze_knowledge(workspace, self.root / "promoted"))
        self.save()

    def summary(self):
        summary = {"complete": self.state.get("complete", False), "selected": self.state["selected"],
                   "promotion": self.state.get("promotion")}
        for phase in ("unseen", "heldout"):
            records = [record for key, record in self.state["trials"].items() if key.startswith(phase + "_")]
            summary[phase] = {"denominator": len(self.state["spec"]["partitions"][phase]),
                              "native_successes": sum(record.get("result", {}).get("primary_native_success", False) for record in records)}
        return summary


def partitions_to_refs(partitions):
    if set(partitions) != {"development", "unseen", "heldout"}:
        raise ValueError("three explicit partitions required")
    return {name: [entry["scene_ref"] for entry in group] for name, group in partitions.items()}
