"""Host-only skills-transfer batch ledger with no automatic rerun of interrupted trials."""

import json
import os
from pathlib import Path
import tempfile

from embodied_harness.evaluation import stage_knowledge, verify_knowledge, verify_staged_knowledge


def _commit(path, value):
    handle, temporary = tempfile.mkstemp(prefix=".ledger-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class BatchRunner:
    def __init__(self, root, jobs, frozen_knowledge):
        self.root, self.frozen = Path(root), Path(frozen_knowledge)
        frozen_hash = verify_knowledge(self.frozen)
        jobs = json.loads(json.dumps(list(jobs), allow_nan=False))
        if not jobs or any(not isinstance(job, dict) or set(job) != {"job_id", "task", "scene_ref"} for job in jobs):
            raise ValueError("jobs require job_id, task and host-only scene_ref")
        ids = [job["job_id"] for job in jobs]
        if len(set(ids)) != len(ids) or any(not isinstance(key, str) or not key.isidentifier() for key in ids):
            raise ValueError("job IDs must be unique path-safe identifiers")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "batch.json"
        plan = {"protocol": "skills-transfer-v1", "frozen_hash": frozen_hash, "jobs": jobs}
        if self.path.exists():
            self.ledger = json.loads(self.path.read_text())
            if self.ledger["plan"] != plan:
                raise ValueError("resume plan or frozen knowledge differs")
        else:
            self.ledger = {"plan": plan, "results": {job["job_id"]: {"status": "pending"} for job in jobs}}
            _commit(self.path, self.ledger)

    def run(self, open_episode):
        """Factory returns a fresh episode driver with worker/agent IDs, run and close.

        The deployment owns simulator/agent provisioning. This runner never sees
        credentials or selects scenes through agent tools.
        """
        used_workers, used_agents = set(), set()
        for record in self.ledger["results"].values():
            if "worker_session_id" in record:
                used_workers.add(record["worker_session_id"])
                used_agents.add(record["agent_session_id"])
            if record["status"] == "running":
                record.update(status="interrupted", reason="previous process stopped; no automatic rerun")
        _commit(self.path, self.ledger)
        for job in self.ledger["plan"]["jobs"]:
            record = self.ledger["results"][job["job_id"]]
            if record["status"] != "pending":
                continue
            if verify_knowledge(self.frozen) != self.ledger["plan"]["frozen_hash"]:
                raise ValueError("knowledge changed during batch")
            directory = self.root / job["job_id"]
            directory.mkdir(exist_ok=False)
            record["status"] = "running"
            _commit(self.path, self.ledger)
            driver = None
            try:
                workspace = directory / "workspace"
                stage_knowledge(self.frozen, workspace)
                driver = open_episode(dict(job), workspace)
                worker_id, agent_id = driver.worker_session_id, driver.agent_session_id
                if not isinstance(worker_id, str) or not worker_id or not isinstance(agent_id, str) or not agent_id:
                    raise ValueError("fresh worker and agent session IDs required")
                if worker_id in used_workers or agent_id in used_agents:
                    raise ValueError("worker/agent session reused across Test scenes")
                used_workers.add(worker_id)
                used_agents.add(agent_id)
                record.update(worker_session_id=worker_id, agent_session_id=agent_id)
                _commit(self.path, self.ledger)
                result = driver.run()
                verify_staged_knowledge(self.frozen, workspace)
                required = {"primary_native_success", "recovered_native_success", "interventions"}
                if not isinstance(result, dict) or not required <= result.keys():
                    raise ValueError("episode result missing explicit native outcomes")
                if any(type(result[key]) is not bool for key in required - {"interventions"}):
                    raise ValueError("native success fields must be boolean")
                if type(result["interventions"]) is not int or result["interventions"] < 0:
                    raise ValueError("interventions must be a nonnegative integer")
                if (result["primary_native_success"] and result["recovered_native_success"]
                        or result["recovered_native_success"] and result["interventions"] == 0):
                    raise ValueError("inconsistent primary/recovered outcome")
                result = json.loads(json.dumps(result, allow_nan=False))
                record.update(status="completed", result=result)
            except Exception as exc:
                record.update(status="failed", reason=type(exc).__name__)
            finally:
                if driver is not None:
                    try:
                        driver.close()
                    except Exception as exc:
                        record["cleanup_error"] = type(exc).__name__
                _commit(self.path, self.ledger)
        return self.summary()

    def summary(self):
        values = list(self.ledger["results"].values())
        return {"denominator": len(values),
                "primary_native_successes": sum(v.get("result", {}).get("primary_native_success", False) for v in values),
                "recovered_native_successes": sum(v.get("result", {}).get("recovered_native_success", False) for v in values),
                "failed": sum(v["status"] == "failed" for v in values),
                "interrupted": sum(v["status"] == "interrupted" for v in values),
                "pending": sum(v["status"] == "pending" for v in values)}
