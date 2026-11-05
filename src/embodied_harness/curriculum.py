"""Ordered multi-task knowledge transfer and one separately labelled ASPIRE rerun sweep."""

from pathlib import Path
import json

from embodied_harness.batch import _commit
from embodied_harness.campaign import Campaign
from embodied_harness.evaluation import verify_knowledge


class Curriculum:
    def __init__(self, root, tasks, frozen_knowledge, *, rerun_threshold=.8, max_versions=3):
        if not tasks or any(set(task) != {"task_id", "task", "goal", "partitions"} for task in tasks):
            raise ValueError("curriculum tasks require task_id/task/goal/partitions")
        ids = [task["task_id"] for task in tasks]
        if len(set(ids)) != len(ids) or any(not isinstance(key, str) or not key.isidentifier() for key in ids):
            raise ValueError("unique path-safe curriculum task IDs required")
        if isinstance(rerun_threshold, bool) or not 0 <= rerun_threshold <= 1:
            raise ValueError("rerun threshold must be 0..1")
        self.root, self.tasks, self.frozen = Path(root), tasks, Path(frozen_knowledge)
        self.root.mkdir(parents=True, exist_ok=True)
        self.threshold, self.max_versions = rerun_threshold, max_versions
        spec = json.loads(json.dumps({"tasks": tasks, "initial_library_sha256": verify_knowledge(self.frozen),
                                      "rerun_threshold": rerun_threshold, "max_versions": max_versions}))
        path = self.root / "spec.json"
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError("curriculum order/conditions/library changed on resume")
        if not path.exists():
            _commit(path, spec)

    def run(self, agent, open_episode):
        library, results, learned_under = self.frozen, {}, {}
        _commit(self.root / "curriculum.json", {"order": [task["task_id"] for task in self.tasks],
                "reason": "operator-defined task order", "rerun_threshold": self.threshold})
        for task in self.tasks:
            key = task["task_id"]
            learned_under[key] = verify_knowledge(library)
            runner = Campaign(self.root / key / "attempt_01", task=task["task"], goal=task["goal"],
                              partitions=task["partitions"], frozen_knowledge=library, max_versions=self.max_versions)
            results[key] = runner.run(agent, open_episode)
            if results[key]["promotion"]["status"] == "verified":
                library = runner.root / "promoted"
        final_hash = verify_knowledge(library)
        reruns = {}
        for task in self.tasks:
            key = task["task_id"]
            score = results[key]["heldout"]["native_successes"] / results[key]["heldout"]["denominator"]
            if score < self.threshold and learned_under[key] != final_hash:
                # New library = separate experiment; never overwrite the original held-out result.
                runner = Campaign(self.root / key / "rerun_02", task=task["task"], goal=task["goal"],
                                  partitions=task["partitions"], frozen_knowledge=library, max_versions=self.max_versions)
                reruns[key] = runner.run(agent, open_episode)
        report = {"attempt_01": results, "rerun_02": reruns, "final_library": str(library),
                  "final_library_sha256": final_hash, "rerun_condition": "final-library-separate-sweep"}
        _commit(self.root / "results.json", report)
        return report
