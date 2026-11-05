"""Host-only category selection; all execution stays in the campaign protocol."""

import json
from pathlib import Path
import re

from embodied_harness.campaign import partition, partitions_to_refs
from embodied_harness.evaluation import verify_knowledge
from embodied_harness.rpc import RpcClient


def categories(path):
    entries = json.loads(Path(path).read_text())["categories"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("nonempty category mapping required")
    for name, entry in entries.items():
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                or set(entry) != {"task", "config"}
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", entry["task"])
                or not isinstance(entry["config"], str) or not entry["config"]):
            raise ValueError("category requires a native task and campaign config path")
    return entries


def select_category(path, name):
    entries = categories(path)
    if name not in entries:
        raise ValueError(f"unknown category {name!r}; available: {', '.join(entries)}")
    entry = entries[name]
    config_path = (Path(path).resolve().parent / entry["config"]).resolve()
    config = json.loads(config_path.read_text())
    harness = config["harness"]
    if not re.fullmatch(r"sam3:code:[0-9a-f]{64}:checkpoint:[0-9a-f]{64}", harness.get("sam3_model_version", "")):
        raise ValueError("explicit real SAM3 code/checkpoint pin required")
    selector = harness.get("selector", "fixed")
    if selector not in ("fixed", "jev") or (selector == "jev" and not harness.get("jev_model")):
        raise ValueError("valid selector and pinned Jev model required")
    for service in ("agent", "provisioner"):
        RpcClient(config[service + "_url"], "configuration-check-only")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config[service + "_token_env"]):
            raise ValueError("token environment variable name required")
    RpcClient(harness["sam3_url"], "configuration-check-only")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", harness["worker_token_env"]):
        raise ValueError("worker token environment variable name required")
    # Resolve campaign paths against its file, independent of the caller's cwd.
    for key in ("partition_manifest", "frozen_knowledge", "run_dir"):
        config[key] = str((config_path.parent / config[key]).resolve())
    manifest = json.loads(Path(config["partition_manifest"]).read_text())
    if manifest["task"] != entry["task"]:
        raise ValueError("category native task differs from partition manifest")
    if not isinstance(config["goal"], str) or not config["goal"].strip() or "REPLACE_WITH" in config["goal"]:
        raise ValueError("real public task instruction required")
    groups = manifest["partitions"]
    catalog = {item["scene_ref"]: item for group in groups.values() for item in group}
    partition(catalog, **partitions_to_refs(groups))
    verify_knowledge(config["frozen_knowledge"])
    if type(config.get("max_versions", 3)) is not int or not 1 <= config.get("max_versions", 3) <= 10:
        raise ValueError("max_versions must be 1..10")
    return config, {"category": name, "task": manifest["task"], "goal": config["goal"],
                    "config": str(config_path), "run_dir": config["run_dir"],
                    "partitions": {key: len(group) for key, group in groups.items()},
                    "scope": "configuration_only_no_services_models_or_physics"}
