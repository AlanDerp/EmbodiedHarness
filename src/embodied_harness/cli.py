"""Local composition and explicit remote service entrypoints."""

import argparse
import importlib
import json
import os
from pathlib import Path

from embodied_harness.adapters.robodojo import RemoteRoboDojo, RoboDojoWorker
from embodied_harness.artifacts import SnapshotStore
from embodied_harness.artifacts.events import EventLog
from embodied_harness.control import Executor
from embodied_harness.batch import BatchRunner
from embodied_harness.embodiments import RobotProfile
from embodied_harness.evaluation import freeze_knowledge, verify_knowledge
from embodied_harness.perception import Perception
from embodied_harness.providers.sam3 import Sam3Provider
from embodied_harness.rpc import RpcClient, RpcServer
from embodied_harness.runtime import Session
from embodied_harness.runtime.selectors import FixedSelector, JevSelector


def _token(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required token environment variable missing: {name}")
    return value


def build_session(config):
    allowed = {"worker_url", "worker_token_env", "run_dir", "max_submissions", "sam3_url",
               "selector", "jev_model", "timeout_s", "rpc_token_env", "host", "port", "extensions", "skill_python"}
    if set(config) - allowed:
        raise ValueError("unknown harness config fields")
    if config.get("selector", "fixed") not in ("fixed", "jev"):
        raise ValueError("unknown selector")
    if config.get("selector") == "jev" and not config.get("jev_model"):
        raise ValueError("pinned Jev model required")
    RpcClient(config["worker_url"], _token(config["worker_token_env"]),
              timeout_s=config.get("timeout_s", 60))
    if config.get("sam3_url"):
        Sam3Provider(config["sam3_url"])
    run = Path(config["run_dir"])
    run.mkdir(parents=True, exist_ok=False)
    # No credentials are written into the run directory.
    (run / "config.json").write_text(json.dumps(config, sort_keys=True))
    adapter = RemoteRoboDojo(RpcClient(config["worker_url"], _token(config["worker_token_env"]),
                                      timeout_s=config.get("timeout_s", 60)))
    store = SnapshotStore(run / "snapshots")
    provider = Sam3Provider(config["sam3_url"]) if config.get("sam3_url") else None
    kind = config.get("selector", "fixed")
    if kind not in ("fixed", "jev"):
        raise ValueError("unknown selector")
    selector = FixedSelector() if kind == "fixed" else JevSelector(config["jev_model"])
    session = Session(Executor(adapter, max_submissions=config.get("max_submissions", 1000)),
                   store, EventLog(run / "events.jsonl"), selector=selector,
                   perception=Perception(store, provider), skill_python=config.get("skill_python"))
    for extension in config.get("extensions", []):
        module, function = extension.split(":", 1)
        getattr(importlib.import_module(module), function)(session)
    return session


def main():
    parser = argparse.ArgumentParser(description="Embodied harness remote composition")
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("serve", help="host tools for remote System 2")
    host.add_argument("--config", required=True)
    worker = commands.add_parser("worker", help="run on the Isaac Sim device")
    worker.add_argument("--factory", required=True, help="trusted installed module:function returning native backend")
    worker.add_argument("--profile", required=True)
    worker.add_argument("--episode-id", required=True)
    worker.add_argument("--native-action-limit", type=int, required=True)
    worker.add_argument("--phase", choices=["playground", "test"], default="playground")
    worker.add_argument("--host", default="127.0.0.1")
    worker.add_argument("--port", type=int, default=8120)
    worker.add_argument("--token-env", default="EMBODIED_WORKER_TOKEN")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("workspace")
    freeze.add_argument("destination")
    verify = commands.add_parser("verify")
    verify.add_argument("directory")
    check = commands.add_parser("validate-profile", help="local profile schema check, not physical commissioning")
    check.add_argument("profile")
    batch = commands.add_parser("batch", help="host-only frozen-knowledge batch with explicit deployment driver")
    batch.add_argument("--plan", required=True)
    batch.add_argument("--frozen", required=True)
    batch.add_argument("--run-dir", required=True)
    batch.add_argument("--factory", required=True, help="trusted module:open_episode(job, workspace)")
    args = parser.parse_args()
    if args.command == "batch":
        module, function = args.factory.split(":", 1)
        runner = BatchRunner(args.run_dir, json.loads(Path(args.plan).read_text()), args.frozen)
        print(json.dumps(runner.run(getattr(importlib.import_module(module), function))))
        return
    if args.command == "validate-profile":
        profile = RobotProfile.from_file(args.profile)
        print(json.dumps({"schema_valid": True, "commissioned": profile.commissioned,
                          "calibration_id": profile.calibration_id}))
        return
    if args.command == "freeze":
        print(freeze_knowledge(args.workspace, args.destination))
        return
    if args.command == "verify":
        print(verify_knowledge(args.directory))
        return
    backend = None
    if args.command == "worker":
        token = _token(args.token_env)
        module, function = args.factory.split(":", 1)
        backend = getattr(importlib.import_module(module), function)()
        instance = RoboDojoWorker(backend, RobotProfile.from_file(args.profile), args.episode_id,
                                 native_action_limit=args.native_action_limit, phase=args.phase)
        address, methods = (args.host, args.port), instance.methods()
    else:
        config = json.loads(Path(args.config).read_text())
        token = _token(config["rpc_token_env"])
        instance = build_session(config)
        address, methods = (config.get("host", "127.0.0.1"), config.get("port", 8121)), instance.tools()
    try:
        with RpcServer(address, token, methods,
                       concurrent_methods=("stop",) if args.command == "serve" else ()) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
