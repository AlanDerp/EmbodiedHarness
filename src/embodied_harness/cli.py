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
from embodied_harness.runtime.selectors import FixedSelector, JevSelector, RemoteSelector


def _token(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required token environment variable missing: {name}")
    return value


def build_session(config):
    allowed = {"worker_url", "worker_token_env", "run_dir", "max_submissions", "sam3_url",
               "selector", "jev_model", "timeout_s", "rpc_token_env", "host", "port", "extensions", "skill_python",
               "system2_url", "system2_token_env", "max_interventions", "sam3_model_version",
               "system1_url", "system1_token_env"}
    if set(config) - allowed:
        raise ValueError("unknown harness config fields")
    if config.get("selector", "fixed") not in ("fixed", "jev", "remote"):
        raise ValueError("unknown selector")
    if config.get("selector") in ("jev", "remote") and not config.get("jev_model"):
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
    provider = Sam3Provider(config["sam3_url"], expected_model_version=config.get("sam3_model_version")) if config.get("sam3_url") else None
    kind = config.get("selector", "fixed")
    if kind == "remote":
        selector = RemoteSelector(RpcClient(config["system1_url"], _token(config["system1_token_env"]), timeout_s=35), config["jev_model"])
    else:
        selector = FixedSelector() if kind == "fixed" else JevSelector(config["jev_model"])
    session = Session(Executor(adapter, max_submissions=config.get("max_submissions", 1000)),
                   store, EventLog(run / "events.jsonl"), selector=selector,
                   perception=Perception(store, provider), skill_python=config.get("skill_python"))
    for extension in config.get("extensions", []):
        module, function = extension.split(":", 1)
        getattr(importlib.import_module(module), function)(session)
    if config.get("system2_url"):
        from embodied_harness.runtime.relay import Relay, RemoteSystem2
        bridge = RemoteSystem2(RpcClient(config["system2_url"], _token(config["system2_token_env"]), timeout_s=240))
        session.relay = Relay(session, bridge, max_interventions=config.get("max_interventions", 2))
    return session


def run_campaign(config, mode="campaign"):
    from embodied_harness.agents import RemoteCampaignAgent
    from embodied_harness.campaign import Campaign
    from embodied_harness.deployment import ProvisionedEpisodes
    import re
    harness = config["harness"]
    if not re.fullmatch(r"sam3:code:[0-9a-f]{64}:checkpoint:[0-9a-f]{64}", harness.get("sam3_model_version", "")):
        raise ValueError("campaign requires pinned real SAM3 code/checkpoint model version")
    if mode == "campaign":
        manifest = json.loads(Path(config["partition_manifest"]).read_text())
        runner = Campaign(config["run_dir"], task=manifest["task"], goal=config["goal"],
                          partitions=manifest["partitions"], frozen_knowledge=config["frozen_knowledge"],
                          max_versions=config.get("max_versions", 3),
                          protocol=config.get("protocol", "aspire-fixloop-dualloop-robodojo-v1"))
    else:
        from embodied_harness.curriculum import Curriculum
        tasks = []
        for task in config["tasks"]:
            manifest = json.loads(Path(task["partition_manifest"]).read_text())
            tasks.append({"task_id": task["task_id"], "goal": task["goal"], **manifest})
        runner = Curriculum(config["run_dir"], tasks, config["frozen_knowledge"],
                            rerun_threshold=config.get("rerun_threshold", .8), max_versions=config.get("max_versions", 3))
    agent = RemoteCampaignAgent(RpcClient(config["agent_url"], _token(config["agent_token_env"]), timeout_s=config.get("agent_timeout_s", 600)))
    episodes = ProvisionedEpisodes(RpcClient(config["provisioner_url"], _token(config["provisioner_token_env"]), timeout_s=600), harness)
    return runner.run(agent, episodes)


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
    native = commands.add_parser("native-worker", help="start provisioned RoboDojo without an external factory")
    native.add_argument("--config", required=True)
    native.add_argument("--backend", choices=["robodojo", "libero_plus"], default="robodojo")
    native.add_argument("--profile", required=True)
    native.add_argument("--episode-id", required=True)
    native.add_argument("--phase", choices=["playground", "test"], default="playground")
    native.add_argument("--host", default="127.0.0.1")
    native.add_argument("--port", type=int, default=8120)
    native.add_argument("--token-env", default="EMBODIED_WORKER_TOKEN")
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
    sam3 = commands.add_parser("check-sam3", help="validate real SAM3 against a captured public snapshot")
    for name in ("url", "store", "episode", "snapshot", "camera", "query", "checkpoint-sha256", "log"):
        sam3.add_argument("--" + name, required=True)
    partitions = commands.add_parser("partition", help="lock host-only numeric layout partitions")
    for name in ("runtime-root", "task", "refs", "output"):
        partitions.add_argument("--" + name, required=True)
    campaign = commands.add_parser("campaign", help="ASPIRE dev repair, selection, unseen, heldout, skill verification")
    campaign.add_argument("--config", required=True)
    curriculum = commands.add_parser("curriculum", help="multi-task skill transfer and one final-library rerun sweep")
    curriculum.add_argument("--config", required=True)
    category = commands.add_parser("test-category", help="select a configured RoboDojo category; preflight then campaign")
    category.add_argument("--registry", default="configs/categories.local.json")
    choice = category.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list", action="store_true")
    choice.add_argument("--category")
    category.add_argument("--dry-run", action="store_true", help="validate config/partitions/library without services or writes")
    diagnostics = commands.add_parser("doctor", help="read-only campaign service readiness, no actions/model generation")
    diagnostics.add_argument("--config", required=True)
    agents = commands.add_parser("agent-service", help="run isolated role conversations on the System 2 device")
    agent_source = agents.add_mutually_exclusive_group(required=True)
    agent_source.add_argument("--factory", help="trusted module:function constructing a fresh model driver")
    agent_source.add_argument("--config", help="built-in Claude Code driver configuration")
    agents.add_argument("--host", default="127.0.0.1")
    agents.add_argument("--port", type=int, default=8123)
    agents.add_argument("--token-env", default="EMBODIED_AGENT_TOKEN")
    system1 = commands.add_parser("system1-service", help="serve pinned Jev numeric choices with private provider credentials")
    system1.add_argument("--config", required=True)
    provision = commands.add_parser("provisioner", help="ordinary subprocess lifecycle on the simulator device")
    provision.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "system1-service":
        from embodied_harness.services.system1 import System1Service
        config = json.loads(Path(args.config).read_text())
        address = (config.pop("host", "127.0.0.1"), config.pop("port", 8124))
        token = _token(config.pop("rpc_token_env"))
        audit = EventLog(config.pop("audit_log"))
        service = System1Service(JevSelector(**config), audit)
        with RpcServer(address, token, service.methods()) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    if args.command == "test-category":
        from embodied_harness.categories import categories, select_category
        from embodied_harness.diagnostics import doctor
        if args.list:
            print(json.dumps(categories(args.registry)))
            return
        config, report = select_category(args.registry, args.category)
        if args.dry_run:
            print(json.dumps(report))
            return
        doctor(config)
        print(json.dumps(run_campaign(config)))
        return
    if args.command == "doctor":
        from embodied_harness.diagnostics import doctor
        print(json.dumps(doctor(json.loads(Path(args.config).read_text()))))
        return
    if args.command == "partition":
        from embodied_harness.campaign import partition
        from embodied_harness.adapters.robodojo_native import layout_catalog
        manifest = partition(layout_catalog(args.runtime_root, args.task), **json.loads(Path(args.refs).read_text()))
        with Path(args.output).open("x") as output:
            json.dump({"task": args.task, "partitions": manifest}, output, sort_keys=True)
        return
    if args.command in ("campaign", "curriculum"):
        print(json.dumps(run_campaign(json.loads(Path(args.config).read_text()), args.command)))
        return
    if args.command == "agent-service":
        from embodied_harness.agents import AgentService
        if args.config:
            from embodied_harness.drivers.claude_code import ClaudeCodeDriver
            config = json.loads(Path(args.config).read_text())
            driver_name = config.pop("driver", "claude-code")
            if driver_name == "deepseek-api":
                from embodied_harness.drivers.deepseek_api import DeepSeekApiDriver as Driver
            elif driver_name == "claude-code":
                Driver = ClaudeCodeDriver
            else:
                raise ValueError("unknown model driver")
            address = (config.pop("host", "127.0.0.1"), config.pop("port", 8123))
            token = _token(config.pop("rpc_token_env"))
            driver = Driver(**config)
            service = AgentService(lambda: Driver(**config), lambda: {
                "driver": driver_name, "model": driver.model,
                "credentials_configured": driver.configured()})
        else:
            module, function = args.factory.split(":", 1)
            service = AgentService(getattr(importlib.import_module(module), function))
            address, token = (args.host, args.port), _token(args.token_env)
        with RpcServer(address, token, service.methods()) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    if args.command == "provisioner":
        from embodied_harness.deployment import NativeProvisioner
        config = json.loads(Path(args.config).read_text())
        address = (config.pop("host", "127.0.0.1"), config.pop("port", 8122))
        token = _token(config.pop("rpc_token_env"))
        provisioner = NativeProvisioner(**config)
        try:
            with RpcServer(address, token, provisioner.methods()) as server:
                server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            provisioner.close()
        return
    if args.command == "check-sam3":
        from embodied_harness.diagnostics import check_sam3
        report = check_sam3(args.url, args.store, args.episode, args.snapshot, args.camera,
                            args.query, args.checkpoint_sha256, args.log)
        print(json.dumps({"passed": report["passed"], "model_version": report["model_version"]}))
        if not report["passed"]:
            raise SystemExit(1)
        return
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
    if args.command in ("worker", "native-worker"):
        token = _token(args.token_env)
        if args.command == "native-worker":
            if args.backend == "libero_plus":
                from embodied_harness.adapters.libero_plus import from_config
            else:
                from embodied_harness.adapters.robodojo_native import from_config
            backend = from_config(args.config)
            limit = backend.action_limit
        else:
            module, function = args.factory.split(":", 1)
            backend = getattr(importlib.import_module(module), function)()
            limit = args.native_action_limit
        instance = RoboDojoWorker(backend, RobotProfile.from_file(args.profile), args.episode_id,
                                 native_action_limit=limit, phase=args.phase)
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
