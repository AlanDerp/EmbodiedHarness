"""Persistent Python subprocess and host-owned per-execution action-call gate."""

import os
import select
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from embodied_harness import wire
from embodied_harness.contracts.types import _count, _number, _text


class PersistentProgram:
    def __init__(self, methods, *, python_executable=None, timeout_s=30):
        executable = python_executable or sys.executable
        _text(executable, "python_executable")
        _number(timeout_s, "timeout_s")
        if timeout_s == 0:
            raise ValueError("positive timeout required")
        self.methods, self.timeout_s, self.buffer = dict(methods), timeout_s, b""
        self.workspace = tempfile.TemporaryDirectory(prefix="embodied-skill-")
        self.closed = False
        env = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL", "TZ"}}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            self.process = subprocess.Popen([
                executable, "-u", "-m", "embodied_harness.programs.worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                cwd=self.workspace.name, env=env, start_new_session=True)
        except BaseException:
            self.workspace.cleanup()
            raise

    def _send(self, value, deadline):
        raw = memoryview(wire.dumps(value) + b"\n")
        fd = self.process.stdin.fileno()
        os.set_blocking(fd, False)
        while raw:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise TimeoutError("skill input deadline exceeded")
            try:
                count = os.write(fd, raw[:4096])
                raw = raw[count:]
            except BlockingIOError:
                continue

    def _read(self, deadline):
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("skill program deadline exceeded")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("skill process exited without a result")
            self.buffer += chunk
            if len(self.buffer) > wire.MAX_BYTES:
                raise ValueError("skill output exceeds byte limit")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return wire.loads(line)

    def execute(self, code, *, bindings=None, exports=(), max_actions=0):
        _text(code, "code")
        _count(max_actions, "max_actions")
        if max_actions not in (0, 1):
            raise ValueError("skill hooks permit zero or one action only")
        deadline, calls, violation = time.monotonic() + self.timeout_s, 0, False
        try:
            self._send({"code": code, "bindings": bindings or {}, "exports": list(exports)}, deadline)
            while True:
                message = self._read(deadline)
                if not isinstance(message, dict):
                    raise ValueError("invalid child protocol")
                if message.get("kind") == "done":
                    message["action_calls"] = calls
                    if violation:
                        message["ok"] = False
                        message["error_type"] = "ActionBudgetViolation"
                    return message
                if message.get("kind") != "call" or set(message) != {"kind", "method", "params"}:
                    raise ValueError("invalid child primitive request")
                method = message["method"]
                try:
                    if method not in {"get_observation", "step", "localize_object"} or method not in self.methods:
                        raise PermissionError("primitive unavailable")
                    if method == "step":
                        if calls >= max_actions:
                            violation = True
                            raise RuntimeError("hook action budget exceeded")
                        calls += 1
                    value = self.methods[method](**message["params"])
                    self._send({"result": value}, deadline)
                except Exception as exc:
                    self._send({"error": type(exc).__name__}, deadline)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.stdin.close()
        self.process.stdout.close()
        self.workspace.cleanup()


class PythonSkill:
    def __init__(self, program: PersistentProgram):
        self.program = program

    def initialize(self, source, parameters):
        return self.program.execute(source + "\nskill_state = skill_init(parameters)\n",
                                    bindings={"parameters": parameters}, exports=("skill_state",), max_actions=0)

    def refresh(self):
        return self.program.execute("feedback = skill_observe(skill_state)\n"
                                    "context = skill_context(skill_state)\noptions = skill_options(skill_state)\n",
                                    exports=("feedback", "context", "options"), max_actions=0)

    def tick(self, option):
        return self.program.execute("feedback = skill_tick(skill_state, selected_option)\n",
                                    bindings={"selected_option": option},
                                    exports=("feedback",), max_actions=1)
