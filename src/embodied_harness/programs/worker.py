"""Subprocess Python namespace. Host validates every primitive call independently."""

import contextlib
import io
import sys
import traceback

from embodied_harness import wire


class BoundedOutput(io.StringIO):
    def write(self, value):
        remaining = max(0, 8192 - self.tell())
        super().write(value[:remaining])
        return len(value)


def main():
    channel_in, channel_out = sys.stdin.buffer, sys.stdout.buffer

    def send(value):
        channel_out.write(wire.dumps(value) + b"\n")
        channel_out.flush()

    def invoke(method, **params):
        send({"kind": "call", "method": method, "params": params})
        response = wire.loads(channel_in.readline(wire.MAX_BYTES + 1))
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    namespace = {"__name__": "__skill__"}
    while True:
        line = channel_in.readline(wire.MAX_BYTES + 1)
        if not line:
            return
        command = wire.loads(line)
        output, errors = BoundedOutput(), BoundedOutput()
        result = {"kind": "done", "ok": False, "exports": {}}
        try:
            bindings = command.get("bindings", {})
            if any(not isinstance(name, str) or not name.isidentifier() or name.startswith("_") for name in bindings):
                raise ValueError("invalid program binding")
            namespace.update(bindings)
            namespace.update(get_observation=lambda: invoke("get_observation"),
                             step=lambda action: invoke("step", action=action),
                             localize_object=lambda **params: invoke("localize_object", **params))
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                exec(compile(command["code"], "<skill>", "exec"), namespace)
            result["exports"] = {key: namespace[key] for key in command.get("exports", ())}
            wire.dumps(result["exports"])
            result["ok"] = True
        except BaseException as exc:
            result["exports"] = {}
            result["error_type"] = type(exc).__name__
            result["traceback"] = traceback.format_exc(limit=8)[-8192:]
        result.update(stdout=output.getvalue(), stderr=errors.getvalue())
        send(result)


if __name__ == "__main__":
    main()
