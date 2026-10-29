"""Append-only event evidence. Write errors stop the caller."""

from pathlib import Path
import time

from embodied_harness.wire import dumps


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind, **data):
        event = {"monotonic_s": time.monotonic(), "kind": kind, **data}
        with self.path.open("ab") as stream:
            stream.write(dumps(event) + b"\n")
            stream.flush()
