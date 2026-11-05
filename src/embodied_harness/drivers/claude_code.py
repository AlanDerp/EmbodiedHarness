"""Claude Code print-mode driver with DeepSeek's Anthropic-compatible endpoint."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import base64
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import uuid

import numpy as np

from embodied_harness.contracts import Observation
from embodied_harness.contracts.types import _number, _text
from embodied_harness.credentials import read_key


def public_message(context):
    """Actual RGB becomes image input; depth stays numerical metadata, never a fake RGB image."""
    from PIL import Image
    images = []
    def encode(value):
        if isinstance(value, Observation):
            cameras = {}
            for name, camera in value.cameras.items():
                stream = io.BytesIO()
                Image.fromarray(camera.rgb).save(stream, format="PNG")
                label = f"snapshot={value.source.snapshot_id}, camera={name}"
                images.extend([{"type": "text", "text": label}, {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": base64.b64encode(stream.getvalue()).decode()}}])
                cameras[name] = {"image_label": label, "intrinsics": camera.intrinsics.tolist(),
                    "T_world_from_camera_optical": camera.T_world_from_camera_optical.tolist(),
                    "depth_m": encode(camera.depth_m)}
            return {"source": encode(value.source), "instruction": value.instruction,
                    "robot_state": encode(value.robot_state), "cameras": cameras}
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, np.ndarray):
            if value.size <= 256:
                return encode(value.tolist())
            finite = value[np.isfinite(value)]
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "finite_fraction": float(finite.size/value.size),
                    "min": float(finite.min()) if finite.size else None,
                    "max": float(finite.max()) if finite.size else None,
                    "note": "summary only; generated skills use public tools for full data"}
        if is_dataclass(value):
            return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Mapping):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError("unsupported public model context")
    text = json.dumps(encode(context), allow_nan=False)
    if len(images) > 24:
        raise ValueError("too many public images in one role request")
    message = {"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": text}, *images]}}
    raw = json.dumps(message, allow_nan=False).encode() + b"\n"
    if len(raw) > 32*1024*1024:
        raise ValueError("model context exceeds byte limit")
    return raw


class ClaudeCodeDriver:
    def __init__(self, executable, credentials_file, *, model="deepseek-flash[1m]",
                 base_url="https://api.deepseek.com/anthropic", timeout_s=540, workspace_root=None,
                 audit_dir=None):
        self.executable = str(Path(executable).resolve(strict=True))
        self.credentials_file = str(Path(credentials_file).absolute())
        _text(model, "model")
        _number(timeout_s, "timeout_s")
        if not 0 < timeout_s <= 1800 or base_url != "https://api.deepseek.com/anthropic":
            raise ValueError("bounded timeout and official DeepSeek endpoint required")
        self.model, self.base_url, self.timeout = model, base_url, timeout_s
        self.workspace_root = workspace_root
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.last_metadata = None

    def configured(self):
        return read_key(self.credentials_file, "DEEPSEEK_API_KEY", required=False) is not None

    def respond(self, role, instructions, context):
        key = read_key(self.credentials_file, "DEEPSEEK_API_KEY")
        payload = public_message(context)
        with tempfile.TemporaryDirectory(prefix="role-", dir=self.workspace_root) as directory:
            config = Path(directory)/"claude-config"
            config.mkdir()
            env = {name: value for name, value in os.environ.items() if name in {
                "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
                "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"}}
            env.update(ANTHROPIC_BASE_URL=self.base_url, ANTHROPIC_AUTH_TOKEN=key,
                       ANTHROPIC_API_KEY="", ANTHROPIC_MODEL=self.model,
                       CLAUDE_CONFIG_DIR=str(config), DISABLE_AUTOUPDATER="1",
                       CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
            command = [self.executable, "-p", "--restricted", "--tools", "", "--strict-mcp-config",
                "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "", "--no-session-persistence",
                "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--max-turns", "1",
                "--model", self.model, "--system-prompt", instructions +
                "\nReturn exactly the requested role JSON object. No markdown or surrounding commentary."]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=directory, env=env, start_new_session=True)
            failure = None
            try:
                output, errors = process.communicate(payload, timeout=self.timeout)
            except BaseException as exc:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                output, errors = process.communicate()
                failure = exc
            if self.audit_dir is not None:
                self.audit_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                # Preserve rejected generations before parsing/Plan validation; never save auth.
                record = {"role": role, "model": self.model, "returncode": process.returncode,
                          "failure": type(failure).__name__ if failure else None,
                          "stdout": output.decode(errors="replace").replace(key, "[REDACTED]"),
                          "stderr": errors.decode(errors="replace").replace(key, "[REDACTED]")}
                path = self.audit_dir / (uuid.uuid4().hex + ".json")
                with open(path, "x", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
                    json.dump(record, stream)
            if isinstance(failure, subprocess.TimeoutExpired):
                raise TimeoutError("Claude Code role request exceeded deadline") from None
            if failure is not None:
                raise failure
            # Verbose streams include per-token thinking events, not just the role reply.
            if process.returncode or len(output) > 64*1024*1024 or len(errors) > 1024*1024:
                # CLI errors may echo auth headers; never propagate raw stderr or output.
                raise RuntimeError(f"Claude Code request failed (exit {process.returncode}); credentials and model access require checking")
            records = [json.loads(line) for line in output.splitlines() if line.strip()]
            results = [item for item in records if item.get("type") == "result"]
            if len(results) != 1:
                raise RuntimeError("Claude Code stream must contain exactly one terminal result")
            envelope = results[0]
            if len(json.dumps(envelope).encode()) > 8*1024*1024:
                raise ValueError("Claude Code terminal result exceeds byte limit")
            if envelope.get("is_error") is True or envelope.get("subtype") != "success":
                raise RuntimeError("Claude Code did not return a successful role result")
            result = envelope.get("structured_output", envelope.get("result"))
            if isinstance(result, str):
                result = result.strip()
                if result.startswith("```json\n") and result.endswith("\n```"):
                    result = result[8:-4]
                result = json.loads(result)
            if not isinstance(result, dict):
                raise ValueError("Claude Code role result must be a JSON object")
            self.last_metadata = {"session_id": envelope.get("session_id"), "model_usage": envelope.get("modelUsage"),
                                  "duration_ms": envelope.get("duration_ms"), "num_turns": envelope.get("num_turns")}
            return result
