"""Stateless DeepSeek Messages calls with an explicit non-thinking condition."""

import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import uuid

from embodied_harness.credentials import read_key
from .claude_code import public_message


class DeepSeekApiDriver:
    def __init__(self, credentials_file, *, model="deepseek-flash", timeout_s=540,
                 max_tokens=8192, audit_dir=None):
        if model != "deepseek-flash" or not 0 < timeout_s <= 540:
            raise ValueError("explicit DeepSeek model and bounded timeout required")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32000:
            raise ValueError("bounded output token limit required")
        self.credentials_file, self.model = credentials_file, model
        self.timeout, self.max_tokens = timeout_s, max_tokens
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.last_metadata = None

    def configured(self):
        return read_key(self.credentials_file, "DEEPSEEK_API_KEY", required=False) is not None

    def respond(self, role, instructions, context):
        key = read_key(self.credentials_file, "DEEPSEEK_API_KEY")
        message = json.loads(public_message(context))["message"]
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "thinking": {"type": "disabled"}, "stream": False,
                   "system": instructions + "\nReturn only the requested JSON object, without Markdown.",
                   "messages": [message]}
        raw = json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request("https://api.deepseek.com/anthropic/v1/messages", raw,
            headers={"Content-Type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"})
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        output, failure = b"", None
        try:
            with opener.open(request, timeout=self.timeout) as response:
                output = response.read(8*1024*1024 + 1)
        except urllib.error.HTTPError as exc:
            output = exc.read(8*1024*1024 + 1)
            failure = f"HTTP {exc.code}"
        except (OSError, TimeoutError) as exc:
            failure = type(exc).__name__
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            record = {"role": role, "model": self.model, "thinking": "disabled",
                      "max_tokens": self.max_tokens, "request_sha256": hashlib.sha256(raw).hexdigest(),
                      "failure": failure, "response": output.decode(errors="replace").replace(key, "[REDACTED]")}
            with open(self.audit_dir/(uuid.uuid4().hex+".json"), "x",
                      opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
                json.dump(record, stream)
        if failure or len(output) > 8*1024*1024:
            raise RuntimeError(f"DeepSeek request failed: {failure or 'response byte limit'}; no automatic retry")
        envelope = json.loads(output)
        if envelope.get("model") != self.model or envelope.get("stop_reason") != "end_turn":
            raise RuntimeError("DeepSeek model mismatch or incomplete response")
        content = envelope.get("content", [])
        if not content or any(item.get("type") != "text" for item in content):
            raise RuntimeError("non-thinking text-only role response required")
        text = "".join(item["text"] for item in content).strip()
        if text.startswith("```json\n") and text.endswith("\n```"):
            text = text[8:-4]
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("role response must be a JSON object")
        self.last_metadata = {"request_id": envelope.get("id"), "model": envelope["model"],
                              "usage": envelope.get("usage"), "thinking": "disabled"}
        return result
