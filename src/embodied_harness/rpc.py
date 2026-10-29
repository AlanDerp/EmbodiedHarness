"""Authenticated, bounded synchronous RPC for worker and external System 2."""

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import urllib.request
from urllib.parse import urlsplit
import uuid

from embodied_harness import wire
from embodied_harness.contracts.types import _number, _text


class RpcClient:
    def __init__(self, url, token, *, timeout_s=60):
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("RPC URL must use HTTP(S), without inline credentials/query/fragment")
        _text(token, "token")
        _number(timeout_s, "timeout_s")
        if timeout_s == 0:
            raise ValueError("timeout_s must be positive")
        self.url, self.token, self.timeout_s = url.rstrip("/") + "/rpc", token, timeout_s
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def call(self, method, **params):
        request_id = uuid.uuid4().hex
        body = wire.dumps({"version": 1, "request_id": request_id, "method": method, "params": params})
        request = urllib.request.Request(self.url, body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        # Never retry an action after an uncertain transport failure.
        with self.opener.open(request, timeout=self.timeout_s) as response:
            value = wire.loads(response.read(wire.MAX_BYTES + 1))
        if type(value.get("version")) is not int or value["version"] != 1 or value.get("request_id") != request_id:
            raise ValueError("RPC response identity mismatch")
        if "error" in value:
            raise RuntimeError(value["error"])
        return value["result"]


class RpcServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, token, methods, *, concurrent_methods=()):
        _text(token, "token")
        self.token, self.methods = token, dict(methods)
        self.concurrent_methods = frozenset(concurrent_methods)
        if not self.concurrent_methods <= self.methods.keys():
            raise ValueError("concurrent methods must be registered")
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(4)
        super().__init__(address, RpcHandler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            try:
                request.settimeout(5)
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class RpcHandler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(120)
        super().setup()

    def log_message(self, *args):
        pass

    def reply(self, status, value):
        raw = wire.dumps(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path != "/rpc":
            self.reply(404, {"error": "unknown route"})
            return
        expected = "Bearer " + self.server.token
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            self.reply(401, {"error": "unauthorized"})
            return
        request_id = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (self.headers.get("Transfer-Encoding") or not 0 < length <= wire.MAX_BYTES
                    or self.headers.get_content_type() != "application/json"):
                raise ValueError("invalid request body")
            value = wire.loads(self.rfile.read(length))
            if not isinstance(value, dict) or set(value) != {"version", "request_id", "method", "params"}:
                raise ValueError("invalid RPC envelope")
            request_id = value["request_id"]
            _text(request_id, "request_id")
            if type(value["version"]) is not int or value["version"] != 1:
                raise ValueError("unknown RPC version")
            if value["method"] not in self.server.methods or not isinstance(value["params"], dict):
                raise ValueError("unknown method or invalid params")
            if value["method"] in self.server.concurrent_methods:
                result = self.server.methods[value["method"]](**value["params"])
            else:
                with self.server.lock:
                    result = self.server.methods[value["method"]](**value["params"])
            self.reply(200, {"version": 1, "request_id": request_id, "result": result})
        except Exception as exc:
            self.reply(400, {"version": 1, "request_id": request_id,
                             "error": f"RPC failed: {type(exc).__name__}"})
