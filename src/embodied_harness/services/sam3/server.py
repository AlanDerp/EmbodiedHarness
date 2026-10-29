"""Small bounded HTTP service; backend injection permits CPU protocol tests."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import numpy as np

from embodied_harness.providers.sam3.wire import (
    MAX_BODY_BYTES, MAX_CANDIDATES, POINT_PATH, TEXT_PATH, decode_request, encode_candidate,
)


class Sam3Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend=None, *, request_timeout_s=30):
        self.backend = backend
        self.request_timeout_s = request_timeout_s
        self._slots = threading.BoundedSemaphore(2)
        self.inference_lock = threading.Lock()
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(self.request_timeout_s)
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(self.server.request_timeout_s)
        super().setup()

    def log_message(self, format, *args):
        pass  # Image payloads and prompts are not HTTP log content.

    def _reply(self, status, value):
        body = json.dumps(value, allow_nan=False).encode()
        if len(body) > MAX_BODY_BYTES:
            status, body = 500, b'{"error":"response exceeds byte limit"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"alive": True})
        elif self.path == "/ready":
            backend = self.server.backend
            self._reply(200 if backend else 503,
                        {"ready": backend is not None,
                         "model_version": backend.model_version if backend else None})
        else:
            self._reply(404, {"error": "unknown route"})

    def do_POST(self):
        if self.path not in (TEXT_PATH, POINT_PATH):
            self._reply(404, {"error": "unknown route"})
            return
        if self.server.backend is None:
            self._reply(503, {"error": "model not ready"})
            return
        try:
            if self.headers.get("Transfer-Encoding") or self.headers.get_content_type() != "application/json":
                raise ValueError("expected non-chunked application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                self._reply(413, {"error": "invalid body size"})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete request body")
            payload = json.loads(raw)
            rgb, prompt = decode_request(payload, self.path)
        except (ValueError, TypeError, KeyError, OSError):
            self._reply(400, {"error": "invalid image or request"})
            return
        if not self.server.inference_lock.acquire(timeout=self.server.request_timeout_s):
            self._reply(503, {"error": "inference busy"})
            return
        try:
            backend = self.server.backend
            candidates = backend.segment(rgb, prompt, point=self.path == POINT_PATH)
            encoded = []
            for mask, score in candidates:
                if np.asarray(mask).shape != rgb.shape[:2]:
                    raise ValueError("backend returned a mask of the wrong size")
                encoded.append(encode_candidate(mask, score))
                if len(encoded) == MAX_CANDIDATES:
                    break
            value = {"schema_version": 1, "request_id": payload["request_id"],
                     "input_rgb_sha256": payload["input_rgb_sha256"], "provider": "sam3",
                     "model_version": backend.model_version, "candidates": encoded}
        except Exception:
            self._reply(500, {"error": "model inference failed"})
            return
        finally:
            self.server.inference_lock.release()
        self._reply(200, value)


def main():
    parser = argparse.ArgumentParser(description="Run the isolated SAM3 service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    from .backend import Sam3Backend
    backend = Sam3Backend(args.checkpoint, device=args.device)
    with Sam3Server((args.host, args.port), backend) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
