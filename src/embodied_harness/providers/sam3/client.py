"""Bounded standard-library HTTP client; imports no torch or SAM3."""

import json
import http.client
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

from embodied_harness.contracts import ErrorCode, HarnessError, Result
from embodied_harness.contracts.types import _number, _text
from .wire import (
    MAX_BODY_BYTES, POINT_PATH, TEXT_PATH, decode_response, encode_request,
)


class Sam3Provider:
    def __init__(self, base_url="http://127.0.0.1:8114", *, timeout_s=30):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment
                or parsed.username or parsed.password):
            raise ValueError("base_url must be an HTTP(S) service URL")
        _number(timeout_s, "timeout_s")
        if timeout_s == 0:
            raise ValueError("timeout_s must be positive")
        self.base_url, self.timeout_s = base_url.rstrip("/"), timeout_s
        # Explicit service address; neither environment proxies nor redirects change its destination.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def segment_text(self, observation, camera_id, query):
        _text(query, "query")
        return self._call(observation, camera_id, query, query, TEXT_PATH)

    def segment_point(self, observation, camera_id, point_xy):
        if not isinstance(point_xy, (tuple, list)) or len(point_xy) != 2:
            raise ValueError("point_xy must contain [x, y]")
        camera = observation.cameras[camera_id]
        for value in point_xy:
            _number(value, "pixel coordinate")
        if point_xy[0] >= camera.rgb.shape[1] or point_xy[1] >= camera.rgb.shape[0]:
            raise ValueError("point is outside image")
        prompt = [float(x) for x in point_xy]
        return self._call(observation, camera_id, prompt, f"point:{prompt[0]},{prompt[1]}", POINT_PATH)

    def _call(self, observation, camera_id, wire_prompt, local_prompt, path):
        payload = encode_request(observation, camera_id, wire_prompt, uuid.uuid4().hex)
        body = json.dumps(payload, allow_nan=False).encode()
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("request exceeds byte limit")
        request = urllib.request.Request(self.base_url + path, data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read(MAX_BODY_BYTES + 1)
            if len(raw) > MAX_BODY_BYTES:
                raise ValueError("response exceeds byte limit")
            candidates = decode_response(json.loads(raw), payload, observation, camera_id, local_prompt)
            return Result(candidates)
        except (TimeoutError, socket.timeout):
            return Result(error=HarnessError(ErrorCode.TIMEOUT, "SAM3 request timed out"))
        except urllib.error.HTTPError as exc:
            code = ErrorCode.UNAVAILABLE if exc.code in (429, 503) else ErrorCode.INVALID_RESPONSE
            return Result(error=HarnessError(code, f"SAM3 HTTP status {exc.code}"))
        except urllib.error.URLError as exc:
            code = ErrorCode.TIMEOUT if isinstance(exc.reason, (TimeoutError, socket.timeout)) else ErrorCode.UNAVAILABLE
            return Result(error=HarnessError(code, "SAM3 service unavailable"))
        except (ValueError, TypeError, KeyError) as exc:
            return Result(error=HarnessError(ErrorCode.INVALID_RESPONSE, str(exc)))
        except (OSError, http.client.HTTPException):
            return Result(error=HarnessError(ErrorCode.UNAVAILABLE, "SAM3 connection interrupted"))
