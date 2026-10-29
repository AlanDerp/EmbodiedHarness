"""SAM3 v1 wire contract shared by client and server."""

import base64
import hashlib
import io

import numpy as np

from embodied_harness.contracts import Segmentation
from embodied_harness.contracts.types import _number, _text

TEXT_PATH = "/v1/segment/text"
POINT_PATH = "/v1/segment/point"
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 1568 * 1568
MAX_CANDIDATES = 5


def _decode(value, max_bytes):
    if not isinstance(value, str) or len(value) > 4 * ((max_bytes + 2) // 3):
        raise ValueError("base64 payload exceeds limit or is not a string")
    raw = base64.b64decode(value, validate=True)
    if len(raw) > max_bytes:
        raise ValueError("decoded payload exceeds limit")
    return raw


def encode_request(observation, camera_id, prompt, request_id):
    from PIL import Image

    camera = observation.cameras[camera_id]
    _text(request_id, "request_id")
    if camera.rgb.shape[0] * camera.rgb.shape[1] > MAX_PIXELS:
        raise ValueError("image exceeds pixel limit")
    stream = io.BytesIO()
    Image.fromarray(camera.rgb).save(stream, format="PNG")
    return {"schema_version": 1, "request_id": request_id,
            "input_rgb_sha256": camera.rgb_sha256,
            "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
            "prompt": prompt}


def decode_request(payload, path):
    from PIL import Image

    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "request_id", "input_rgb_sha256", "image_png_base64", "prompt"
    }:
        raise ValueError("invalid request fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    _text(payload["request_id"], "request_id")
    raw = _decode(payload["image_png_base64"], MAX_BODY_BYTES)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("expected native RGB PNG")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("image exceeds pixel limit")
            rgb = np.array(image)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid PNG image") from exc
    if hashlib.sha256(rgb.tobytes()).hexdigest() != payload["input_rgb_sha256"]:
        raise ValueError("image hash mismatch")
    prompt = payload["prompt"]
    if path == TEXT_PATH:
        _text(prompt, "prompt")
    elif path == POINT_PATH:
        if not isinstance(prompt, list) or len(prompt) != 2:
            raise ValueError("point prompt must be [x, y]")
        for number in prompt:
            _number(number, "pixel coordinate")
        if not (prompt[0] < rgb.shape[1] and prompt[1] < rgb.shape[0]):
            raise ValueError("point is outside image")
    else:
        raise ValueError("unknown route")
    return rgb, prompt


def encode_candidate(mask, score):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise ValueError("candidate must have a nonempty boolean mask")
    _number(score, "score")
    if score > 1:
        raise ValueError("score must be in [0, 1]")
    rows, cols = np.nonzero(mask)
    return {"mask_base64": base64.b64encode(mask.astype(np.uint8).tobytes()).decode("ascii"),
            "shape": list(mask.shape), "box_xyxy": [int(cols.min()), int(rows.min()),
                                                     int(cols.max()) + 1, int(rows.max()) + 1],
            "score": float(score)}


def decode_response(payload, request, observation, camera_id, prompt):
    if not isinstance(payload, dict):
        raise ValueError("response must be an object")
    if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
            or payload.get("request_id") != request["request_id"]
            or payload.get("input_rgb_sha256") != request["input_rgb_sha256"]
            or payload.get("provider") != "sam3"):
        raise ValueError("response version, request identity, image hash or provider mismatch")
    _text(payload.get("model_version"), "model_version")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES:
        raise ValueError("invalid candidate list")
    shape = observation.cameras[camera_id].depth_m.shape
    result = []
    for item in candidates:
        if not isinstance(item, dict) or item.get("shape") != list(shape):
            raise ValueError("candidate shape does not match snapshot")
        raw = _decode(item["mask_base64"], shape[0] * shape[1])
        mask = np.frombuffer(raw, dtype=np.uint8)
        if mask.size != shape[0] * shape[1] or np.any(mask > 1):
            raise ValueError("invalid mask bytes")
        candidate = Segmentation(observation.source, camera_id, mask.reshape(shape).astype(bool),
                                 item["box_xyxy"], item["score"], prompt, "sam3",
                                 payload["model_version"], request["input_rgb_sha256"])
        candidate.validate_against(observation)
        result.append(candidate)
    return tuple(result)
