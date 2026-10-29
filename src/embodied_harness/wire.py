"""Explicit allowlisted RPC values; never pickle, import or execute wire-supplied types."""

import base64
from dataclasses import fields, is_dataclass
from enum import Enum
import io
import json
import math
from collections.abc import Mapping

import numpy as np

MAX_BYTES = 64 * 1024 * 1024


def _types():
    from embodied_harness.contracts import types
    from embodied_harness.geometry import ObjectGeometry, Projection
    from embodied_harness.perception import Localization
    names = ("Pose", "ObservationSource", "CameraObservation", "ArmState", "RobotState",
             "Observation", "Segmentation", "PointCloud", "ArmCommand", "RobotAction",
             "HarnessError", "Result", "StepReceipt")
    return {**{name: getattr(types, name) for name in names},
            "ObjectGeometry": ObjectGeometry, "Projection": Projection, "Localization": Localization}


def encode(value):
    if isinstance(value, Enum):
        return {"$enum": "ErrorCode", "value": value.value}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or value.nbytes > MAX_BYTES:
            raise ValueError("unsupported or oversized wire array")
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {"$array": base64.b64encode(stream.getvalue()).decode("ascii")}
    if is_dataclass(value):
        if type(value).__name__ not in _types() or type(value) is not _types()[type(value).__name__]:
            raise TypeError("unregistered wire type")
        return {"$type": type(value).__name__, "fields": {
            field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or key.startswith("$") for key in value):
            raise ValueError("wire mapping requires non-reserved string keys")
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unsupported wire value")


def decode(value):
    if isinstance(value, list):
        return tuple(decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "$array" in value:
        if set(value) != {"$array"}:
            raise ValueError("invalid array envelope")
        raw = base64.b64decode(value["$array"], validate=True)
        if len(raw) > MAX_BYTES:
            raise ValueError("wire array exceeds limit")
        stream = io.BytesIO(raw)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported NPY version")
        expected_bytes = math.prod(shape) * dtype.itemsize
        if dtype.kind not in "biuf" or expected_bytes > MAX_BYTES or len(raw) - stream.tell() != expected_bytes:
            raise ValueError("invalid array size or dtype")
        array = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf":
            raise ValueError("invalid wire array")
        return array
    if "$enum" in value:
        from embodied_harness.contracts import ErrorCode
        if set(value) != {"$enum", "value"} or value["$enum"] != "ErrorCode":
            raise ValueError("unknown enum")
        return ErrorCode(value["value"])
    if "$type" in value:
        if set(value) != {"$type", "fields"} or value["$type"] not in _types():
            raise ValueError("unknown contract type")
        cls = _types()[value["$type"]]
        if not isinstance(value["fields"], dict):
            raise ValueError("invalid fields")
        expected = {f.name for f in fields(cls)}
        if set(value["fields"]) != expected:
            raise ValueError("contract fields differ from schema")
        return cls(**{key: decode(item) for key, item in value["fields"].items()})
    if any(key.startswith("$") for key in value):
        raise ValueError("unknown reserved wire key")
    return {key: decode(item) for key, item in value.items()}


def dumps(value):
    raw = json.dumps(encode(value), allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("wire message exceeds limit")
    return raw


def loads(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("wire message exceeds limit")
    def invalid_constant(value):
        raise ValueError("nonfinite JSON number")
    return decode(json.loads(raw, parse_constant=invalid_constant))
