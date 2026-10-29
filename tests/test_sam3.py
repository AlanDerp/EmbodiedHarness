from dataclasses import replace
import io
import json
import socket
import threading
import subprocess
import sys
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

from embodied_harness.artifacts import SnapshotStore
from embodied_harness.contracts import ErrorCode
from embodied_harness.perception import Perception
from embodied_harness.providers.sam3 import Sam3Provider
from embodied_harness.providers.sam3.wire import (
    POINT_PATH, TEXT_PATH, decode_request, decode_response, encode_candidate, encode_request,
)
from embodied_harness.services.sam3.server import Sam3Server
from tempfile import TemporaryDirectory
from test_contracts import scene


class FixtureBackend:
    model_version = "fixture-not-real-sam3"

    def segment(self, rgb, prompt, *, point=False):
        return [(np.ones(rgb.shape[:2], bool), 0.8)]


class WireTests(unittest.TestCase):
    def test_png_roundtrip_and_point_validation(self):
        obs = scene()
        request = encode_request(obs, "head", "bowl", "request1")
        rgb, prompt = decode_request(request, TEXT_PATH)
        np.testing.assert_array_equal(rgb, obs.cameras["head"].rgb)
        self.assertEqual(prompt, "bowl")
        for point in ([0, 0], [3, 2]):
            decode_request(dict(request, prompt=point), POINT_PATH)
        for point in ([-1, 0], [4, 0], [0, 3], [True, 0], [np.nan, 0]):
            with self.assertRaises(ValueError):
                decode_request(dict(request, prompt=point), POINT_PATH)

    def test_invalid_image_hash_schema_and_fields(self):
        request = encode_request(scene(), "head", "bowl", "request1")
        for changes in ({"input_rgb_sha256": "bad"}, {"schema_version": True},
                        {"image_png_base64": "invalid"}, {"unexpected": 1}):
            with self.assertRaises((ValueError, OSError)):
                decode_request(dict(request, **changes), TEXT_PATH)

    def test_response_identity_shape_and_mask_bytes(self):
        obs = scene()
        request = encode_request(obs, "head", "bowl", "request1")
        item = encode_candidate(np.ones((3, 4), bool), 0.9)
        response = {"schema_version": 1, "request_id": "request1", "provider": "sam3",
                    "model_version": "fixture", "input_rgb_sha256": request["input_rgb_sha256"],
                    "candidates": [item]}
        candidates = decode_response(response, request, obs, "head", "bowl")
        candidates[0].validate_against(obs)
        self.assertEqual(candidates[0].source, obs.source)
        for changes in ({"request_id": "other"}, {"input_rgb_sha256": "bad"},
                        {"schema_version": 2}, {"candidates": [dict(item, shape=[4, 3])]},
                        {"candidates": [dict(item, mask_base64="Ag==")]},
                        {"candidates": [item] * 6}):
            with self.assertRaises(ValueError):
                decode_response(dict(response, **changes), request, obs, "head", "bowl")


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Sam3Server(("127.0.0.1", 0), FixtureBackend(), request_timeout_s=1)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.server.backend = FixtureBackend()

    def test_real_http_client_text_point_and_localization(self):
        provider = Sam3Provider(self.url)
        obs = scene()
        result = provider.segment_text(obs, "head", "bowl")
        self.assertTrue(result.ok)
        self.assertEqual(result.data[0].model_version, "fixture-not-real-sam3")
        result.data[0].validate_against(obs)
        self.assertTrue(provider.segment_point(obs, "head", (1, 1)).ok)
        with TemporaryDirectory() as folder:
            store = SnapshotStore(folder)
            store.put(obs)
            localized = Perception(store, provider).localize_object("episode", "snapshot", "head", "bowl")
            self.assertTrue(localized.ok)
            self.assertEqual(localized.data.geometry.point_count, 12)

    def test_health_and_not_ready(self):
        self.server.backend = None
        with urllib.request.urlopen(self.url + "/health") as response:
            self.assertTrue(json.load(response)["alive"])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.url + "/ready")
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual(Sam3Provider(self.url).segment_text(scene(), "head", "bowl").error.code,
                         ErrorCode.UNAVAILABLE)

    def test_server_rejects_bad_request(self):
        request = urllib.request.Request(self.url + TEXT_PATH, data=b"{}",
                                         headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 400)

    def test_inference_failure_does_not_become_empty_success(self):
        with patch.object(self.server.backend, "segment", side_effect=RuntimeError("GPU failure")):
            result = Sam3Provider(self.url).segment_text(scene(), "head", "bowl")
        self.assertEqual(result.error.code, ErrorCode.INVALID_RESPONSE)

    def test_empty_prediction_and_bad_mask_size(self):
        provider = Sam3Provider(self.url)
        with patch.object(self.server.backend, "segment", return_value=[]):
            result = provider.segment_text(scene(), "head", "bowl")
            self.assertTrue(result.ok)
            self.assertEqual(result.data, ())
        with patch.object(self.server.backend, "segment", return_value=[(np.ones((2, 2), bool), 1)]):
            self.assertEqual(provider.segment_text(scene(), "head", "bowl").error.code,
                             ErrorCode.INVALID_RESPONSE)

    def test_admission_limit_rejects_excess_requests(self):
        self.server._slots.acquire()
        self.server._slots.acquire()
        try:
            result = Sam3Provider(self.url).segment_text(scene(), "head", "bowl")
            self.assertEqual(result.error.code, ErrorCode.UNAVAILABLE)
        finally:
            self.server._slots.release()
            self.server._slots.release()


class ClientErrorTests(unittest.TestCase):
    def test_client_import_does_not_import_model_or_image_runtime(self):
        subprocess.run([sys.executable, "-c", "import sys; import embodied_harness.providers.sam3; "
                        "assert not ({'torch', 'sam3', 'PIL', 'isaacsim'} & sys.modules.keys())"], check=True)

    def test_timeout_and_unreachable_service(self):
        provider = Sam3Provider()
        for error, code in ((TimeoutError(), ErrorCode.TIMEOUT),
                            (urllib.error.URLError(socket.timeout()), ErrorCode.TIMEOUT),
                            (urllib.error.URLError("refused"), ErrorCode.UNAVAILABLE)):
            with patch.object(provider._opener, "open", side_effect=error):
                self.assertEqual(provider.segment_text(scene(), "head", "bowl").error.code, code)

    def test_malformed_response_and_wrong_request_id(self):
        provider = Sam3Provider()
        for raw in (b"not json", b'{}'):
            with patch.object(provider._opener, "open", return_value=io.BytesIO(raw)):
                self.assertEqual(provider.segment_text(scene(), "head", "bowl").error.code,
                                 ErrorCode.INVALID_RESPONSE)

    def test_invalid_local_arguments(self):
        provider = Sam3Provider()
        with self.assertRaises(ValueError):
            Sam3Provider(timeout_s=0)
        with self.assertRaises(ValueError):
            Sam3Provider("file:///tmp/model")
        with self.assertRaises(ValueError):
            provider.segment_point(scene(), "head", (4, 0))
        with self.assertRaises(ValueError):
            provider.segment_text(scene(), "head", "")


if __name__ == "__main__":
    unittest.main()
