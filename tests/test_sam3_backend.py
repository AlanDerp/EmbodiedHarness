from contextlib import nullcontext
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from embodied_harness.services.sam3.backend import Sam3Backend


class BackendTests(unittest.TestCase):
    def backend(self, output):
        backend = Sam3Backend.__new__(Sam3Backend)
        backend.torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext(), bfloat16=None)
        backend.device = "cpu"
        backend.processor = Mock()
        backend.processor.set_image.return_value = {"fixture_state": True}
        backend.processor.set_text_prompt.return_value = output
        backend.model = Mock()
        return backend

    def test_text_preserves_resolution_orders_scores_and_discards_empty_masks(self):
        masks = np.ones((3, 1, 3, 4), bool)
        masks[1] = False
        backend = self.backend({"masks": masks, "scores": np.array([.2, .9, .8])})
        result = backend.segment(np.zeros((3, 4, 3), np.uint8), "bowl")
        self.assertEqual([score for _, score in result], [.8, .2])
        self.assertEqual(result[0][0].shape, (3, 4))
        self.assertEqual(backend.processor.set_image.call_args.args[0].size, (4, 3))

    def test_empty_prediction_is_valid_but_missing_or_corrupt_data_is_not(self):
        good = {"masks": np.empty((0, 1, 3, 4), bool), "scores": np.empty(0)}
        self.assertEqual(self.backend(good).segment(np.zeros((3, 4, 3), np.uint8), "bowl"), [])
        for output in ({}, {"masks": np.ones((1, 2, 4), bool), "scores": [.5]},
                       {"masks": np.ones((1, 3, 4), bool), "scores": [np.nan]},
                       {"masks": np.full((1, 3, 4), .5), "scores": [.5]}):
            with self.assertRaises(ValueError):
                self.backend(output).segment(np.zeros((3, 4, 3), np.uint8), "bowl")

    def test_point_branch_uses_one_positive_prompt(self):
        backend = self.backend({})
        backend.model.predict_inst.return_value = (np.ones((1, 3, 4), bool), np.array([.8]), None)
        result = backend.segment(np.zeros((3, 4, 3), np.uint8), [1, 2], point=True)
        self.assertEqual(len(result), 1)
        kwargs = backend.model.predict_inst.call_args.kwargs
        np.testing.assert_array_equal(kwargs["point_coords"], [[1, 2]])
        np.testing.assert_array_equal(kwargs["point_labels"], [1])
        backend.processor.set_text_prompt.assert_not_called()

    def test_wrong_checkpoint_rejected_before_torch_import(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            path.write_bytes(b"fixture-weights")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                Sam3Backend(path, checkpoint_sha256="0" * 64)

    def test_loader_records_actual_code_and_checkpoint_and_disables_remote_download(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            path.write_bytes(b"fixture-weights")
            code = Path(folder) / "fixture.py"
            code.write_text("# fixture only, not SAM3\n")
            modules = {name: ModuleType(name) for name in (
                "sam3", "sam3.model", "sam3.model_builder", "sam3.model.sam3_image_processor", "sam3.model.sam3_image")}
            for module in modules.values():
                module.__file__ = str(code)
            model = Mock()
            model.to.return_value = model
            model.eval.return_value = model
            build = Mock(return_value=model)
            modules["sam3.model_builder"].build_sam3_image_model = build
            modules["sam3.model.sam3_image_processor"].Sam3Processor = Mock()
            modules["torch"] = ModuleType("torch")
            modules["torch"].load = Mock(return_value={"detector.fixture": 1, "tracker.fixture": 2})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict("sys.modules", modules):
                backend = Sam3Backend(path, device="cpu", checkpoint_sha256=digest)
            self.assertIn(digest, backend.model_version)
            self.assertEqual(len(backend.code_sha256), 64)
            self.assertFalse(build.call_args.kwargs["load_from_HF"])
            self.assertEqual(build.call_args.kwargs["device"], "cpu")
            self.assertIsNone(build.call_args.kwargs["checkpoint_path"])
            model.load_state_dict.assert_called_once_with({"fixture": 1, "inst_interactive_predictor.model.fixture": 2}, strict=True)

    def test_checkpoint_rejects_missing_weights_and_uses_strict_loading(self):
        backend = self.backend({})
        backend.torch.load = Mock(return_value={"model": {"unrelated": 0}})
        with self.assertRaises(ValueError):
            backend._load_checkpoint("fixture.pt")
        backend.torch.load.return_value = {"detector.weight": 1}
        backend.model.load_state_dict.side_effect = RuntimeError("missing interactive weights")
        with self.assertRaises(RuntimeError):
            backend._load_checkpoint("fixture.pt")
        self.assertTrue(backend.torch.load.call_args.kwargs["weights_only"])


if __name__ == "__main__":
    unittest.main()
