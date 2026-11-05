"""SAM3 bridge loaded only by the model service process."""

import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np


class Sam3Backend:
    def __init__(self, checkpoint, *, device="cuda", checkpoint_sha256=None):
        checkpoint = Path(checkpoint).resolve(strict=True)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        if checkpoint_sha256 is not None and self.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("checkpoint SHA256 differs from deployment configuration")
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        code = hashlib.sha256()
        package = Path(inspect.getfile(importlib.import_module("sam3"))).parent
        for source in sorted(path for path in package.rglob("*") if path.suffix in (".py", ".gz") and path.is_file()):
            code.update(source.relative_to(package).as_posix().encode())
            code.update(source.read_bytes())
        self.code_sha256 = code.hexdigest()
        self.model_version = f"sam3:code:{self.code_sha256}:checkpoint:{self.checkpoint_sha256}"
        self.device, self.torch = device, torch
        self.model = build_sam3_image_model(checkpoint_path=None,
                                          device=device, load_from_HF=False,
                                          enable_inst_interactivity=True).to(device).eval()
        self._load_checkpoint(checkpoint)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=0.0)

    def _load_checkpoint(self, checkpoint):
        # Upstream's convenience loader tolerates missing weights; an experiment must fail closed.
        state = self.torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError("SAM3 checkpoint must contain a state dictionary")
        weights = {key.replace("detector.", "", 1): value for key, value in state.items() if key.startswith("detector.")}
        weights.update({key.replace("tracker.", "inst_interactive_predictor.model.", 1): value
                        for key, value in state.items() if key.startswith("tracker.")})
        if not weights:
            raise ValueError("SAM3 checkpoint contains no detector/tracker weights")
        self.model.load_state_dict(weights, strict=True)

    @staticmethod
    def _numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    def segment(self, rgb, prompt, *, point=False):
        from PIL import Image

        torch = self.torch
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))
            if point:
                masks, scores, _ = self.model.predict_inst(
                    state, point_coords=np.asarray([prompt], dtype=np.float32),
                    point_labels=np.ones(1, dtype=np.int64), multimask_output=True)
            else:
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
                masks, scores = output.get("masks"), output.get("scores")
            if masks is None or scores is None:
                raise ValueError("SAM3 response is missing masks or scores")
            masks, scores = self._numpy(masks), self._numpy(scores).reshape(-1)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim != 3 or masks.shape != (len(scores), *rgb.shape[:2]):
                raise ValueError("SAM3 mask/score count or original resolution mismatch")
            if (not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)
                    or not np.isfinite(masks).all() or np.any((masks != 0) & (masks != 1))):
                raise ValueError("SAM3 must return finite probabilities and binary masks")
            order = np.argsort(scores)[::-1]
            return [(masks[i] > 0, float(scores[i])) for i in order if np.any(masks[i] > 0)][:5]
