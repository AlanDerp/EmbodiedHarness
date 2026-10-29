"""SAM3 bridge loaded only by the model service process."""

import hashlib
from pathlib import Path

import numpy as np


class Sam3Backend:
    def __init__(self, checkpoint, *, device="cuda"):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = Path(checkpoint).resolve(strict=True)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_version = f"sam3:checkpoint-sha256:{digest.hexdigest()}"
        self.device, self.torch = device, torch
        self.model = build_sam3_image_model(checkpoint_path=str(checkpoint),
                                          device=device, load_from_HF=False,
                                          enable_inst_interactivity=True).to(device).eval()
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=0.0)

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
                return []
            masks, scores = self._numpy(masks), self._numpy(scores).reshape(-1)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.shape[0] != len(scores):
                raise ValueError("SAM3 mask/score count mismatch")
            order = np.argsort(scores)[::-1]
            return [(masks[i] > 0, float(scores[i])) for i in order if np.any(masks[i] > 0)][:5]
