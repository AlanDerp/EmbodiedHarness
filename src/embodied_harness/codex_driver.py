"""Fresh Codex CLI conversations implementing the existing role driver contract."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


class CodexDriver:
    def __init__(self, *, model, executable="codex", timeout_s=540):
        if not model or "REPLACE_WITH" in model:
            raise ValueError("explicit Codex model required")
        self.model, self.executable, self.timeout_s = model, executable, timeout_s

    def respond(self, role, instructions, context):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="embodied-codex-") as folder:
            root = Path(folder)
            images = []

            def public(value):
                if isinstance(value, np.ndarray):
                    if value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3:
                        path = root / f"rgb_{len(images)}.png"
                        Image.fromarray(value).save(path)
                        images.append(path)
                        return {"image_attachment": path.name, "shape": list(value.shape)}
                    if value.size <= 64:
                        return value.tolist()
                    # Large depth/mask arrays remain lossless public inputs, not token dumps.
                    path = root / f"array_{len(list(root.glob('array_*.npy')))}.npy"
                    np.save(path, value, allow_pickle=False)
                    return {"public_array_file": str(path), "shape": list(value.shape), "dtype": str(value.dtype)}
                if is_dataclass(value):
                    return {field.name: public(getattr(value, field.name)) for field in fields(value)}
                if isinstance(value, Mapping):
                    return {key: public(item) for key, item in value.items()}
                if isinstance(value, (tuple, list)):
                    return [public(item) for item in value]
                if isinstance(value, Enum):
                    return value.value
                return value

            prompt = instructions + "\nReturn only a JSON object, without Markdown fences.\n"
            prompt += ("RGB images are attached in the order referenced below. Only the supplied public_array_file "
                       "files may be read as additional context; do not inspect any other files or use network/tools "
                       "to obtain solutions. Generate code for the public runtime API; do not execute robot actions.\n")
            prompt += json.dumps({"role": role, "context": public(context)}, allow_nan=False)
            output = root / "reply.json"
            command = [self.executable, "exec", "--ephemeral", "--ignore-user-config",
                       "--skip-git-repo-check", "--sandbox", "read-only", "--model", self.model,
                       "--output-last-message", str(output)]
            for path in images:
                command.extend(["--image", str(path)])
            command.append("-")
            # Service credentials must never become model subprocess environment variables.
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("EMBODIED_") and key != "TYPESAFE_API_KEY"}
            subprocess.run(command, input=prompt, text=True, cwd=root, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=self.timeout_s, check=True)
            reply = json.loads(output.read_text())
            if not isinstance(reply, dict):
                raise ValueError("Codex role response must be a JSON object")
            return reply


def create():
    return CodexDriver(model=os.environ.get("EMBODIED_CODEX_MODEL"),
                       executable=os.environ.get("EMBODIED_CODEX_BIN", "codex"))
