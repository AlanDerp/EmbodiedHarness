"""Read deployment-owned private credentials without copying them into logs/configs."""

import json
import os
from pathlib import Path
import stat


def read_key(path, field, *, required=True):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError("credential file must be a regular owner-only file (chmod 600)")
    if info.st_size > 16_384:
        raise ValueError("credential file too large")
    value = json.loads(path.read_text()).get(field)
    if not isinstance(value, str) or not value.strip() or value.startswith("PASTE_"):
        if required:
            raise ValueError(f"fill {field} in the configured private credential file")
        return None
    return value.strip()
