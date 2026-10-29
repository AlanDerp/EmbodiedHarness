"""Knowledge-only freeze and verification; scene scheduling stays outside agent tools."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def _inventory(root):
    files = {}
    for folder in ("skills", "lessons"):
        base = root / folder
        if base.is_symlink():
            raise ValueError("knowledge directories cannot be symlinks")
        if base.exists():
            for path in base.rglob("*"):
                if path.is_symlink():
                    raise ValueError("knowledge files cannot be symlinks")
                if path.is_file():
                    files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def freeze_knowledge(workspace, destination):
    source, target = Path(workspace), Path(destination)
    if target.exists():
        raise FileExistsError("frozen knowledge already exists")
    files = _inventory(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".freeze-", dir=target.parent))
    try:
        for relative in files:
            output = temporary / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, output)
        if _inventory(temporary) != files:
            raise ValueError("knowledge changed during freeze")
        (temporary / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}, sort_keys=True))
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_knowledge(target)


def verify_knowledge(root):
    root = Path(root)
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("frozen artifact cannot be symlinked")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("files") != _inventory(root):
        raise ValueError("frozen knowledge hash mismatch")
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def stage_knowledge(frozen, workspace):
    digest = verify_knowledge(frozen)
    destination = Path(workspace)
    if any((destination / name).exists() for name in ("skills", "lessons")):
        raise FileExistsError("knowledge staging requires a fresh workspace")
    manifest = json.loads((Path(frozen) / "manifest.json").read_text())
    for relative in manifest["files"]:
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(frozen) / relative, output)
        output.chmod(0o444)
    if _inventory(destination) != manifest["files"]:
        raise ValueError("staged knowledge differs from frozen artifact")
    return digest


def verify_staged_knowledge(frozen, workspace):
    digest = verify_knowledge(frozen)
    manifest = json.loads((Path(frozen) / "manifest.json").read_text())
    if _inventory(Path(workspace)) != manifest["files"]:
        raise ValueError("staged knowledge was modified")
    return digest
