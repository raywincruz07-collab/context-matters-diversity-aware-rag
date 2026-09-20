"""Small deterministic JSON and immutable-write helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Type


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return sha256_bytes(value.encode("utf-8"))


def stable_json_sha256(value: object) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    conflict_error: Type[Exception] = FileExistsError,
) -> None:
    """Atomically create canonical JSON or reuse an exact byte-identical file."""
    target = Path(path)
    serialized = canonical_json(payload) + "\n"
    if target.exists():
        if not target.is_file() or target.read_text(encoding="utf-8") != serialized:
            raise conflict_error(f"existing immutable artifact differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, target)
    except FileExistsError:
        if not target.is_file() or target.read_text(encoding="utf-8") != serialized:
            raise conflict_error(f"existing immutable artifact differs: {target}")
    finally:
        temporary.unlink(missing_ok=True)
