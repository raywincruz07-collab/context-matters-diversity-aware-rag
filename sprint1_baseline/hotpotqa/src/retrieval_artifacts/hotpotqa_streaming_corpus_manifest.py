"""Streaming corpus manifest for canonical Sprint-3 HotpotQA retrieval.

The canonical corpus is the complete BEIR HotpotQA corpus at the frozen
Hugging Face revision.  Scientific identity binds source document order,
exact source title/text identities, and the exact prepared retrieval content:

    retrieval_content = (title + " " + text).strip()

The manifest uses compact hash-only JSONL shards so millions of source
documents do not need to be materialized as JSON objects in the root manifest.
Physical shard layout is storage metadata and does not affect scientific
identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, BinaryIO


HOTPOTQA_SOURCE = "BeIR/hotpotqa"
HOTPOTQA_REVISION = "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b"
HOTPOTQA_CONFIG = "corpus"
HOTPOTQA_SPLIT = "corpus"
HOTPOTQA_EXPECTED_DOCUMENT_COUNT = 5_233_329

HOTPOTQA_RETRIEVAL_SERIALIZATION = "hotpot_retrieval_content_v1"

HOTPOTQA_CORPUS_MANIFEST_SCHEMA_VERSION = (
    "sprint3.hotpotqa-streaming-corpus-manifest.v1"
)
HOTPOTQA_CORPUS_MANIFEST_ARTIFACT_FORMAT = (
    "sprint3.hotpotqa-streaming-corpus-manifest-artifact.v1"
)
HOTPOTQA_MANIFEST_SHARD_FORMAT = (
    "sprint3.hotpotqa-document-hash-jsonl-shards.v1"
)
HOTPOTQA_DEFAULT_SHARD_SIZE = 1_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_HASH_CHUNK_SIZE = 1024 * 1024

_LOGICAL_ROW_KEYS = {
    "position",
    "retrieval_content_sha256",
    "source_document_id",
    "text_sha256",
    "title_sha256",
}

_SCIENTIFIC_PAYLOAD_KEYS = {
    "config",
    "document_count",
    "document_id_map_sha256",
    "first_source_document_id",
    "last_source_document_id",
    "logical_entry_stream_sha256",
    "retrieval_serialization",
    "revision",
    "schema_version",
    "source",
    "split",
}

_ROOT_KEYS = {
    "artifact_format",
    "scientific_payload",
    "scientific_sha256",
    "storage",
}

_STORAGE_KEYS = {
    "format",
    "manifest_shards",
    "shard_size",
}

_SHARD_KEYS = {
    "first_position",
    "last_position",
    "path",
    "physical_sha256",
    "row_count",
    "size_bytes",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_optional_document_count(value: object) -> int | None:
    if value is None:
        return None
    return _require_positive_integer(value, "expected_document_count")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} fields mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_FILE_HASH_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_retrieval_content(title: str, text: str) -> str:
    """Construct exact frozen HotpotQA retrieval text."""
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return (title + " " + text).strip()


def _finalize_shard(
    handle: BinaryIO,
    path: Path,
    digest: Any,
    *,
    row_count: int,
    first_position: int,
    last_position: int,
    size_bytes: int,
) -> dict[str, Any]:
    try:
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()

    return {
        "filename": path.name,
        "first_position": first_position,
        "last_position": last_position,
        "physical_sha256": digest.hexdigest(),
        "row_count": row_count,
        "size_bytes": size_bytes,
    }


def _build_staged_shards(
    rows: Iterable[Mapping[str, Any]],
    staging_directory: Path,
    *,
    shard_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logical_digest = hashlib.sha256()
    document_id_map_digest = hashlib.sha256()

    seen_source_ids: set[str] = set()

    shard_metadata: list[dict[str, Any]] = []
    shard_handle: BinaryIO | None = None
    shard_path: Path | None = None
    shard_digest: Any = None
    shard_row_count = 0
    shard_first_position = 0
    shard_size_bytes = 0

    document_count = 0
    first_source_document_id: str | None = None
    last_source_document_id: str | None = None

    try:
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"source row at position {position} must be a mapping"
                )

            for field in ("_id", "title", "text"):
                if field not in row:
                    raise ValueError(
                        f"source row at position {position} is missing {field}"
                    )
                if not isinstance(row[field], str):
                    raise TypeError(
                        f"{field} at position {position} must be a string"
                    )

            source_document_id = row["_id"]
            title = row["title"]
            text = row["text"]

            if source_document_id in seen_source_ids:
                raise ValueError(
                    "duplicate source document ID at position "
                    f"{position}: {source_document_id!r}"
                )
            seen_source_ids.add(source_document_id)

            retrieval_content = canonical_retrieval_content(title, text)

            logical_row_object = {
                "position": position,
                "source_document_id": source_document_id,
                "text_sha256": _sha256_text(text),
                "title_sha256": _sha256_text(title),
                "retrieval_content_sha256": _sha256_text(
                    retrieval_content
                ),
            }
            logical_row = _canonical_json_bytes(logical_row_object) + b"\n"

            id_map_row = (
                _canonical_json_bytes(
                    {
                        "position": position,
                        "source_document_id": source_document_id,
                    }
                )
                + b"\n"
            )

            if shard_handle is None:
                shard_index = len(shard_metadata)
                shard_path = staging_directory / (
                    f"part-{shard_index:06d}.jsonl"
                )
                shard_handle = shard_path.open("xb")
                shard_digest = hashlib.sha256()
                shard_row_count = 0
                shard_first_position = position
                shard_size_bytes = 0

            shard_handle.write(logical_row)
            shard_digest.update(logical_row)

            logical_digest.update(logical_row)
            document_id_map_digest.update(id_map_row)

            shard_row_count += 1
            shard_size_bytes += len(logical_row)

            if first_source_document_id is None:
                first_source_document_id = source_document_id
            last_source_document_id = source_document_id
            document_count = position + 1

            if shard_row_count == shard_size:
                assert shard_path is not None
                shard_metadata.append(
                    _finalize_shard(
                        shard_handle,
                        shard_path,
                        shard_digest,
                        row_count=shard_row_count,
                        first_position=shard_first_position,
                        last_position=position,
                        size_bytes=shard_size_bytes,
                    )
                )
                shard_handle = None
                shard_path = None
                shard_digest = None

    except BaseException:
        if shard_handle is not None:
            shard_handle.close()
        raise

    if shard_handle is not None:
        assert shard_path is not None
        shard_metadata.append(
            _finalize_shard(
                shard_handle,
                shard_path,
                shard_digest,
                row_count=shard_row_count,
                first_position=shard_first_position,
                last_position=document_count - 1,
                size_bytes=shard_size_bytes,
            )
        )

    if document_count == 0:
        raise ValueError("HotpotQA corpus must contain at least one document")

    assert first_source_document_id is not None
    assert last_source_document_id is not None

    summary = {
        "document_count": document_count,
        "document_id_map_sha256": document_id_map_digest.hexdigest(),
        "first_source_document_id": first_source_document_id,
        "last_source_document_id": last_source_document_id,
        "logical_entry_stream_sha256": logical_digest.hexdigest(),
    }
    return summary, shard_metadata


def _storage_directory_name(
    root_manifest_path: Path,
    *,
    shard_size: int,
    shard_metadata: list[dict[str, Any]],
) -> str:
    layout = {
        "format": HOTPOTQA_MANIFEST_SHARD_FORMAT,
        "shard_size": shard_size,
        "shards": shard_metadata,
    }
    layout_sha256 = hashlib.sha256(
        _canonical_json_bytes(layout)
    ).hexdigest()
    return f"{root_manifest_path.name}.shards-{layout_sha256}"


def _staged_shards_match_existing(
    final_directory: Path,
    shard_metadata: list[dict[str, Any]],
) -> bool:
    if not final_directory.is_dir() or final_directory.is_symlink():
        return False

    expected_names = {
        metadata["filename"] for metadata in shard_metadata
    }
    actual_names = {path.name for path in final_directory.iterdir()}

    if actual_names != expected_names:
        return False

    for metadata in shard_metadata:
        path = final_directory / metadata["filename"]
        if not path.is_file() or path.is_symlink():
            return False
        if path.stat().st_size != metadata["size_bytes"]:
            return False
        if _file_sha256(path) != metadata["physical_sha256"]:
            return False

    return True


def _publish_staged_shards(
    staging_directory: Path,
    final_directory: Path,
    shard_metadata: list[dict[str, Any]],
) -> None:
    if final_directory.exists():
        if _staged_shards_match_existing(
            final_directory,
            shard_metadata,
        ):
            return
        raise FileExistsError(
            f"existing HotpotQA shard directory differs: {final_directory}"
        )

    try:
        os.replace(staging_directory, final_directory)
    except OSError:
        if (
            final_directory.exists()
            and _staged_shards_match_existing(
                final_directory,
                shard_metadata,
            )
        ):
            return
        raise


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    serialized = _canonical_json_bytes(value) + b"\n"

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_hotpotqa_streaming_corpus_manifest(
    rows: Iterable[Mapping[str, Any]],
    root_manifest_path: str | os.PathLike[str],
    *,
    shard_size: int = HOTPOTQA_DEFAULT_SHARD_SIZE,
    expected_document_count: int | None = HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
) -> dict[str, Any]:
    """Build deterministic HotpotQA corpus-manifest hash shards."""
    shard_size = _require_positive_integer(shard_size, "shard_size")
    expected_document_count = _require_optional_document_count(
        expected_document_count
    )

    root_path = Path(root_manifest_path)
    root_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=root_path.parent,
        prefix=f".{root_path.name}.build-",
    ) as temporary_root:
        staging_directory = Path(temporary_root) / "shards"
        staging_directory.mkdir()

        stream_summary, shard_metadata = _build_staged_shards(
            rows,
            staging_directory,
            shard_size=shard_size,
        )

        document_count = stream_summary["document_count"]

        if (
            expected_document_count is not None
            and document_count != expected_document_count
        ):
            raise ValueError(
                "HotpotQA document count mismatch: "
                f"expected {expected_document_count}, "
                f"found {document_count}"
            )

        scientific_payload = {
            "config": HOTPOTQA_CONFIG,
            "document_count": document_count,
            "document_id_map_sha256": stream_summary[
                "document_id_map_sha256"
            ],
            "first_source_document_id": stream_summary[
                "first_source_document_id"
            ],
            "last_source_document_id": stream_summary[
                "last_source_document_id"
            ],
            "logical_entry_stream_sha256": stream_summary[
                "logical_entry_stream_sha256"
            ],
            "retrieval_serialization": HOTPOTQA_RETRIEVAL_SERIALIZATION,
            "revision": HOTPOTQA_REVISION,
            "schema_version": HOTPOTQA_CORPUS_MANIFEST_SCHEMA_VERSION,
            "source": HOTPOTQA_SOURCE,
            "split": HOTPOTQA_SPLIT,
        }

        scientific_sha256 = hashlib.sha256(
            _canonical_json_bytes(scientific_payload)
        ).hexdigest()

        storage_directory_name = _storage_directory_name(
            root_path,
            shard_size=shard_size,
            shard_metadata=shard_metadata,
        )

        final_storage_directory = (
            root_path.parent / storage_directory_name
        )

        storage_shards = [
            {
                "first_position": metadata["first_position"],
                "last_position": metadata["last_position"],
                "path": PurePosixPath(
                    storage_directory_name,
                    metadata["filename"],
                ).as_posix(),
                "physical_sha256": metadata["physical_sha256"],
                "row_count": metadata["row_count"],
                "size_bytes": metadata["size_bytes"],
            }
            for metadata in shard_metadata
        ]

        artifact = {
            "artifact_format": (
                HOTPOTQA_CORPUS_MANIFEST_ARTIFACT_FORMAT
            ),
            "scientific_payload": scientific_payload,
            "scientific_sha256": scientific_sha256,
            "storage": {
                "format": HOTPOTQA_MANIFEST_SHARD_FORMAT,
                "manifest_shards": storage_shards,
                "shard_size": shard_size,
            },
        }

        _publish_staged_shards(
            staging_directory,
            final_storage_directory,
            shard_metadata,
        )
        _atomic_write_json(root_path, artifact)

    return artifact


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"duplicate JSON object key: {key!r}"
                )
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
        )

    if not isinstance(value, dict):
        raise ValueError(
            "HotpotQA corpus-manifest root must be a JSON object"
        )

    return value


def _safe_shard_path(
    root_path: Path,
    stored_path: object,
) -> Path:
    if not isinstance(stored_path, str) or not stored_path:
        raise ValueError(
            "storage shard path must be a non-empty string"
        )

    if "\\" in stored_path:
        raise ValueError(
            "storage shard path must use POSIX separators"
        )

    relative = PurePosixPath(stored_path)

    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError(
            "storage shard path must be a safe relative path"
        )

    candidate = root_path.parent.joinpath(*relative.parts)

    resolved_parent = root_path.parent.resolve()
    resolved_candidate = candidate.resolve()

    try:
        resolved_candidate.relative_to(resolved_parent)
    except ValueError as error:
        raise ValueError(
            "storage shard path escapes manifest directory"
        ) from error

    return candidate


def verify_hotpotqa_streaming_corpus_manifest(
    root_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify root identity and every compact manifest shard."""
    root_path = Path(root_manifest_path)
    artifact = _load_json_object(root_path)

    _require_exact_keys(
        artifact,
        _ROOT_KEYS,
        "root artifact",
    )

    if (
        artifact["artifact_format"]
        != HOTPOTQA_CORPUS_MANIFEST_ARTIFACT_FORMAT
    ):
        raise ValueError(
            "unexpected HotpotQA corpus-manifest artifact format"
        )

    payload = artifact["scientific_payload"]

    if not isinstance(payload, Mapping):
        raise ValueError(
            "scientific_payload must be an object"
        )

    _require_exact_keys(
        payload,
        _SCIENTIFIC_PAYLOAD_KEYS,
        "scientific_payload",
    )

    expected_constants = {
        "config": HOTPOTQA_CONFIG,
        "retrieval_serialization": (
            HOTPOTQA_RETRIEVAL_SERIALIZATION
        ),
        "revision": HOTPOTQA_REVISION,
        "schema_version": (
            HOTPOTQA_CORPUS_MANIFEST_SCHEMA_VERSION
        ),
        "source": HOTPOTQA_SOURCE,
        "split": HOTPOTQA_SPLIT,
    }

    for key, expected in expected_constants.items():
        if payload[key] != expected:
            raise ValueError(
                f"scientific_payload {key} "
                f"must equal {expected!r}"
            )

    document_count = _require_positive_integer(
        payload["document_count"],
        "scientific_payload document_count",
    )

    for key in (
        "document_id_map_sha256",
        "logical_entry_stream_sha256",
    ):
        _require_sha256(
            payload[key],
            f"scientific_payload {key}",
        )

    for key in (
        "first_source_document_id",
        "last_source_document_id",
    ):
        if not isinstance(payload[key], str):
            raise TypeError(
                f"scientific_payload {key} must be a string"
            )

    declared_scientific_sha256 = _require_sha256(
        artifact["scientific_sha256"],
        "scientific_sha256",
    )

    computed_scientific_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()

    if declared_scientific_sha256 != computed_scientific_sha256:
        raise ValueError(
            "scientific_sha256 does not match scientific_payload"
        )

    storage = artifact["storage"]

    if not isinstance(storage, Mapping):
        raise ValueError("storage must be an object")

    _require_exact_keys(
        storage,
        _STORAGE_KEYS,
        "storage",
    )

    if storage["format"] != HOTPOTQA_MANIFEST_SHARD_FORMAT:
        raise ValueError(
            "unexpected HotpotQA manifest shard format"
        )

    shard_size = _require_positive_integer(
        storage["shard_size"],
        "storage shard_size",
    )

    shards = storage["manifest_shards"]

    if not isinstance(shards, list):
        raise ValueError(
            "storage manifest_shards must be a list"
        )

    expected_shard_count = (
        document_count + shard_size - 1
    ) // shard_size

    if len(shards) != expected_shard_count:
        raise ValueError(
            "manifest shard count does not match "
            "document_count and shard_size"
        )

    logical_digest = hashlib.sha256()
    document_id_map_digest = hashlib.sha256()

    seen_paths: set[Path] = set()
    seen_source_ids: set[str] = set()

    global_position = 0
    first_source_document_id: str | None = None
    last_source_document_id: str | None = None

    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError(
                f"manifest shard {shard_index} must be an object"
            )

        _require_exact_keys(
            shard,
            _SHARD_KEYS,
            f"manifest shard {shard_index}",
        )

        shard_path = _safe_shard_path(
            root_path,
            shard["path"],
        )

        resolved_shard_path = shard_path.resolve()

        if resolved_shard_path in seen_paths:
            raise ValueError(
                "manifest shard paths must be unique"
            )
        seen_paths.add(resolved_shard_path)

        remaining = document_count - global_position
        expected_row_count = min(shard_size, remaining)

        declared_row_count = _require_positive_integer(
            shard["row_count"],
            f"manifest shard {shard_index} row_count",
        )

        if declared_row_count != expected_row_count:
            raise ValueError(
                f"manifest shard {shard_index} "
                "row_count is inconsistent"
            )

        expected_first_position = global_position
        expected_last_position = (
            global_position + expected_row_count - 1
        )

        for field, expected in (
            ("first_position", expected_first_position),
            ("last_position", expected_last_position),
        ):
            value = shard[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"manifest shard {shard_index} "
                    f"{field} must be an integer"
                )
            if value != expected:
                raise ValueError(
                    f"manifest shard {shard_index} "
                    f"{field} is inconsistent"
                )

        declared_size = _require_positive_integer(
            shard["size_bytes"],
            f"manifest shard {shard_index} size_bytes",
        )

        declared_physical_sha = _require_sha256(
            shard["physical_sha256"],
            f"manifest shard {shard_index} physical_sha256",
        )

        physical_digest = hashlib.sha256()
        actual_size = 0
        actual_rows = 0

        with shard_path.open("rb") as handle:
            for raw_line in handle:
                physical_digest.update(raw_line)
                actual_size += len(raw_line)

                if not raw_line.endswith(b"\n"):
                    raise ValueError(
                        f"manifest shard {shard_index} "
                        "has a non-terminated row"
                    )

                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"manifest shard {shard_index} "
                        "contains invalid JSON"
                    ) from error

                if not isinstance(row, Mapping):
                    raise ValueError(
                        "manifest logical row must be an object"
                    )

                _require_exact_keys(
                    row,
                    _LOGICAL_ROW_KEYS,
                    "manifest logical row",
                )

                position = row["position"]

                if (
                    isinstance(position, bool)
                    or not isinstance(position, int)
                ):
                    raise TypeError(
                        "manifest logical row position "
                        "must be an integer"
                    )

                if position != global_position:
                    raise ValueError(
                        "manifest logical row position "
                        "is not contiguous"
                    )

                source_document_id = row["source_document_id"]

                if not isinstance(source_document_id, str):
                    raise TypeError(
                        "manifest source_document_id "
                        "must be a string"
                    )

                if source_document_id in seen_source_ids:
                    raise ValueError(
                        "duplicate source document ID "
                        "in manifest shards"
                    )
                seen_source_ids.add(source_document_id)

                for field in (
                    "title_sha256",
                    "text_sha256",
                    "retrieval_content_sha256",
                ):
                    _require_sha256(
                        row[field],
                        f"manifest logical row {field}",
                    )

                canonical_line = (
                    _canonical_json_bytes(row) + b"\n"
                )

                if canonical_line != raw_line:
                    raise ValueError(
                        "manifest logical row is not "
                        "canonically serialized"
                    )

                logical_digest.update(canonical_line)

                document_id_map_digest.update(
                    _canonical_json_bytes(
                        {
                            "position": position,
                            "source_document_id": (
                                source_document_id
                            ),
                        }
                    )
                    + b"\n"
                )

                if first_source_document_id is None:
                    first_source_document_id = (
                        source_document_id
                    )

                last_source_document_id = source_document_id
                global_position += 1
                actual_rows += 1

        if actual_rows != declared_row_count:
            raise ValueError(
                f"manifest shard {shard_index} "
                "row_count does not match file"
            )

        if actual_size != declared_size:
            raise ValueError(
                f"manifest shard {shard_index} "
                "size_bytes does not match file"
            )

        if (
            physical_digest.hexdigest()
            != declared_physical_sha
        ):
            raise ValueError(
                f"manifest shard {shard_index} "
                "physical SHA-256 mismatch"
            )

    if global_position != document_count:
        raise ValueError(
            "verified document count does not match manifest"
        )

    if (
        first_source_document_id
        != payload["first_source_document_id"]
    ):
        raise ValueError(
            "first source document ID does not match "
            "scientific_payload"
        )

    if (
        last_source_document_id
        != payload["last_source_document_id"]
    ):
        raise ValueError(
            "last source document ID does not match "
            "scientific_payload"
        )

    if (
        logical_digest.hexdigest()
        != payload["logical_entry_stream_sha256"]
    ):
        raise ValueError(
            "logical entry stream SHA-256 does not match "
            "scientific_payload"
        )

    if (
        document_id_map_digest.hexdigest()
        != payload["document_id_map_sha256"]
    ):
        raise ValueError(
            "document ID map SHA-256 does not match "
            "scientific_payload"
        )

    return artifact


def verify_canonical_hotpotqa_streaming_corpus_manifest(
    root_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify integrity and require full canonical HotpotQA count."""
    artifact = verify_hotpotqa_streaming_corpus_manifest(
        root_manifest_path
    )

    payload = artifact["scientific_payload"]

    if payload["document_count"] != HOTPOTQA_EXPECTED_DOCUMENT_COUNT:
        raise ValueError(
            "canonical HotpotQA corpus mismatch: "
            f"document_count must equal "
            f"{HOTPOTQA_EXPECTED_DOCUMENT_COUNT}; "
            f"found {payload['document_count']}"
        )

    return artifact


read_and_verify_hotpotqa_streaming_corpus_manifest = (
    verify_hotpotqa_streaming_corpus_manifest
)


__all__ = [
    "HOTPOTQA_CONFIG",
    "HOTPOTQA_CORPUS_MANIFEST_ARTIFACT_FORMAT",
    "HOTPOTQA_CORPUS_MANIFEST_SCHEMA_VERSION",
    "HOTPOTQA_DEFAULT_SHARD_SIZE",
    "HOTPOTQA_EXPECTED_DOCUMENT_COUNT",
    "HOTPOTQA_MANIFEST_SHARD_FORMAT",
    "HOTPOTQA_RETRIEVAL_SERIALIZATION",
    "HOTPOTQA_REVISION",
    "HOTPOTQA_SOURCE",
    "HOTPOTQA_SPLIT",
    "build_hotpotqa_streaming_corpus_manifest",
    "canonical_retrieval_content",
    "read_and_verify_hotpotqa_streaming_corpus_manifest",
    "verify_canonical_hotpotqa_streaming_corpus_manifest",
    "verify_hotpotqa_streaming_corpus_manifest",
]
