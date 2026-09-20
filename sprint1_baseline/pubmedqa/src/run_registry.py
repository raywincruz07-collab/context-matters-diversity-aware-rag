"""Canonical append-only run registry and provenance helpers for new runs.

Registry v1 governs current-protocol and prospective-backfill execution only;
verified historical evidence remains in its immutable artifacts/audit records.
Infrastructure retries keep one scientific ``run_id`` and append successive
``RUNNING`` attempt snapshots. ``FAILED`` means execution has ended after the
applicable retry budget, while ``COMPLETE`` and ``FAILED`` are terminal.

The scientific ID excludes file-byte integrity hashes, branch names, hardware,
and environment/runtime execution provenance. Those values remain recorded;
environment/runtime must remain compatible across an exact retry/resume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from evaluation.contracts import ContextMode
from evaluation.metric_registry import DatasetId
from retrieval_artifacts.contracts import canonical_stable_id


RUN_SCHEMA_VERSION = "sprint3.run-registry-record.v1"
REGISTRY_FORMAT = "context-matters.run-registry.jsonl.v1"
REGISTRY_HEADER = {
    "record_type": "registry_header",
    "registry_format": REGISTRY_FORMAT,
    "run_schema_version": RUN_SCHEMA_VERSION,
}
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/run_registry/run_registry_v1.jsonl"
)
DEFAULT_EVIDENCE_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/run_registry/evidence_manifest_authority_v1.json"
)
EVIDENCE_AUTHORITY_SCHEMA_VERSION = "sprint3.evidence-manifest-authority.v1"
CANDIDATE_SET_SCHEMA_VERSION = "sprint3.candidate-set-inventory.v1"
CANDIDATE_SET_ARTIFACT_FORMAT = "sprint3.candidate-set-inventory-artifact.v1"

RUN_STATUSES = frozenset({"PLANNED", "RUNNING", "COMPLETE", "FAILED"})
RUN_TYPES = frozenset(
    {
        "RETRIEVAL",
        "DIVERSIFICATION",
        "GENERATION",
        "EVALUATION",
        "CALIBRATION",
        "RESOURCE_PILOT",
        "ANALYSIS",
    }
)
EVIDENCE_ROLES = frozenset(
    {
        "SYNTHETIC",
        "DEVELOPMENT",
        "SELECTION",
        "PROJECT_PROTECTED_FINAL",
        "OFFICIAL_TEST_FULL",
        "HISTORICAL_OBSERVED",
        "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
    }
)
ORIGINS = frozenset({"PROSPECTIVE_BACKFILL", "CURRENT_PROTOCOL"})
DIVERSIFICATION_METHODS = frozenset(
    {"none", "mmr", "kmeans", "agglo", "dpp_map", "dpp_sample"}
)
CANONICAL_CANDIDATE_POOL = 20
CANONICAL_TOP_K = 5
MAX_INFRASTRUCTURE_ATTEMPTS_BY_RUN_TYPE = {
    "GENERATION": 3,
    "RETRIEVAL": 3,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^run-[a-z0-9][a-z0-9-]*-[0-9a-f]{24}$")
_SAMPLE_MANIFEST_ID_RE = re.compile(r"^sample-manifest:sha256:([0-9a-f]{64})$")
_CORPUS_MANIFEST_ID_RE = re.compile(r"^corpus-manifest:sha256:([0-9a-f]{64})$")
_CANDIDATE_ID_RE = re.compile(r"^candidate:sha256:[0-9a-f]{64}$")
_CANDIDATE_SET_ID_RE = re.compile(r"^candidate-set:sha256:([0-9a-f]{64})$")
_INDEX_ID_RE = re.compile(r"^index:sha256:[0-9a-f]{64}$")
_SELECTED_CONTEXT_ID_RE = re.compile(r"^selected-context:sha256:[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "created_at",
        "sprint",
        "stage",
        "run_type",
        "evidence_role",
        "origin",
        "protocol_config_bundle_sha256",
        "git",
        "data",
        "retrieval",
        "diversification",
        "context_mode",
        "generation",
        "evaluation",
        "execution",
        "output",
    }
)


class RunRecordValidationError(ValueError):
    """A run record violates the canonical schema or stage semantics."""


class RunRegistryConflictError(ValueError):
    """A run ID already exists with conflicting identity or lifecycle state."""


def canonical_json(value: object) -> str:
    """Serialize JSON-compatible identity data deterministically."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RunRecordValidationError("value is not canonical JSON data") from exc


def stable_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunRecordValidationError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise RunRecordValidationError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    keys = frozenset(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise RunRecordValidationError(
            f"{name} keys mismatch; missing={missing}, extra={extra}"
        )


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunRecordValidationError(f"{name} must be a non-empty string")
    return value


def _require_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name)


def _require_sha256(value: object, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RunRecordValidationError(
            f"{name} must be a lowercase 64-character SHA-256"
        )


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunRecordValidationError(f"{name} must be a nonnegative integer")
    return value


def _require_relative_path(value: object, name: str) -> str:
    text = _require_text(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise RunRecordValidationError(
            f"{name} must be a repository-relative path without '..'"
        )
    return path.as_posix()


def _parse_timestamp(value: object, name: str, *, optional: bool = False) -> datetime | None:
    if optional and value is None:
        return None
    text = _require_text(value, name)
    if not text.endswith("Z"):
        raise RunRecordValidationError(f"{name} must be an ISO-8601 UTC timestamp ending Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise RunRecordValidationError(f"{name} must be a valid timestamp") from exc
    return parsed


def _validate_artifact_ref(
    value: object,
    name: str,
    *,
    optional: bool = False,
    artifact_id_pattern: re.Pattern[str] | None = None,
    artifact_id_required: bool = False,
) -> None:
    if optional and value is None:
        return
    ref = _require_mapping(value, name)
    _require_exact_keys(ref, frozenset({"path", "sha256", "artifact_id"}), name)
    _require_relative_path(ref["path"], f"{name}.path")
    _require_sha256(ref["sha256"], f"{name}.sha256")
    artifact_id = _require_optional_text(ref["artifact_id"], f"{name}.artifact_id")
    if artifact_id_required and artifact_id is None:
        raise RunRecordValidationError(f"{name}.artifact_id is required")
    if (
        artifact_id is not None
        and artifact_id_pattern is not None
        and artifact_id_pattern.fullmatch(artifact_id) is None
    ):
        raise RunRecordValidationError(f"{name}.artifact_id has invalid syntax")


def artifact_ref(
    path: Path,
    *,
    repository_root: Path,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Create a compact reference to one existing repository artifact."""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(Path(repository_root).resolve()).as_posix()
    except ValueError as exc:
        raise RunRecordValidationError("artifact path is outside repository_root") from exc
    if not resolved.is_file():
        raise RunRecordValidationError(f"artifact file does not exist: {relative}")
    result = {
        "path": relative,
        "sha256": file_sha256(resolved),
        "artifact_id": artifact_id,
    }
    _validate_artifact_ref(result, "artifact")
    return result


def _atomic_write_new(path: Path, serialized: str) -> None:
    """Create one immutable JSON artifact without overwriting a concurrent file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
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
        os.link(temporary_path, path)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise RunRegistryConflictError(f"existing artifact differs: {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def _canonical_sample_id(value: object, name: str) -> str | int:
    try:
        return canonical_stable_id(value, name)
    except (TypeError, ValueError) as exc:
        raise RunRecordValidationError(str(exc)) from exc


def candidate_set_scientific_payload(
    *,
    dataset: str,
    evidence_role: str,
    sample_manifest_id: str,
    retriever: str,
    expected_query_count: int,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the stable sample-to-candidate scientific inventory payload."""
    if dataset not in {item.value for item in DatasetId}:
        raise RunRecordValidationError("candidate set dataset is not canonical")
    if evidence_role not in EVIDENCE_ROLES:
        raise RunRecordValidationError("candidate set evidence_role is not canonical")
    if _SAMPLE_MANIFEST_ID_RE.fullmatch(sample_manifest_id) is None:
        raise RunRecordValidationError("candidate set sample_manifest_id is invalid")
    _require_text(retriever, "candidate set retriever")
    count = _require_nonnegative_int(expected_query_count, "expected_query_count")
    if count == 0:
        raise RunRecordValidationError("expected_query_count must be positive")
    canonical_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        entry = _require_mapping(raw, f"entries[{index}]")
        _require_exact_keys(
            entry,
            frozenset({"sample_id", "candidate_artifact_id"}),
            f"entries[{index}]",
        )
        sample_id = _canonical_sample_id(entry["sample_id"], f"entries[{index}].sample_id")
        identity_key = canonical_json(sample_id)
        if identity_key in seen:
            raise RunRecordValidationError("candidate set sample IDs must be unique")
        seen.add(identity_key)
        candidate_id = _require_text(
            entry["candidate_artifact_id"],
            f"entries[{index}].candidate_artifact_id",
        )
        if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
            raise RunRecordValidationError("candidate artifact ID has invalid syntax")
        canonical_entries.append(
            {"sample_id": sample_id, "candidate_artifact_id": candidate_id}
        )
    if len(canonical_entries) != count:
        raise RunRecordValidationError(
            "candidate set entries must equal expected_query_count"
        )
    canonical_entries.sort(key=lambda entry: canonical_json(entry["sample_id"]))
    return {
        "schema_version": CANDIDATE_SET_SCHEMA_VERSION,
        "dataset": dataset,
        "evidence_role": evidence_role,
        "sample_manifest_id": sample_manifest_id,
        "retriever": retriever,
        "expected_query_count": count,
        "entries": canonical_entries,
    }


def candidate_set_artifact_payload(
    scientific_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = candidate_set_scientific_payload(
        dataset=scientific_payload["dataset"],
        evidence_role=scientific_payload["evidence_role"],
        sample_manifest_id=scientific_payload["sample_manifest_id"],
        retriever=scientific_payload["retriever"],
        expected_query_count=scientific_payload["expected_query_count"],
        entries=scientific_payload["entries"],
    )
    scientific_sha256 = stable_json_sha256(payload)
    provenance_value = {} if provenance is None else dict(provenance)
    canonical_json(provenance_value)
    return {
        "artifact_format": CANDIDATE_SET_ARTIFACT_FORMAT,
        "candidate_set_id": f"candidate-set:sha256:{scientific_sha256}",
        "scientific_sha256": scientific_sha256,
        "scientific_payload": payload,
        "provenance": provenance_value,
    }


def write_candidate_set_artifact(
    path: Path,
    scientific_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    wrapper = candidate_set_artifact_payload(
        scientific_payload, provenance=provenance
    )
    _atomic_write_new(Path(path), canonical_json(wrapper) + "\n")


def read_candidate_set_artifact(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    wrapper = _require_mapping(stored, "candidate set artifact")
    _require_exact_keys(
        wrapper,
        frozenset(
            {
                "artifact_format",
                "candidate_set_id",
                "scientific_sha256",
                "scientific_payload",
                "provenance",
            }
        ),
        "candidate set artifact",
    )
    if wrapper["artifact_format"] != CANDIDATE_SET_ARTIFACT_FORMAT:
        raise RunRecordValidationError("candidate set artifact format is invalid")
    rebuilt = candidate_set_artifact_payload(
        _require_mapping(wrapper["scientific_payload"], "scientific_payload"),
        provenance=_require_mapping(wrapper["provenance"], "provenance"),
    )
    if rebuilt != dict(wrapper):
        raise RunRecordValidationError("candidate set artifact identity mismatch")
    return rebuilt


def candidate_set_artifact_ref(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    wrapper = read_candidate_set_artifact(path)
    return artifact_ref(
        path,
        repository_root=repository_root,
        artifact_id=wrapper["candidate_set_id"],
    )


def _verify_sample_manifest_artifact(
    path: Path, *, expected_manifest_id: str
) -> None:
    with Path(path).open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    wrapper = _require_mapping(stored, "sample manifest artifact")
    scientific_payload = _require_mapping(
        wrapper.get("scientific_payload"), "sample manifest scientific_payload"
    )
    scientific_sha256 = stable_json_sha256(scientific_payload)
    scientific_id = f"sample-manifest:sha256:{scientific_sha256}"
    if wrapper.get("sha256") != scientific_sha256:
        raise RunRecordValidationError("sample manifest scientific SHA-256 mismatch")
    if wrapper.get("manifest_id") != scientific_id:
        raise RunRecordValidationError("sample manifest scientific ID mismatch")
    if expected_manifest_id != scientific_id:
        raise RunRecordValidationError(
            "evidence authority does not match referenced sample manifest"
        )


def load_evidence_manifest_authority(
    path: Path, *, repository_root: Path | None = None
) -> dict[tuple[str, str], str]:
    """Load the prospective dataset/role-to-sample-manifest authority map."""
    with Path(path).open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    root = _require_mapping(stored, "evidence manifest authority")
    _require_exact_keys(root, frozenset({"schema_version", "authorities"}), "authority")
    if root["schema_version"] != EVIDENCE_AUTHORITY_SCHEMA_VERSION:
        raise RunRecordValidationError("evidence authority schema_version is invalid")
    authorities = root["authorities"]
    if not isinstance(authorities, list):
        raise RunRecordValidationError("authority.authorities must be a list")
    result: dict[tuple[str, str], str] = {}
    for index, raw in enumerate(authorities):
        entry = _require_mapping(raw, f"authorities[{index}]")
        _require_exact_keys(
            entry,
            frozenset(
                {
                    "dataset",
                    "evidence_role",
                    "sample_manifest_id",
                    "sample_manifest_path",
                    "authority_protocols",
                }
            ),
            f"authorities[{index}]",
        )
        dataset = entry["dataset"]
        role = entry["evidence_role"]
        if dataset not in {item.value for item in DatasetId}:
            raise RunRecordValidationError("authority dataset is not canonical")
        if role not in EVIDENCE_ROLES or role == "SYNTHETIC":
            raise RunRecordValidationError("authority evidence role is invalid")
        manifest_id = _require_text(
            entry["sample_manifest_id"], f"authorities[{index}].sample_manifest_id"
        )
        if _SAMPLE_MANIFEST_ID_RE.fullmatch(manifest_id) is None:
            raise RunRecordValidationError("authority sample manifest ID is invalid")
        manifest_path = _require_relative_path(
            entry["sample_manifest_path"], f"authorities[{index}].sample_manifest_path"
        )
        protocols = entry["authority_protocols"]
        if not isinstance(protocols, list) or not protocols:
            raise RunRecordValidationError("authority_protocols must be a non-empty list")
        for protocol_index, protocol in enumerate(protocols):
            _require_relative_path(
                protocol,
                f"authorities[{index}].authority_protocols[{protocol_index}]",
            )
        key = (dataset, role)
        if key in result:
            raise RunRecordValidationError("duplicate dataset/evidence authority")
        result[key] = manifest_id
        if repository_root is not None:
            _verify_sample_manifest_artifact(
                Path(repository_root) / manifest_path,
                expected_manifest_id=manifest_id,
            )
    return result


def _validate_evidence_manifest_binding(
    record: Mapping[str, Any], *, authority_path: Path
) -> None:
    if record["evidence_role"] == "SYNTHETIC":
        return
    authority_path = Path(authority_path)
    repository_root = None
    if authority_path.resolve() == DEFAULT_EVIDENCE_AUTHORITY_PATH.resolve():
        repository_root = Path(__file__).resolve().parents[1]
    authorities = load_evidence_manifest_authority(
        authority_path, repository_root=repository_root
    )
    key = (record["data"]["dataset"], record["evidence_role"])
    authorized_id = authorities.get(key)
    if authorized_id is None:
        raise RunRecordValidationError(
            "no authorized sample manifest for dataset/evidence_role"
        )
    declared_id = record["data"]["sample_manifest"]["artifact_id"]
    if declared_id != authorized_id:
        raise RunRecordValidationError(
            "sample manifest does not match dataset/evidence-role authority"
        )


def output_artifact(
    path: Path,
    *,
    repository_root: Path,
    row_count: int | None,
    status_counts: Mapping[str, int],
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Inventory one output file with optional row/status accounting."""
    reference = artifact_ref(
        path, repository_root=repository_root, artifact_id=artifact_id
    )
    counts = dict(status_counts)
    if not all(isinstance(key, str) and key.strip() for key in counts):
        raise RunRecordValidationError("status_counts keys must be non-empty strings")
    for key, count in counts.items():
        _require_nonnegative_int(count, f"status_counts[{key!r}]")
    if row_count is None:
        if counts:
            raise RunRecordValidationError(
                "status_counts must be empty when row_count is not applicable"
            )
    else:
        _require_nonnegative_int(row_count, "row_count")
        if sum(counts.values()) != row_count:
            raise RunRecordValidationError("status_counts must sum to row_count")
    return {**reference, "row_count": row_count, "status_counts": counts}


def output_inventory_sha256(artifacts: Sequence[Mapping[str, Any]]) -> str:
    values = [dict(value) for value in artifacts]
    for index, value in enumerate(values):
        _validate_output_artifact(value, f"artifacts[{index}]")
    return stable_json_sha256(values)


def _validate_output_artifact(value: object, name: str) -> None:
    item = _require_mapping(value, name)
    _require_exact_keys(
        item,
        frozenset({"path", "sha256", "artifact_id", "row_count", "status_counts"}),
        name,
    )
    _validate_artifact_ref(
        {key: item[key] for key in ("path", "sha256", "artifact_id")}, name
    )
    row_count = item["row_count"]
    counts = _require_mapping(item["status_counts"], f"{name}.status_counts")
    if not all(isinstance(key, str) and key.strip() for key in counts):
        raise RunRecordValidationError(f"{name}.status_counts has an invalid key")
    for key, count in counts.items():
        _require_nonnegative_int(count, f"{name}.status_counts[{key!r}]")
    if row_count is None:
        if counts:
            raise RunRecordValidationError(
                f"{name}.status_counts must be empty when row_count is null"
            )
    else:
        _require_nonnegative_int(row_count, f"{name}.row_count")
        if sum(counts.values()) != row_count:
            raise RunRecordValidationError(
                f"{name}.status_counts must sum to row_count"
            )


def _validate_data(value: object) -> None:
    data = _require_mapping(value, "data")
    _require_exact_keys(
        data,
        frozenset(
            {
                "dataset",
                "split",
                "source",
                "revision",
                "sample_manifest",
                "corpus_manifest",
            }
        ),
        "data",
    )
    if data["dataset"] not in {item.value for item in DatasetId}:
        raise RunRecordValidationError("data.dataset is not canonical")
    _require_text(data["split"], "data.split")
    _require_text(data["source"], "data.source")
    _require_text(data["revision"], "data.revision")
    _validate_artifact_ref(
        data["sample_manifest"],
        "data.sample_manifest",
        artifact_id_pattern=_SAMPLE_MANIFEST_ID_RE,
        artifact_id_required=True,
    )
    _validate_artifact_ref(
        data["corpus_manifest"],
        "data.corpus_manifest",
        optional=True,
        artifact_id_pattern=_CORPUS_MANIFEST_ID_RE,
        artifact_id_required=True,
    )


def _validate_retrieval(value: object, *, canonical_required: bool) -> None:
    retrieval = _require_mapping(value, "retrieval")
    _require_exact_keys(
        retrieval,
        frozenset(
            {
                "retriever",
                "config_sha256",
                "index",
                "candidate_set",
                "selected_context",
                "candidate_pool",
                "top_k",
            }
        ),
        "retrieval",
    )
    _require_text(retrieval["retriever"], "retrieval.retriever")
    _require_sha256(retrieval["config_sha256"], "retrieval.config_sha256")
    _validate_artifact_ref(
        retrieval["index"],
        "retrieval.index",
        optional=True,
        artifact_id_pattern=_INDEX_ID_RE,
        artifact_id_required=True,
    )
    _validate_artifact_ref(
        retrieval["candidate_set"],
        "retrieval.candidate_set",
        optional=True,
        artifact_id_pattern=_CANDIDATE_SET_ID_RE,
        artifact_id_required=True,
    )
    _validate_artifact_ref(
        retrieval["selected_context"],
        "retrieval.selected_context",
        optional=True,
        artifact_id_pattern=_SELECTED_CONTEXT_ID_RE,
        artifact_id_required=True,
    )
    if retrieval["retriever"].lower() == "bm25":
        if retrieval["index"] is not None:
            raise RunRecordValidationError("BM25 must not carry a neural/index artifact")
    elif retrieval["index"] is None:
        raise RunRecordValidationError("neural retriever requires an index identity")
    candidate_pool = _require_nonnegative_int(
        retrieval["candidate_pool"], "retrieval.candidate_pool"
    )
    top_k = _require_nonnegative_int(retrieval["top_k"], "retrieval.top_k")
    if candidate_pool == 0 or top_k == 0 or top_k > candidate_pool:
        raise RunRecordValidationError("retrieval requires candidate_pool >= top_k > 0")
    if canonical_required and (
        candidate_pool != CANONICAL_CANDIDATE_POOL or top_k != CANONICAL_TOP_K
    ):
        raise RunRecordValidationError(
            "canonical run requires candidate_pool=20 and top_k=5"
        )


def _validate_diversification(value: object) -> None:
    diversification = _require_mapping(value, "diversification")
    _require_exact_keys(
        diversification,
        frozenset({"method", "parameters", "config_sha256", "seed"}),
        "diversification",
    )
    method = diversification["method"]
    if method not in DIVERSIFICATION_METHODS:
        raise RunRecordValidationError("diversification.method is not canonical")
    parameters = _require_mapping(diversification["parameters"], "parameters")
    _require_sha256(
        diversification["config_sha256"], "diversification.config_sha256"
    )
    if diversification["config_sha256"] != stable_json_sha256(parameters):
        raise RunRecordValidationError(
            "diversification.config_sha256 does not match parameters"
        )
    seed = diversification["seed"]
    if seed is not None:
        _require_nonnegative_int(seed, "diversification.seed")
    if method == "dpp_sample":
        if seed is None:
            raise RunRecordValidationError("stochastic DPP requires an explicit seed")
    elif method == "kmeans":
        if seed != 42:
            raise RunRecordValidationError("canonical KMeans requires frozen seed=42")
    elif seed is not None:
        raise RunRecordValidationError(
            f"deterministic {method} must not carry a random seed"
        )


def _validate_generation(value: object) -> None:
    generation = _require_mapping(value, "generation")
    _require_exact_keys(
        generation,
        frozenset(
            {
                "llm_logical_id",
                "provider",
                "physical_model_id",
                "model_revision",
                "model_revision_kind",
                "prompt_sha256",
                "decoding_sha256",
            }
        ),
        "generation",
    )
    for name in ("llm_logical_id", "provider", "physical_model_id"):
        _require_text(generation[name], f"generation.{name}")
    revision = _require_optional_text(
        generation["model_revision"], "generation.model_revision"
    )
    revision_kind = generation["model_revision_kind"]
    if revision_kind not in {
        "IMMUTABLE_REVISION",
        "PROVIDER_SNAPSHOT",
        "NOT_PROVIDED_BY_PROVIDER",
    }:
        raise RunRecordValidationError("generation.model_revision_kind is invalid")
    if revision_kind == "NOT_PROVIDED_BY_PROVIDER":
        if revision is not None:
            raise RunRecordValidationError(
                "NOT_PROVIDED_BY_PROVIDER requires null model_revision"
            )
    elif revision is None:
        raise RunRecordValidationError(
            "physical model identity requires an immutable revision or provider snapshot"
        )
    _require_sha256(generation["prompt_sha256"], "generation.prompt_sha256")
    _require_sha256(generation["decoding_sha256"], "generation.decoding_sha256")


def _validate_evaluation(value: object) -> None:
    evaluation = _require_mapping(value, "evaluation")
    _require_exact_keys(
        evaluation,
        frozenset({"bundle_version", "bundle_sha256", "metric_registry_sha256"}),
        "evaluation",
    )
    _require_text(evaluation["bundle_version"], "evaluation.bundle_version")
    _require_sha256(evaluation["bundle_sha256"], "evaluation.bundle_sha256")
    _require_sha256(
        evaluation["metric_registry_sha256"], "evaluation.metric_registry_sha256"
    )


def _validate_git(value: object) -> None:
    git = _require_mapping(value, "git")
    _require_exact_keys(
        git,
        frozenset({"commit", "branch", "worktree_clean", "worktree_diff_sha256"}),
        "git",
    )
    if not isinstance(git["commit"], str) or _GIT_SHA_RE.fullmatch(git["commit"]) is None:
        raise RunRecordValidationError("git.commit must be a full lowercase Git SHA")
    _require_text(git["branch"], "git.branch")
    if type(git["worktree_clean"]) is not bool:
        raise RunRecordValidationError("git.worktree_clean must be a bool")
    _require_sha256(
        git["worktree_diff_sha256"], "git.worktree_diff_sha256", optional=True
    )
    if git["worktree_clean"] and git["worktree_diff_sha256"] is not None:
        raise RunRecordValidationError("clean Git state must not have a diff hash")
    if not git["worktree_clean"] and git["worktree_diff_sha256"] is None:
        raise RunRecordValidationError("dirty Git state requires a diff hash")


def _validate_execution(
    value: object, *, run_type: str
) -> tuple[str, datetime | None, datetime | None]:
    execution = _require_mapping(value, "execution")
    _require_exact_keys(
        execution,
        frozenset(
            {
                "environment_sha256",
                "runtime_sha256",
                "hardware_summary",
                "started_at",
                "completed_at",
                "status",
                "attempt_count",
                "failure_reason",
                "parent_run_id",
                "resume_of",
            }
        ),
        "execution",
    )
    _require_sha256(
        execution["environment_sha256"], "execution.environment_sha256", optional=True
    )
    _require_sha256(
        execution["runtime_sha256"], "execution.runtime_sha256", optional=True
    )
    _require_optional_text(execution["hardware_summary"], "execution.hardware_summary")
    started = _parse_timestamp(
        execution["started_at"], "execution.started_at", optional=True
    )
    completed = _parse_timestamp(
        execution["completed_at"], "execution.completed_at", optional=True
    )
    status = execution["status"]
    if status not in RUN_STATUSES:
        raise RunRecordValidationError("execution.status is not canonical")
    attempts = _require_nonnegative_int(
        execution["attempt_count"], "execution.attempt_count"
    )
    max_attempts = MAX_INFRASTRUCTURE_ATTEMPTS_BY_RUN_TYPE.get(run_type)
    if max_attempts is not None and attempts > max_attempts:
        raise RunRecordValidationError(
            f"{run_type} permits at most {max_attempts} total attempts"
        )
    reason = _require_optional_text(
        execution["failure_reason"], "execution.failure_reason"
    )
    parent = _require_optional_text(execution["parent_run_id"], "execution.parent_run_id")
    resume = _require_optional_text(execution["resume_of"], "execution.resume_of")
    if parent is not None and resume is not None:
        raise RunRecordValidationError("parent_run_id and resume_of are mutually exclusive")
    if status == "PLANNED":
        if started is not None or completed is not None or attempts != 0 or reason is not None:
            raise RunRecordValidationError("PLANNED lifecycle fields are inconsistent")
    elif status == "RUNNING":
        if started is None or completed is not None or attempts < 1:
            raise RunRecordValidationError("RUNNING lifecycle fields are inconsistent")
        if attempts == 1 and reason is not None:
            raise RunRecordValidationError(
                "first RUNNING attempt must not have a prior failure reason"
            )
        if attempts > 1 and reason is None:
            raise RunRecordValidationError(
                "retried RUNNING attempt requires the prior infrastructure failure reason"
            )
    elif status == "COMPLETE":
        if started is None or completed is None or attempts < 1 or reason is not None:
            raise RunRecordValidationError("COMPLETE lifecycle fields are inconsistent")
    elif status == "FAILED":
        if started is None or completed is None or attempts < 1 or reason is None:
            raise RunRecordValidationError("FAILED lifecycle fields are inconsistent")
    if started is not None and completed is not None and completed < started:
        raise RunRecordValidationError("completed_at precedes started_at")
    return status, started, completed


def _validate_output(value: object, *, status: str) -> None:
    output = _require_mapping(value, "output")
    _require_exact_keys(
        output,
        frozenset(
            {
                "expected_row_count",
                "completed_row_count",
                "successful_row_count",
                "failed_row_count",
                "output_directory",
                "partial_output_retained",
                "artifacts",
                "output_inventory_sha256",
                "raw_artifact_sha256",
            }
        ),
        "output",
    )
    expected = _require_nonnegative_int(
        output["expected_row_count"], "output.expected_row_count"
    )
    completed = _require_nonnegative_int(
        output["completed_row_count"], "output.completed_row_count"
    )
    successful = _require_nonnegative_int(
        output["successful_row_count"], "output.successful_row_count"
    )
    failed = _require_nonnegative_int(
        output["failed_row_count"], "output.failed_row_count"
    )
    if completed != successful + failed or completed > expected:
        raise RunRecordValidationError("output row counts are inconsistent")
    _require_relative_path(output["output_directory"], "output.output_directory")
    if type(output["partial_output_retained"]) is not bool:
        raise RunRecordValidationError("output.partial_output_retained must be a bool")
    artifacts = output["artifacts"]
    if not isinstance(artifacts, list):
        raise RunRecordValidationError("output.artifacts must be a list")
    for index, artifact in enumerate(artifacts):
        _validate_output_artifact(artifact, f"output.artifacts[{index}]")
    _require_sha256(
        output["output_inventory_sha256"],
        "output.output_inventory_sha256",
        optional=True,
    )
    _require_sha256(
        output["raw_artifact_sha256"], "output.raw_artifact_sha256", optional=True
    )
    if artifacts:
        expected_inventory = output_inventory_sha256(artifacts)
        if output["output_inventory_sha256"] != expected_inventory:
            raise RunRecordValidationError("output inventory hash does not match artifacts")
    elif output["output_inventory_sha256"] is not None:
        raise RunRecordValidationError("empty output inventory must have a null hash")
    if status == "PLANNED" and (
        completed != 0
        or output["partial_output_retained"]
        or artifacts
        or output["output_inventory_sha256"] is not None
        or output["raw_artifact_sha256"] is not None
    ):
        raise RunRecordValidationError("PLANNED output fields are inconsistent")
    if status == "COMPLETE" and (
        completed != expected
        or output["partial_output_retained"]
        or not artifacts
        or output["output_inventory_sha256"] is None
        or output["raw_artifact_sha256"] is None
    ):
        raise RunRecordValidationError("COMPLETE output fields are inconsistent")
    if status == "FAILED":
        retained = output["partial_output_retained"]
        if retained and (completed == 0 or not artifacts):
            raise RunRecordValidationError(
                "retained partial output requires completed rows and artifacts"
            )
        if not retained and (
            completed != 0
            or artifacts
            or output["output_inventory_sha256"] is not None
            or output["raw_artifact_sha256"] is not None
        ):
            raise RunRecordValidationError(
                "FAILED without retained partial output must preserve explicit absence"
            )


def _artifact_identity(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"artifact_id": value["artifact_id"]}


def run_identity_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return pre-outcome identity; lifecycle and performance never enter it."""
    retrieval = record["retrieval"]
    data = record["data"]
    return {
        "schema_version": record["schema_version"],
        "sprint": record["sprint"],
        "stage": record["stage"],
        "run_type": record["run_type"],
        "evidence_role": record["evidence_role"],
        "origin": record["origin"],
        "protocol_config_bundle_sha256": record["protocol_config_bundle_sha256"],
        "git": {
            "commit": record["git"]["commit"],
            "worktree_diff_sha256": record["git"]["worktree_diff_sha256"],
        },
        "data": {
            "dataset": data["dataset"],
            "split": data["split"],
            "source": data["source"],
            "revision": data["revision"],
            "sample_manifest": _artifact_identity(data["sample_manifest"]),
            "corpus_manifest": _artifact_identity(data["corpus_manifest"]),
        },
        "retrieval": (
            None
            if retrieval is None
            else {
                "retriever": retrieval["retriever"],
                "config_sha256": retrieval["config_sha256"],
                "index": _artifact_identity(retrieval["index"]),
                "candidate_set": _artifact_identity(retrieval["candidate_set"]),
                "selected_context": _artifact_identity(
                    retrieval["selected_context"]
                ),
                "candidate_pool": retrieval["candidate_pool"],
                "top_k": retrieval["top_k"],
            }
        ),
        "diversification": record["diversification"],
        "context_mode": record["context_mode"],
        "generation": record["generation"],
        "evaluation": record["evaluation"],
        "expected_row_count": record["output"]["expected_row_count"],
    }


def _slug(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "na"


def derive_run_id(record: Mapping[str, Any]) -> str:
    """Build a human-readable ID plus 96-bit digest of immutable pre-run identity."""
    retrieval = record.get("retrieval")
    generation = record.get("generation")
    prefix_parts = (
        record.get("sprint", "na"),
        record.get("data", {}).get("dataset", "na"),
        f"s{record.get('stage', 'na')}",
        record.get("context_mode") or "na",
        "shared" if record.get("context_mode") == ContextMode.WITHOUT_CONTEXT.value else (
            retrieval.get("retriever", "na") if isinstance(retrieval, Mapping) else "na"
        ),
        record.get("diversification", {}).get("method", "none")
        if isinstance(record.get("diversification"), Mapping)
        else "none",
        generation.get("llm_logical_id", "na")
        if isinstance(generation, Mapping)
        else "na",
    )
    digest = stable_json_sha256(run_identity_payload(record))[:24]
    return "run-" + "-".join(_slug(value) for value in prefix_parts) + f"-{digest}"


def _validate_dataset_role_origin(
    *, dataset: str, role: str, origin: str, run_type: str
) -> None:
    if role == "SYNTHETIC":
        if origin != "CURRENT_PROTOCOL":
            raise RunRecordValidationError(
                "synthetic evidence requires CURRENT_PROTOCOL origin"
            )
        return
    if dataset == "pubmedqa":
        if role not in {
            "HISTORICAL_OBSERVED",
            "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        }:
            raise RunRecordValidationError(
                "PubMedQA is exposed historical/control evidence, never selection or protected"
            )
        if role == "HISTORICAL_OBSERVED" and run_type != "ANALYSIS":
            raise RunRecordValidationError(
                "historical PubMedQA evidence may only be referenced by ANALYSIS"
            )
        if role == "HISTORICAL_OBSERVED_CONTROL_REPLICATION" and origin != "PROSPECTIVE_BACKFILL":
            raise RunRecordValidationError(
                "PubMedQA control replication requires PROSPECTIVE_BACKFILL origin"
            )
    elif dataset in {"hotpotqa", "asqa"}:
        authorized_roles = {
            "DEVELOPMENT",
            "SELECTION",
            "PROJECT_PROTECTED_FINAL",
            "HISTORICAL_OBSERVED",
        }
        if dataset == "hotpotqa":
            authorized_roles.add("OFFICIAL_TEST_FULL")

        if role not in authorized_roles:
            raise RunRecordValidationError(
                f"{dataset} evidence role is not authorized"
            )
        if role == "HISTORICAL_OBSERVED" and run_type != "ANALYSIS":
            raise RunRecordValidationError(
                "historical evidence may only be referenced by ANALYSIS"
            )

        current_roles = {
            "DEVELOPMENT",
            "SELECTION",
            "PROJECT_PROTECTED_FINAL",
        }
        if dataset == "hotpotqa":
            current_roles.add("OFFICIAL_TEST_FULL")

        if role in current_roles and origin != "CURRENT_PROTOCOL":
            raise RunRecordValidationError(
                "current HotpotQA/ASQA roles require CURRENT_PROTOCOL origin"
            )


def _validate_stage_role(*, stage: int, role: str, run_type: str) -> None:
    historical_roles = {
        "HISTORICAL_OBSERVED",
        "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
    }
    if stage == 1:
        if role not in {"SYNTHETIC", "DEVELOPMENT", *historical_roles}:
            raise RunRecordValidationError("Stage 1 evidence role is invalid")
        return
    if stage == 2:
        if run_type != "ANALYSIS" or role not in {"DEVELOPMENT", *historical_roles}:
            raise RunRecordValidationError(
                "Stage 2 permits only gate analysis of development/historical evidence"
            )
        return
    if stage == 3:
        if role not in {"SELECTION", "HISTORICAL_OBSERVED_CONTROL_REPLICATION"}:
            raise RunRecordValidationError("Stage 3 evidence role is invalid")
        return
    if stage == 4:
        if run_type != "ANALYSIS" or role not in {"SELECTION", *historical_roles}:
            raise RunRecordValidationError(
                "Stage 4 permits only pre-protected analysis/locking of exposed evidence"
            )
        return
    if stage == 5:
        if role not in {
            "PROJECT_PROTECTED_FINAL",
            "OFFICIAL_TEST_FULL",
            "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        }:
            raise RunRecordValidationError("Stage 5 evidence role is invalid")
        return
    if stage == 6:
        if run_type != "ANALYSIS" or role not in {
            "DEVELOPMENT",
            "SELECTION",
            "PROJECT_PROTECTED_FINAL",
            "OFFICIAL_TEST_FULL",
            *historical_roles,
        }:
            raise RunRecordValidationError(
                "Stage 6 permits predefined analysis with the consumed evidence role retained"
            )
        return
    raise RunRecordValidationError("Stage 0 does not register research execution")


def _validate_run_type_payload(record: Mapping[str, Any]) -> None:
    run_type = record["run_type"]
    retrieval = record["retrieval"]
    diversification = record["diversification"]
    generation = record["generation"]
    evaluation = record["evaluation"]
    context_mode = record["context_mode"]
    if run_type == "RETRIEVAL":
        if retrieval is None or any(
            value is not None
            for value in (diversification, generation, evaluation, context_mode)
        ):
            raise RunRecordValidationError(
                "RETRIEVAL requires retrieval only and no treatment/evaluator payload"
            )
        if retrieval["candidate_set"] is not None or retrieval["selected_context"] is not None:
            raise RunRecordValidationError(
                "RETRIEVAL candidate/selected-context sets are outputs, not pre-run inputs"
            )
    elif run_type == "DIVERSIFICATION":
        if (
            retrieval is None
            or diversification is None
            or generation is not None
            or evaluation is not None
            or context_mode is not None
        ):
            raise RunRecordValidationError(
                "DIVERSIFICATION requires retrieval and diversification only"
            )
        if retrieval["candidate_set"] is None:
            raise RunRecordValidationError(
                "DIVERSIFICATION requires an aggregate candidate-set identity"
            )
        if retrieval["selected_context"] is not None:
            raise RunRecordValidationError(
                "selected context is a DIVERSIFICATION output, not a pre-run input"
            )
    elif run_type == "GENERATION":
        if generation is None or context_mode is None or evaluation is not None:
            raise RunRecordValidationError(
                "GENERATION requires generation/context and forbids evaluation payload"
            )
        if context_mode == ContextMode.WITH_CONTEXT.value:
            if retrieval is None or diversification is None:
                raise RunRecordValidationError(
                    "WITH_CONTEXT GENERATION requires retrieval and canonical diversification"
                )
            if retrieval["candidate_set"] is None:
                raise RunRecordValidationError(
                    "WITH_CONTEXT GENERATION requires aggregate candidate-set identity"
                )
            if retrieval["selected_context"] is None:
                raise RunRecordValidationError(
                    "WITH_CONTEXT GENERATION requires selected-context identity"
                )
        elif retrieval is not None or diversification is not None:
            raise RunRecordValidationError(
                "WITHOUT_CONTEXT GENERATION forbids retrieval/diversification"
            )
    elif run_type == "EVALUATION":
        if evaluation is None or generation is not None or context_mode is not None:
            raise RunRecordValidationError(
                "EVALUATION requires evaluation payload and forbids generation/context payload"
            )
    elif run_type == "CALIBRATION":
        if evaluation is None or generation is not None or context_mode is not None:
            raise RunRecordValidationError(
                "CALIBRATION requires evaluation payload and forbids generation/context payload"
            )
    elif run_type == "ANALYSIS":
        if generation is not None or context_mode is not None:
            raise RunRecordValidationError(
                "ANALYSIS consumes existing artifacts and forbids generation/context execution"
            )
        if retrieval is not None:
            if diversification is None:
                raise RunRecordValidationError(
                    "ANALYSIS retrieval lineage requires canonical diversification, including method=none baseline"
                )
            if retrieval["candidate_set"] is None:
                raise RunRecordValidationError(
                    "ANALYSIS retrieval lineage requires aggregate candidate-set identity"
                )
        elif diversification is not None:
            raise RunRecordValidationError(
                "ANALYSIS diversification lineage requires retrieval identity"
            )
    elif run_type == "RESOURCE_PILOT":
        if generation is not None or context_mode is not None:
            raise RunRecordValidationError(
                "RESOURCE_PILOT forbids generation/context execution"
            )
        if retrieval is None:
            if diversification is not None:
                raise RunRecordValidationError(
                    "RESOURCE_PILOT diversification lineage requires retrieval identity"
                )
        elif (
            retrieval["candidate_set"] is not None
            or retrieval["selected_context"] is not None
        ) and diversification is None:
            raise RunRecordValidationError(
                "RESOURCE_PILOT downstream retrieval lineage requires canonical diversification"
            )


def validate_run_record(
    record: Mapping[str, Any],
    *,
    evidence_authority_path: Path = DEFAULT_EVIDENCE_AUTHORITY_PATH,
) -> dict[str, Any]:
    """Validate and return a detached canonical JSON-compatible record."""
    record = _require_mapping(record, "record")
    _require_exact_keys(record, _TOP_LEVEL_KEYS, "record")
    if record["schema_version"] != RUN_SCHEMA_VERSION:
        raise RunRecordValidationError(
            f"schema_version must equal {RUN_SCHEMA_VERSION!r}"
        )
    _parse_timestamp(record["created_at"], "created_at")
    if record["sprint"] not in {"sprint1", "sprint2", "sprint3"}:
        raise RunRecordValidationError(
            "sprint must be exactly sprint1, sprint2, or sprint3"
        )
    stage = _require_nonnegative_int(record["stage"], "stage")
    if stage > 6:
        raise RunRecordValidationError("stage must be in 0..6")
    if record["run_type"] not in RUN_TYPES:
        raise RunRecordValidationError("run_type is not canonical")
    role = record["evidence_role"]
    if role not in EVIDENCE_ROLES:
        raise RunRecordValidationError("evidence_role is not canonical")
    origin = record["origin"]
    if origin not in ORIGINS:
        raise RunRecordValidationError("origin is not canonical")
    _validate_stage_role(stage=stage, role=role, run_type=record["run_type"])
    _require_sha256(
        record["protocol_config_bundle_sha256"],
        "protocol_config_bundle_sha256",
    )
    _validate_git(record["git"])
    _validate_data(record["data"])
    _validate_dataset_role_origin(
        dataset=record["data"]["dataset"],
        role=role,
        origin=origin,
        run_type=record["run_type"],
    )
    _validate_evidence_manifest_binding(
        record, authority_path=Path(evidence_authority_path)
    )
    context_mode = record["context_mode"]
    if context_mode is not None and context_mode not in {item.value for item in ContextMode}:
        raise RunRecordValidationError("context_mode is not canonical")
    if context_mode == ContextMode.WITHOUT_CONTEXT.value and (
        record["retrieval"] is not None
        or record["diversification"] is not None
        or record["data"]["corpus_manifest"] is not None
    ):
        raise RunRecordValidationError(
            "WITHOUT_CONTEXT must not carry corpus/retriever/diversifier identity"
        )
    canonical_required = origin == "PROSPECTIVE_BACKFILL" or stage in {3, 5}
    if record["retrieval"] is not None:
        _validate_retrieval(record["retrieval"], canonical_required=canonical_required)
        if record["data"]["corpus_manifest"] is None:
            raise RunRecordValidationError("retrieval requires a corpus manifest")
    if record["diversification"] is not None:
        if record["retrieval"] is None:
            raise RunRecordValidationError("diversification requires retrieval identity")
        _validate_diversification(record["diversification"])
    if record["generation"] is None:
        if context_mode is not None:
            raise RunRecordValidationError("context_mode requires generation identity")
    else:
        _validate_generation(record["generation"])
        if context_mode is None:
            raise RunRecordValidationError("generation requires context_mode")
    if record["evaluation"] is not None:
        _validate_evaluation(record["evaluation"])
    _validate_run_type_payload(record)
    status, _, _ = _validate_execution(
        record["execution"], run_type=record["run_type"]
    )
    _validate_output(record["output"], status=status)
    run_id = record["run_id"]
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise RunRecordValidationError("run_id has invalid syntax")
    resume_of = record["execution"]["resume_of"]
    if resume_of is not None and resume_of != run_id:
        raise RunRecordValidationError("execution.resume_of must equal this run_id")
    if (
        record["execution"]["status"] in {"COMPLETE", "FAILED"}
        and record["execution"]["attempt_count"] > 1
        and resume_of != run_id
    ):
        raise RunRecordValidationError(
            "terminal retried execution requires resume_of=run_id"
        )
    parent_run_id = record["execution"]["parent_run_id"]
    if parent_run_id is not None and _RUN_ID_RE.fullmatch(parent_run_id) is None:
        raise RunRecordValidationError("execution.parent_run_id has invalid syntax")
    expected_run_id = derive_run_id(record)
    if run_id != expected_run_id:
        raise RunRecordValidationError(
            f"run_id does not match immutable identity; expected {expected_run_id}"
        )
    return json.loads(canonical_json(record))


def finalize_planned_record(
    record_without_run_id: Mapping[str, Any],
    *,
    evidence_authority_path: Path = DEFAULT_EVIDENCE_AUTHORITY_PATH,
) -> dict[str, Any]:
    """Derive run_id and validate a pre-run PLANNED record."""
    record = deepcopy(dict(record_without_run_id))
    if "run_id" in record:
        raise RunRecordValidationError("record_without_run_id must omit run_id")
    record["run_id"] = derive_run_id(record)
    validated = validate_run_record(
        record, evidence_authority_path=evidence_authority_path
    )
    if validated["execution"]["status"] != "PLANNED":
        raise RunRecordValidationError("new pre-run registration must be PLANNED")
    return validated


def _parse_registry_text(
    text: str, *, evidence_authority_path: Path
) -> list[dict[str, Any]]:
    lines = text.splitlines()
    if not lines:
        return []
    if any(not line.strip() for line in lines):
        raise RunRecordValidationError("registry must not contain blank lines")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RunRecordValidationError("registry header is not valid JSON") from exc
    if header != REGISTRY_HEADER:
        raise RunRecordValidationError("registry header does not match canonical format")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunRecordValidationError(
                f"registry line {line_number} is not valid JSON"
            ) from exc
        records.append(
            validate_run_record(
                parsed, evidence_authority_path=evidence_authority_path
            )
        )
    return records


def read_registry(
    path: Path,
    *,
    evidence_authority_path: Path = DEFAULT_EVIDENCE_AUTHORITY_PATH,
) -> tuple[dict[str, Any], ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            records = _parse_registry_text(
                handle.read(), evidence_authority_path=evidence_authority_path
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return tuple(records)


def _transition_allowed(previous: str, current: str) -> bool:
    return current in {
        "PLANNED": {"RUNNING"},
        "RUNNING": {"RUNNING", "COMPLETE", "FAILED"},
        "COMPLETE": set(),
        "FAILED": set(),
    }[previous]


def _identity_diff(left: object, right: object, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_identity_diff(left[key], right[key], path))
        return paths
    return [] if left == right else [prefix]


def _validate_resume_compatibility(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    for field in ("environment_sha256", "runtime_sha256"):
        if previous["execution"][field] != current["execution"][field]:
            raise RunRegistryConflictError(
                f"retry/resume requires identical execution.{field}"
            )
    previous_attempt = previous["execution"]["attempt_count"]
    current_attempt = current["execution"]["attempt_count"]
    previous_status = previous["execution"]["status"]
    current_status = current["execution"]["status"]
    if current_attempt < previous_attempt:
        raise RunRegistryConflictError("attempt_count must be monotonic")
    if previous_status == "RUNNING" and current_status == "RUNNING":
        if current_attempt != previous_attempt + 1:
            raise RunRegistryConflictError(
                "successive RUNNING retry snapshots must increment attempt_count by one"
            )
        if current["execution"]["resume_of"] != current["run_id"]:
            raise RunRegistryConflictError(
                "successive infrastructure attempt requires resume_of=run_id"
            )
    elif previous_status == "RUNNING" and current_status in {"COMPLETE", "FAILED"}:
        if current_attempt != previous_attempt:
            raise RunRegistryConflictError(
                "terminal snapshot must retain the immediately preceding attempt_count"
            )
    elif current_attempt == 0:
        raise RunRegistryConflictError("started lifecycle requires attempt_count >= 1")


def append_run_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    evidence_authority_path: Path = DEFAULT_EVIDENCE_AUTHORITY_PATH,
) -> bool:
    """Append one valid lifecycle snapshot; return False for an exact duplicate."""
    validated = validate_run_record(
        record, evidence_authority_path=evidence_authority_path
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="\n") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            text = handle.read()
            existing = _parse_registry_text(
                text, evidence_authority_path=evidence_authority_path
            )
            same_run = [item for item in existing if item["run_id"] == validated["run_id"]]
            if same_run:
                latest = same_run[-1]
                if latest == validated:
                    return False
                latest_identity = run_identity_payload(latest)
                current_identity = run_identity_payload(validated)
                if latest_identity != current_identity:
                    differing = ", ".join(
                        _identity_diff(latest_identity, current_identity)[:5]
                    )
                    raise RunRegistryConflictError(
                        "existing run_id has conflicting scientific identity"
                        + (f": {differing}" if differing else "")
                    )
                previous_status = latest["execution"]["status"]
                current_status = validated["execution"]["status"]
                if previous_status == current_status == "PLANNED":
                    return False
                if not _transition_allowed(previous_status, current_status):
                    raise RunRegistryConflictError(
                        f"invalid lifecycle transition {previous_status}->{current_status}"
                    )
                _validate_resume_compatibility(latest, validated)
            elif validated["execution"]["status"] != "PLANNED":
                raise RunRegistryConflictError("a new run_id must begin as PLANNED")

            handle.seek(0, os.SEEK_END)
            if not text:
                handle.write(canonical_json(REGISTRY_HEADER) + "\n")
            handle.write(canonical_json(validated) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
