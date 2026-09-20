"""Governed missing-only candidate production and aggregate inventory helpers.

The helpers in this module are deliberately retrieval-neutral.  They inspect
immutable per-query CandidateArtifact files, derive an exact manifest-ordered
work scope, and build registry-compatible planning/output artifacts.  They do
not load a dataset, model, index, or retriever and never overwrite an existing
candidate artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from retrieval_artifacts.candidate_io import (
    CandidateArtifactConflictError,
    read_candidate_artifact,
)
from retrieval_artifacts.contracts import (
    CandidateArtifact,
    CorpusProvenance,
    DatasetProvenance,
    RetrieverProvenance,
    canonical_stable_id,
)
from retrieval_artifacts.producer import CorpusRecord, document_content_sha256
from retrieval_artifacts.sample_manifest import SampleManifest, verify_manifest_sample
from run_registry import (
    candidate_set_artifact_payload,
    candidate_set_scientific_payload,
    canonical_json,
    file_sha256,
    finalize_planned_record,
    output_artifact,
    output_inventory_sha256,
    stable_json_sha256,
    validate_run_record,
    write_candidate_set_artifact,
)


CANDIDATE_PRODUCTION_PLAN_SCHEMA_VERSION = (
    "sprint3.candidate-production-plan.v1"
)
CANDIDATE_PRODUCTION_PLAN_ARTIFACT_FORMAT = (
    "sprint3.candidate-production-plan-artifact.v1"
)
CANDIDATE_PRODUCTION_OUTPUT_SCHEMA_VERSION = (
    "sprint3.candidate-production-output-inventory.v1"
)
CANDIDATE_PRODUCTION_OUTPUT_ARTIFACT_FORMAT = (
    "sprint3.candidate-production-output-inventory-artifact.v1"
)

class CandidateArtifactState(str, Enum):
    VALID_EXISTING = "VALID_EXISTING"
    MISSING = "MISSING"
    INVALID_EXISTING = "INVALID_EXISTING"


class CandidateProductionValidationError(CandidateArtifactConflictError):
    """Existing candidate state is unsafe for missing-only execution."""

    def __init__(self, message: str, *, inspection: "CandidateDirectoryInspection"):
        super().__init__(message)
        self.inspection = inspection


@dataclass(frozen=True)
class CandidateSampleInspection:
    position: int
    sample_id: str | int
    path: Path
    state: CandidateArtifactState
    artifact_id: str | None = None
    byte_sha256: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class CandidateDirectoryInspection:
    sample_manifest_id: str
    candidate_directory: Path
    entries: tuple[CandidateSampleInspection, ...]
    extra_artifact_paths: tuple[Path, ...]

    @property
    def valid_entries(self) -> tuple[CandidateSampleInspection, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.state is CandidateArtifactState.VALID_EXISTING
        )

    @property
    def missing_entries(self) -> tuple[CandidateSampleInspection, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.state is CandidateArtifactState.MISSING
        )

    @property
    def invalid_entries(self) -> tuple[CandidateSampleInspection, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.state is CandidateArtifactState.INVALID_EXISTING
        )

    def require_runnable(self) -> None:
        if self.invalid_entries or self.extra_artifact_paths:
            invalid_ids = [entry.sample_id for entry in self.invalid_entries]
            extras = [path.name for path in self.extra_artifact_paths]
            raise CandidateProductionValidationError(
                "existing candidate artifact conflicts with run; "
                "candidate directory is not safe for missing-only execution; "
                f"invalid_sample_ids={invalid_ids}, extra_artifacts={extras}",
                inspection=self,
            )


@dataclass(frozen=True)
class CandidateProductionPlan:
    inspection: CandidateDirectoryInspection
    requested_sample_ids: tuple[str | int, ...]
    scheduled_sample_ids: tuple[str | int, ...]
    skipped_valid_sample_ids: tuple[str | int, ...]


def _stable_id_key(value: object) -> str:
    return canonical_json(canonical_stable_id(value, "sample_id"))


def _stable_ids_equal(left: object, right: object) -> bool:
    return _stable_id_key(left) == _stable_id_key(right)


def _candidate_path(candidate_directory: Path, position: int) -> Path:
    return Path(candidate_directory) / f"sample_{position:04d}.json"


def select_manifest_queries(
    ordered_queries: Sequence[object],
    *,
    requested_sample_ids: Sequence[str | int] | None = None,
    max_samples: int | None = None,
) -> tuple[object, ...]:
    """Select exact IDs while retaining canonical sample-manifest order."""
    queries = tuple(ordered_queries)
    if requested_sample_ids is not None and max_samples is not None:
        raise ValueError("requested_sample_ids and max_samples are mutually exclusive")
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError("max_samples must be a non-boolean integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        return queries[:max_samples]
    if requested_sample_ids is None:
        return queries

    requested = tuple(requested_sample_ids)
    requested_keys = [_stable_id_key(sample_id) for sample_id in requested]
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError("requested sample IDs must not contain duplicates")
    query_by_key = {
        _stable_id_key(query.sample_id): query
        for query in queries
    }
    unknown = [
        sample_id
        for sample_id, key in zip(requested, requested_keys, strict=True)
        if key not in query_by_key
    ]
    if unknown:
        raise ValueError(f"requested sample IDs are absent from manifest: {unknown}")
    requested_key_set = set(requested_keys)
    return tuple(
        query
        for query in queries
        if _stable_id_key(query.sample_id) in requested_key_set
    )


def _validate_candidate_artifact(
    artifact: CandidateArtifact,
    *,
    query: object,
    dataset_provenance: DatasetProvenance,
    corpus_provenance: CorpusProvenance,
    retriever_provenance: RetrieverProvenance,
    corpus_records: tuple[CorpusRecord, ...],
    candidate_pool: int,
) -> None:
    expected_identity = (
        _stable_ids_equal(artifact.sample_id, query.sample_id),
        artifact.query_text == query.query_text,
        artifact.dataset == dataset_provenance,
        artifact.corpus == corpus_provenance,
        artifact.retriever == retriever_provenance,
        artifact.requested_top_n == candidate_pool,
        len(artifact.candidates) == candidate_pool,
    )
    if not all(expected_identity):
        raise ValueError("candidate artifact scientific identity mismatch")

    for entry in artifact.candidates:
        if entry.corpus_position is None:
            raise ValueError("candidate corpus_position is required")
        if entry.corpus_position >= len(corpus_records):
            raise ValueError("candidate corpus_position is outside the corpus")
        record = corpus_records[entry.corpus_position]
        if not _stable_ids_equal(entry.document_id, record.document_id):
            raise ValueError("candidate document_id does not match corpus position")
        if not _stable_ids_equal(
            entry.source_document_id, record.source_document_id
        ):
            raise ValueError(
                "candidate source_document_id does not match corpus position"
            )
        if entry.document_content_sha256 != document_content_sha256(
            record.retrieval_content
        ):
            raise ValueError("candidate document content hash mismatch")


def inspect_candidate_directory(
    *,
    sample_manifest: SampleManifest,
    ordered_queries: Sequence[object],
    candidate_directory: Path,
    dataset_provenance: DatasetProvenance,
    corpus_provenance: CorpusProvenance,
    retriever_provenance: RetrieverProvenance,
    corpus_records: tuple[CorpusRecord, ...],
    candidate_pool: int,
) -> CandidateDirectoryInspection:
    """Classify every manifest sample without writing or repairing artifacts."""
    queries = tuple(ordered_queries)
    if len(queries) != sample_manifest.actual_sample_size:
        raise ValueError("ordered queries do not cover the SampleManifest")
    if isinstance(candidate_pool, bool) or not isinstance(candidate_pool, int):
        raise TypeError("candidate_pool must be a non-boolean integer")
    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be positive")

    expected_paths: set[Path] = set()
    inspected: list[CandidateSampleInspection] = []
    candidate_directory = Path(candidate_directory)
    for manifest_entry, query in zip(
        sample_manifest.entries, queries, strict=True
    ):
        if (
            manifest_entry.position != query.position
            or not _stable_ids_equal(manifest_entry.sample_id, query.sample_id)
        ):
            raise ValueError("ordered query identity does not match SampleManifest")
        verify_manifest_sample(
            sample_manifest,
            sample_id=query.sample_id,
            query_text=query.query_text,
        )
        path = _candidate_path(candidate_directory, manifest_entry.position)
        expected_paths.add(path)
        if not path.exists():
            inspected.append(
                CandidateSampleInspection(
                    position=manifest_entry.position,
                    sample_id=manifest_entry.sample_id,
                    path=path,
                    state=CandidateArtifactState.MISSING,
                )
            )
            continue
        try:
            if not path.is_file():
                raise ValueError("candidate artifact path is not a regular file")
            artifact = read_candidate_artifact(path)
            _validate_candidate_artifact(
                artifact,
                query=query,
                dataset_provenance=dataset_provenance,
                corpus_provenance=corpus_provenance,
                retriever_provenance=retriever_provenance,
                corpus_records=corpus_records,
                candidate_pool=candidate_pool,
            )
            inspected.append(
                CandidateSampleInspection(
                    position=manifest_entry.position,
                    sample_id=manifest_entry.sample_id,
                    path=path,
                    state=CandidateArtifactState.VALID_EXISTING,
                    artifact_id=artifact.artifact_id,
                    byte_sha256=file_sha256(path),
                )
            )
        except Exception as error:
            inspected.append(
                CandidateSampleInspection(
                    position=manifest_entry.position,
                    sample_id=manifest_entry.sample_id,
                    path=path,
                    state=CandidateArtifactState.INVALID_EXISTING,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

    extras = ()
    if candidate_directory.is_dir():
        extras = tuple(
            sorted(
                (
                    path
                    for path in candidate_directory.iterdir()
                    if path.is_file()
                    and path.name.startswith("sample_")
                    and path.suffix == ".json"
                    and path not in expected_paths
                ),
                key=lambda path: path.name,
            )
        )
    return CandidateDirectoryInspection(
        sample_manifest_id=sample_manifest.manifest_id,
        candidate_directory=candidate_directory,
        entries=tuple(inspected),
        extra_artifact_paths=extras,
    )


def plan_candidate_production(
    *,
    sample_manifest: SampleManifest,
    ordered_queries: Sequence[object],
    candidate_directory: Path,
    dataset_provenance: DatasetProvenance,
    corpus_provenance: CorpusProvenance,
    retriever_provenance: RetrieverProvenance,
    corpus_records: tuple[CorpusRecord, ...],
    candidate_pool: int,
    requested_sample_ids: Sequence[str | int] | None = None,
    max_samples: int | None = None,
) -> CandidateProductionPlan:
    """Return only missing work; fail closed on any invalid or extra artifact."""
    queries = tuple(ordered_queries)
    selected = select_manifest_queries(
        queries,
        requested_sample_ids=requested_sample_ids,
        max_samples=max_samples,
    )
    inspection = inspect_candidate_directory(
        sample_manifest=sample_manifest,
        ordered_queries=queries,
        candidate_directory=candidate_directory,
        dataset_provenance=dataset_provenance,
        corpus_provenance=corpus_provenance,
        retriever_provenance=retriever_provenance,
        corpus_records=corpus_records,
        candidate_pool=candidate_pool,
    )
    inspection.require_runnable()
    state_by_id = {
        _stable_id_key(entry.sample_id): entry.state
        for entry in inspection.entries
    }
    requested = tuple(query.sample_id for query in selected)
    scheduled = tuple(
        query.sample_id
        for query in selected
        if state_by_id[_stable_id_key(query.sample_id)] is CandidateArtifactState.MISSING
    )
    skipped = tuple(
        query.sample_id
        for query in selected
        if state_by_id[_stable_id_key(query.sample_id)]
        is CandidateArtifactState.VALID_EXISTING
    )
    return CandidateProductionPlan(
        inspection=inspection,
        requested_sample_ids=requested,
        scheduled_sample_ids=scheduled,
        skipped_valid_sample_ids=skipped,
    )


def candidate_production_plan_scientific_payload(
    *,
    dataset: str,
    evidence_role: str,
    sample_manifest_id: str,
    corpus_manifest_id: str,
    retriever: str,
    retriever_config_sha256: str,
    index_artifact_id: str,
    candidate_pool: int,
    top_k: int,
    candidate_directory: str,
    scheduled_sample_ids: Sequence[str | int],
) -> dict[str, Any]:
    sample_ids = [
        canonical_stable_id(value, "scheduled_sample_id")
        for value in scheduled_sample_ids
    ]
    keys = [_stable_id_key(value) for value in sample_ids]
    if not sample_ids:
        raise ValueError("candidate production plan requires missing sample IDs")
    if len(set(keys)) != len(keys):
        raise ValueError("scheduled sample IDs must be unique")
    payload = {
        "schema_version": CANDIDATE_PRODUCTION_PLAN_SCHEMA_VERSION,
        "dataset": dataset,
        "evidence_role": evidence_role,
        "sample_manifest_id": sample_manifest_id,
        "corpus_manifest_id": corpus_manifest_id,
        "retriever": retriever,
        "retriever_config_sha256": retriever_config_sha256,
        "index_artifact_id": index_artifact_id,
        "candidate_pool": candidate_pool,
        "top_k": top_k,
        "candidate_directory": candidate_directory,
        "scheduled_sample_ids": sample_ids,
    }
    canonical_json(payload)
    return payload


def candidate_production_plan_artifact(
    scientific_payload: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    payload = dict(scientific_payload)
    scientific_sha256 = stable_json_sha256(payload)
    return {
        "artifact_format": CANDIDATE_PRODUCTION_PLAN_ARTIFACT_FORMAT,
        "run_id": run_id,
        "scientific_sha256": scientific_sha256,
        "scientific_payload": payload,
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one immutable canonical-JSON artifact; never replace differing bytes."""
    path = Path(path)
    serialized = canonical_json(payload) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"existing immutable artifact differs: {path}")
        return
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
            raise FileExistsError(f"existing immutable artifact differs: {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def write_candidate_production_plan(
    path: Path, scientific_payload: Mapping[str, Any], *, run_id: str
) -> None:
    _write_new_json(
        path,
        candidate_production_plan_artifact(scientific_payload, run_id=run_id),
    )


def read_candidate_production_plan(path: Path) -> dict[str, Any]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping):
        raise ValueError("candidate production plan artifact must be an object")
    required = {
        "artifact_format",
        "run_id",
        "scientific_sha256",
        "scientific_payload",
    }
    if set(stored) != required:
        raise ValueError("candidate production plan artifact keys mismatch")
    if stored["artifact_format"] != CANDIDATE_PRODUCTION_PLAN_ARTIFACT_FORMAT:
        raise ValueError("candidate production plan artifact format mismatch")
    payload = stored["scientific_payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("candidate production plan scientific payload is invalid")
    if stable_json_sha256(payload) != stored["scientific_sha256"]:
        raise ValueError("candidate production plan scientific hash mismatch")
    canonical_json(stored)
    return dict(stored)


def build_retrieval_planned_record(
    *,
    created_at: str,
    plan_payload: Mapping[str, Any],
    git: Mapping[str, Any],
    data: Mapping[str, Any],
    retrieval_index: Mapping[str, Any],
    environment_sha256: str,
    runtime_sha256: str,
    hardware_summary: str,
    output_directory: str,
    evidence_authority_path: Path,
) -> dict[str, Any]:
    """Build the canonical Stage-1 PubMedQA missing-subset RETRIEVAL record."""
    scheduled = plan_payload["scheduled_sample_ids"]
    record = {
        "schema_version": "sprint3.run-registry-record.v1",
        "created_at": created_at,
        "sprint": "sprint1",
        "stage": 1,
        "run_type": "RETRIEVAL",
        "evidence_role": "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        "origin": "PROSPECTIVE_BACKFILL",
        "protocol_config_bundle_sha256": stable_json_sha256(plan_payload),
        "git": dict(git),
        "data": dict(data),
        "retrieval": {
            "retriever": plan_payload["retriever"],
            "config_sha256": plan_payload["retriever_config_sha256"],
            "index": dict(retrieval_index),
            "candidate_set": None,
            "selected_context": None,
            "candidate_pool": plan_payload["candidate_pool"],
            "top_k": plan_payload["top_k"],
        },
        "diversification": None,
        "context_mode": None,
        "generation": None,
        "evaluation": None,
        "execution": {
            "environment_sha256": environment_sha256,
            "runtime_sha256": runtime_sha256,
            "hardware_summary": hardware_summary,
            "started_at": None,
            "completed_at": None,
            "status": "PLANNED",
            "attempt_count": 0,
            "failure_reason": None,
            "parent_run_id": None,
            "resume_of": None,
        },
        "output": {
            "expected_row_count": len(scheduled),
            "completed_row_count": 0,
            "successful_row_count": 0,
            "failed_row_count": 0,
            "output_directory": output_directory,
            "partial_output_retained": False,
            "artifacts": [],
            "output_inventory_sha256": None,
            "raw_artifact_sha256": None,
        },
    }
    return finalize_planned_record(
        record, evidence_authority_path=evidence_authority_path
    )


def running_candidate_record(
    record: Mapping[str, Any],
    *,
    started_at: str,
    attempt_count: int,
    environment_sha256: str,
    runtime_sha256: str,
    hardware_summary: str,
    prior_failure_reason: str | None = None,
    evidence_authority_path: Path,
) -> dict[str, Any]:
    updated = deepcopy(dict(record))
    updated["execution"].update(
        {
            "started_at": started_at,
            "completed_at": None,
            "status": "RUNNING",
            "attempt_count": attempt_count,
            "failure_reason": prior_failure_reason,
            "resume_of": updated["run_id"] if attempt_count > 1 else None,
            "environment_sha256": environment_sha256,
            "runtime_sha256": runtime_sha256,
            "hardware_summary": hardware_summary,
        }
    )
    return validate_run_record(
        updated, evidence_authority_path=evidence_authority_path
    )


def candidate_production_output_inventory(
    *,
    run_id: str,
    producer_identity: str,
    retriever: str,
    repository_root: Path,
    scheduled_sample_ids: Sequence[str | int],
    initial_preexisting_entries: Sequence[CandidateSampleInspection],
    current_inspection: CandidateDirectoryInspection,
    failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Inventory run-scope outputs without relabelling pre-existing artifacts."""
    scheduled = tuple(
        canonical_stable_id(value, "scheduled_sample_id")
        for value in scheduled_sample_ids
    )
    scheduled_keys = {_stable_id_key(value) for value in scheduled}
    failure_by_key: dict[str, dict[str, Any]] = {}
    for raw in failures:
        failure = dict(raw)
        if set(failure) != {"sample_id", "error_type", "message"}:
            raise ValueError("candidate failure keys mismatch")
        key = _stable_id_key(failure["sample_id"])
        if key not in scheduled_keys:
            raise ValueError("candidate failure is outside scheduled run scope")
        if key in failure_by_key:
            raise ValueError("candidate failures contain duplicate sample IDs")
        failure["sample_id"] = canonical_stable_id(
            failure["sample_id"], "failure sample_id"
        )
        failure_by_key[key] = failure

    output_entries: list[dict[str, Any]] = []
    remaining: list[str | int] = []
    status_counts = {"VALID_OUTPUT": 0, "FAILED": 0, "MISSING": 0}
    for entry in current_inspection.entries:
        key = _stable_id_key(entry.sample_id)
        if key not in scheduled_keys:
            continue
        if entry.state is CandidateArtifactState.VALID_EXISTING:
            if key in failure_by_key:
                raise ValueError(
                    "candidate failure conflicts with a valid output artifact"
                )
            relative = entry.path.resolve().relative_to(
                Path(repository_root).resolve()
            )
            output_entries.append(
                {
                    "sample_id": entry.sample_id,
                    "path": relative.as_posix(),
                    "candidate_artifact_id": entry.artifact_id,
                    "byte_sha256": entry.byte_sha256,
                }
            )
            status_counts["VALID_OUTPUT"] += 1
        elif key in failure_by_key:
            status_counts["FAILED"] += 1
            remaining.append(entry.sample_id)
        else:
            status_counts["MISSING"] += 1
            remaining.append(entry.sample_id)

    preexisting = [
        {
            "sample_id": entry.sample_id,
            "candidate_artifact_id": entry.artifact_id,
            "byte_sha256": entry.byte_sha256,
        }
        for entry in initial_preexisting_entries
    ]
    payload = {
        "schema_version": CANDIDATE_PRODUCTION_OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "producer_identity": producer_identity,
        "retriever": retriever,
        "requested_sample_count": len(scheduled),
        "newly_produced_artifact_count": len(output_entries),
        "reused_preexisting_valid_count": len(preexisting),
        "remaining_missing_count": status_counts["MISSING"],
        "unresolved_count": len(remaining),
        "failed_count": len(failure_by_key),
        "status_counts": status_counts,
        "scheduled_sample_ids": list(scheduled),
        "output_artifacts": output_entries,
        "preserved_preexisting_artifacts": preexisting,
        "remaining_sample_ids": remaining,
        "failures": list(failure_by_key.values()),
    }
    canonical_json(payload)
    return payload


def candidate_production_output_artifact(
    payload: Mapping[str, Any]
) -> dict[str, Any]:
    scientific_payload = dict(payload)
    sha256 = stable_json_sha256(scientific_payload)
    return {
        "artifact_format": CANDIDATE_PRODUCTION_OUTPUT_ARTIFACT_FORMAT,
        "artifact_id": f"candidate-run-output:sha256:{sha256}",
        "scientific_sha256": sha256,
        "scientific_payload": scientific_payload,
    }


def write_candidate_production_output(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_json(path, candidate_production_output_artifact(payload))


def materialize_candidate_set_inventory(
    *,
    sample_manifest: SampleManifest,
    ordered_queries: Sequence[object],
    candidate_directory: Path,
    dataset_provenance: DatasetProvenance,
    corpus_provenance: CorpusProvenance,
    retriever_provenance: RetrieverProvenance,
    corpus_records: tuple[CorpusRecord, ...],
    candidate_pool: int,
    evidence_role: str,
    retriever: str,
    output_path: Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a candidate-set identity only for a complete valid manifest."""
    inspection = inspect_candidate_directory(
        sample_manifest=sample_manifest,
        ordered_queries=ordered_queries,
        candidate_directory=candidate_directory,
        dataset_provenance=dataset_provenance,
        corpus_provenance=corpus_provenance,
        retriever_provenance=retriever_provenance,
        corpus_records=corpus_records,
        candidate_pool=candidate_pool,
    )
    inspection.require_runnable()
    if inspection.missing_entries:
        raise CandidateProductionValidationError(
            "candidate set is incomplete; missing_sample_ids="
            f"{[entry.sample_id for entry in inspection.missing_entries]}",
            inspection=inspection,
        )
    scientific = candidate_set_scientific_payload(
        dataset=dataset_provenance.dataset_id.value,
        evidence_role=evidence_role,
        sample_manifest_id=sample_manifest.manifest_id,
        retriever=retriever,
        expected_query_count=sample_manifest.actual_sample_size,
        entries=[
            {
                "sample_id": entry.sample_id,
                "candidate_artifact_id": entry.artifact_id,
            }
            for entry in inspection.valid_entries
        ],
    )
    wrapper = candidate_set_artifact_payload(scientific, provenance=provenance)
    if output_path is not None:
        write_candidate_set_artifact(
            output_path, scientific, provenance=provenance
        )
    return wrapper


def terminal_candidate_record(
    record: Mapping[str, Any],
    *,
    completed_at: str,
    output_inventory_path: Path,
    repository_root: Path,
    successful_count: int,
    failed_count: int,
    status_counts: Mapping[str, int],
    candidate_set_path: Path | None,
    failure_reason: str | None,
    evidence_authority_path: Path,
) -> dict[str, Any]:
    """Build COMPLETE only for a resolved scope; otherwise build FAILED."""
    updated = deepcopy(dict(record))
    expected = updated["output"]["expected_row_count"]
    completed = successful_count + failed_count
    complete = successful_count == expected and failed_count == 0
    if complete != (candidate_set_path is not None):
        raise ValueError(
            "candidate_set_path is required exactly when the scheduled scope is complete"
        )
    if complete and failure_reason is not None:
        raise ValueError("complete candidate production must not carry failure_reason")
    if not complete and not failure_reason:
        raise ValueError("failed candidate production requires failure_reason")

    inventory_reference = output_artifact(
        output_inventory_path,
        repository_root=repository_root,
        row_count=expected,
        status_counts=status_counts,
        artifact_id=json.loads(
            output_inventory_path.read_text(encoding="utf-8")
        )["artifact_id"],
    )
    artifacts = [inventory_reference]
    raw_sha256 = inventory_reference["sha256"]
    if candidate_set_path is not None:
        candidate_wrapper = json.loads(candidate_set_path.read_text(encoding="utf-8"))
        aggregate_count = candidate_wrapper["scientific_payload"][
            "expected_query_count"
        ]
        candidate_reference = output_artifact(
            candidate_set_path,
            repository_root=repository_root,
            row_count=aggregate_count,
            status_counts={"VALID": aggregate_count},
            artifact_id=candidate_wrapper["candidate_set_id"],
        )
        artifacts.append(candidate_reference)
        raw_sha256 = candidate_reference["sha256"]

    updated["execution"].update(
        {
            "completed_at": completed_at,
            "status": "COMPLETE" if complete else "FAILED",
            "failure_reason": None if complete else failure_reason,
        }
    )
    if updated["execution"]["attempt_count"] > 1:
        updated["execution"]["resume_of"] = updated["run_id"]
    retained = not complete and completed > 0
    updated["output"].update(
        {
            "completed_row_count": completed,
            "successful_row_count": successful_count,
            "failed_row_count": failed_count,
            "partial_output_retained": retained,
            "artifacts": artifacts if (complete or retained) else [],
            "output_inventory_sha256": (
                output_inventory_sha256(artifacts)
                if (complete or retained)
                else None
            ),
            "raw_artifact_sha256": raw_sha256 if (complete or retained) else None,
        }
    )
    return validate_run_record(
        updated, evidence_authority_path=evidence_authority_path
    )
