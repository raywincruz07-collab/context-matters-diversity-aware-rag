#!/usr/bin/env python3
"""Run validated PubMedQA Contriever candidate production.

Importing this module performs no dataset access, model loading, indexing,
retrieval, cache writes, or CandidateArtifact writes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Callable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrievers.contriever_config import CONTRIEVER_CONFIG
from retrieval_artifacts import (
    CandidateArtifact,
    CandidateArtifactConflictError,
    RawCandidateResult,
    RetrieverProvenance,
    build_contriever_retriever_provenance,
    produce_contriever_candidate_artifact,
    build_contriever_cache_identity,
    validate_contriever_index_binding,
    write_candidate_artifact,
)
from retrieval_artifacts.candidate_production import (
    CandidateProductionPlan,
    build_retrieval_planned_record,
    candidate_production_output_inventory,
    candidate_production_plan_scientific_payload,
    inspect_candidate_directory,
    materialize_candidate_set_inventory,
    plan_candidate_production,
    read_candidate_production_plan,
    running_candidate_record,
    select_manifest_queries,
    terminal_candidate_record,
    write_candidate_production_output,
    write_candidate_production_plan,
)
from run_registry import (
    DEFAULT_EVIDENCE_AUTHORITY_PATH,
    DEFAULT_REGISTRY_PATH,
    append_run_record,
    artifact_ref,
    canonical_json,
    file_sha256,
    read_registry,
    stable_json_sha256,
)
from config import EMBEDDINGS_DIR, INDEX_DIR
from scripts.build_corpus_manifests import (
    PUBMEDQA_CONFIG,
    PUBMEDQA_REVISION,
    PUBMEDQA_SOURCE,
    PUBMEDQA_SPLIT,
    PUBMEDQA_OUTPUT_PATH,
    PUBMEDQA_SAMPLE_MANIFEST_PATH,
    ValidatedPubMedQARuntimeCorpus,
    load_validated_pubmedqa_runtime_corpus,
)

if TYPE_CHECKING:
    from retrievers.contriever_retriever import ContrieverRetriever


CANDIDATE_POOL_SIZE = 20
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "artifacts/candidates/pubmedqa/contriever"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTRIEVER_CACHE_METADATA_SCHEMA_VERSION = (
    "sprint3.contriever-cache-metadata.v1"
)
DEFAULT_CANDIDATE_SET_PATH = (
    REPOSITORY_ROOT
    / "artifacts/candidates/sets/pubmedqa/contriever_candidate_set_v1.json"
)
DEFAULT_RUN_ARTIFACT_ROOT = (
    REPOSITORY_ROOT / "artifacts/candidates/run_registry"
)


@dataclass(frozen=True)
class ContrieverRetrievalFailure:
    sample_id: str | int
    error_type: str
    message: str


@dataclass(frozen=True)
class ContrieverCandidateRunSummary:
    manifest_sample_count: int
    selected_sample_count: int
    completed_samples: int
    skipped_existing_samples: int
    failed_samples: int
    candidate_pool_size: int
    sample_manifest_id: str
    corpus_manifest_id: str
    retriever_index_fingerprint_sha256: str
    index_artifact_sha256: str
    embedding_artifact_sha256: str
    completed_sample_ids: tuple[str | int, ...]
    skipped_sample_ids: tuple[str | int, ...]
    failures: tuple[ContrieverRetrievalFailure, ...]


@dataclass(frozen=True)
class LocalContrieverIndexIdentity:
    cache_fingerprint_sha256: str
    embedding_path: Path
    embedding_artifact_sha256: str
    index_path: Path
    index_artifact_sha256: str
    metadata_path: Path
    retriever_provenance: RetrieverProvenance
    cache_environment: Mapping[str, object]

    def registry_index_reference(self) -> dict[str, object]:
        return artifact_ref(
            self.index_path,
            repository_root=REPOSITORY_ROOT,
            artifact_id=f"index:sha256:{self.cache_fingerprint_sha256}",
        )


class ContrieverCandidatePoolSizeError(ValueError):
    """Contriever returned a non-canonical PubMedQA candidate pool size."""


@dataclass(frozen=True)
class GovernedContrieverRunSummary:
    run_id: str
    status: str
    attempt_count: int
    candidate_summary: ContrieverCandidateRunSummary
    output_inventory_path: Path
    candidate_set_path: Path | None


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def validate_local_contriever_index_identity(
    runtime: ValidatedPubMedQARuntimeCorpus,
    *,
    transformers_version: str,
) -> LocalContrieverIndexIdentity:
    """Validate the reusable local cache from metadata and physical hashes only."""
    cache_identity = build_contriever_cache_identity(
        corpus_manifest=runtime.corpus_manifest,
        contriever_config=CONTRIEVER_CONFIG,
    )
    fingerprint = cache_identity.fingerprint_sha256
    embedding_path = Path(EMBEDDINGS_DIR) / cache_identity.embedding_cache_filename
    index_path = Path(INDEX_DIR) / cache_identity.faiss_cache_filename
    metadata_path = Path(INDEX_DIR) / f"contriever_cache_{fingerprint}.json"
    for name, path in (
        ("embedding", embedding_path),
        ("FAISS index", index_path),
        ("cache metadata", metadata_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"local Contriever {name} is missing: {path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Contriever cache metadata must be an object")
    expected_binding = {
        "cache_fingerprint_sha256": fingerprint,
        "cache_identity_schema_version": cache_identity.schema_version,
        "cache_schema_version": CONTRIEVER_CACHE_METADATA_SCHEMA_VERSION,
        "corpus_manifest_id": runtime.corpus_manifest.corpus_manifest_id,
        "corpus_manifest_sha256": runtime.corpus_manifest.sha256,
        "contriever_scientific_json": CONTRIEVER_CONFIG.scientific_json(),
        "scientific_payload": cache_identity.scientific_payload(),
    }
    for key, expected in expected_binding.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Contriever cache metadata {key} mismatch")

    embedding = metadata.get("embedding")
    faiss = metadata.get("faiss")
    cache_environment = metadata.get("environment")
    if not isinstance(embedding, dict) or not isinstance(faiss, dict):
        raise ValueError("Contriever cache artifact metadata is incomplete")
    if not isinstance(cache_environment, dict):
        raise ValueError("Contriever cache environment metadata is missing")
    expected_embedding = {
        "document_count": runtime.corpus_manifest.document_count,
        "dtype": CONTRIEVER_CONFIG.embedding_dtype,
        "embedding_dimension": CONTRIEVER_CONFIG.embedding_dimension,
        "filename": embedding_path.name,
        "sha256": embedding.get("sha256"),
        "shape": [
            runtime.corpus_manifest.document_count,
            CONTRIEVER_CONFIG.embedding_dimension,
        ],
    }
    expected_faiss = {
        "dimension": CONTRIEVER_CONFIG.embedding_dimension,
        "filename": index_path.name,
        "index_type": CONTRIEVER_CONFIG.index_type,
        "ntotal": runtime.corpus_manifest.document_count,
        "sha256": faiss.get("sha256"),
    }
    if embedding != expected_embedding:
        raise ValueError("Contriever embedding metadata mismatch")
    if faiss != expected_faiss:
        raise ValueError("Contriever FAISS metadata mismatch")
    embedding_sha = _require_sha256(
        embedding.get("sha256"), "embedding metadata SHA-256"
    )
    index_sha = _require_sha256(
        faiss.get("sha256"), "FAISS metadata SHA-256"
    )
    if file_sha256(embedding_path) != embedding_sha:
        raise ValueError("Contriever embedding physical SHA-256 mismatch")
    if file_sha256(index_path) != index_sha:
        raise ValueError("Contriever FAISS physical SHA-256 mismatch")

    provenance = build_contriever_retriever_provenance(
        cache_identity=cache_identity,
        index_artifact_sha256=index_sha,
        transformers_version=transformers_version,
        contriever_config=CONTRIEVER_CONFIG,
    )
    validate_contriever_index_binding(
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        cache_identity=cache_identity,
        contriever_config=CONTRIEVER_CONFIG,
    )
    return LocalContrieverIndexIdentity(
        cache_fingerprint_sha256=fingerprint,
        embedding_path=embedding_path,
        embedding_artifact_sha256=embedding_sha,
        index_path=index_path,
        index_artifact_sha256=index_sha,
        metadata_path=metadata_path,
        retriever_provenance=provenance,
        cache_environment=dict(cache_environment),
    )


def build_runtime_contriever_provenance(
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever: ContrieverRetriever,
    *,
    transformers_version: str,
) -> RetrieverProvenance:
    """Build provenance from a completely initialized Contriever index."""
    if retriever.is_indexed is not True:
        raise RuntimeError("Contriever retriever must be indexed before provenance")
    if retriever.cache_identity is None:
        raise RuntimeError("indexed Contriever retriever is missing cache_identity")
    index_sha = _require_sha256(
        retriever.index_artifact_sha256, "index_artifact_sha256"
    )
    _require_sha256(
        retriever.embedding_artifact_sha256, "embedding_artifact_sha256"
    )
    provenance = build_contriever_retriever_provenance(
        cache_identity=retriever.cache_identity,
        index_artifact_sha256=index_sha,
        transformers_version=transformers_version,
        contriever_config=CONTRIEVER_CONFIG,
    )
    validate_contriever_index_binding(
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        cache_identity=retriever.cache_identity,
        contriever_config=CONTRIEVER_CONFIG,
    )
    return provenance


def _candidate_path(output_dir: Path, query) -> Path:
    return output_dir / f"sample_{query.position:04d}.json"


def run_pubmedqa_contriever_candidates(
    *,
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever: ContrieverRetriever,
    output_dir: Path,
    producing_git_commit: str,
    worktree_clean: bool,
    environment_fingerprint_sha256: str,
    transformers_version: str,
    max_samples: int | None = None,
    sample_ids: tuple[str | int, ...] | None = None,
    expected_retriever_provenance: RetrieverProvenance | None = None,
    dry_run: bool = False,
    producer: Callable[..., CandidateArtifact] = (
        produce_contriever_candidate_artifact
    ),
) -> ContrieverCandidateRunSummary:
    """Process exact missing IDs in manifest order without rewriting valid files."""
    queries = runtime.ordered_queries
    selected_queries = select_manifest_queries(
        queries,
        requested_sample_ids=sample_ids,
        max_samples=max_samples,
    )

    if retriever.config != CONTRIEVER_CONFIG:
        raise ValueError(
            "Contriever retriever runtime config does not match frozen "
            "CONTRIEVER_CONFIG"
        )
    pre_index_plan: CandidateProductionPlan | None = None
    if expected_retriever_provenance is not None:
        pre_index_plan = plan_candidate_production(
            sample_manifest=runtime.sample_manifest,
            ordered_queries=queries,
            candidate_directory=output_dir,
            dataset_provenance=runtime.dataset_provenance,
            corpus_provenance=runtime.corpus_provenance,
            retriever_provenance=expected_retriever_provenance,
            corpus_records=runtime.corpus_records,
            candidate_pool=CANDIDATE_POOL_SIZE,
            requested_sample_ids=tuple(
                query.sample_id for query in selected_queries
            ),
        )
        selected_queries = select_manifest_queries(
            queries,
            requested_sample_ids=pre_index_plan.scheduled_sample_ids,
        )

    retriever.index_from_corpus_records(
        corpus_manifest=runtime.corpus_manifest,
        corpus_records=runtime.corpus_records,
    )
    retriever_provenance = build_runtime_contriever_provenance(
        runtime,
        retriever,
        transformers_version=transformers_version,
    )
    if (
        expected_retriever_provenance is not None
        and retriever_provenance != expected_retriever_provenance
    ):
        raise CandidateArtifactConflictError(
            "initialized Contriever index identity differs from preflight identity"
        )
    plan = pre_index_plan or plan_candidate_production(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=queries,
        candidate_directory=output_dir,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=retriever_provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=CANDIDATE_POOL_SIZE,
        requested_sample_ids=tuple(query.sample_id for query in selected_queries),
    )
    if pre_index_plan is None:
        selected_queries = select_manifest_queries(
            queries,
            requested_sample_ids=plan.scheduled_sample_ids,
        )

    completed: list[str | int] = []
    skipped: list[str | int] = list(plan.skipped_valid_sample_ids)
    failures: list[ContrieverRetrievalFailure] = []
    for query in selected_queries:
        path = _candidate_path(output_dir, query)
        if path.exists():
            raise CandidateArtifactConflictError(
                "scheduled missing candidate path appeared after planning: "
                f"sample {query.sample_id!r}"
            )
        try:
            retrieved = retriever.retrieve(
                query.query_text,
                top_k=CANDIDATE_POOL_SIZE,
            )
            if len(retrieved) != CANDIDATE_POOL_SIZE:
                raise ContrieverCandidatePoolSizeError(
                    f"expected candidate count = {CANDIDATE_POOL_SIZE}; "
                    f"actual candidate count = {len(retrieved)}"
                )
            raw_results = tuple(
                RawCandidateResult(
                    document_id=document["doc_id"],
                    native_score=score,
                )
                for document, score in retrieved
            )
            artifact = producer(
                sample_manifest=runtime.sample_manifest,
                dataset_provenance=runtime.dataset_provenance,
                corpus_manifest=runtime.corpus_manifest,
                corpus_provenance=runtime.corpus_provenance,
                cache_identity=retriever.cache_identity,
                sample_id=query.sample_id,
                query_text=query.query_text,
                retriever_provenance=retriever_provenance,
                requested_top_n=CANDIDATE_POOL_SIZE,
                raw_results=raw_results,
                corpus_records=runtime.corpus_records,
                producing_git_commit=producing_git_commit,
                worktree_clean=worktree_clean,
                environment_fingerprint_sha256=environment_fingerprint_sha256,
                contriever_config=CONTRIEVER_CONFIG,
            )
            if len(artifact.candidates) != CANDIDATE_POOL_SIZE:
                raise ContrieverCandidatePoolSizeError(
                    "Contriever producer returned a non-canonical candidate count"
                )
            if not dry_run:
                write_candidate_artifact(artifact, path)
            completed.append(query.sample_id)
        except CandidateArtifactConflictError:
            raise
        except Exception as error:
            failures.append(
                ContrieverRetrievalFailure(
                    sample_id=query.sample_id,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    return ContrieverCandidateRunSummary(
        manifest_sample_count=len(queries),
        selected_sample_count=len(plan.requested_sample_ids),
        completed_samples=len(completed),
        skipped_existing_samples=len(skipped),
        failed_samples=len(failures),
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        sample_manifest_id=runtime.sample_manifest.manifest_id,
        corpus_manifest_id=runtime.corpus_manifest.corpus_manifest_id,
        retriever_index_fingerprint_sha256=(
            retriever_provenance.index_fingerprint_sha256
        ),
        index_artifact_sha256=retriever.index_artifact_sha256,
        embedding_artifact_sha256=retriever.embedding_artifact_sha256,
        completed_sample_ids=tuple(completed),
        skipped_sample_ids=tuple(skipped),
        failures=tuple(failures),
    )


def _environment_payload(
    *,
    transformers_version: str,
    torch_version: str | None,
    numpy_version: str | None,
    faiss_version: str | None,
    device: str | None,
) -> dict[str, str | None]:
    return {
        "device": device,
        "faiss": faiss_version,
        "implementation": platform.python_implementation(),
        "numpy": numpy_version,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_version,
        "transformers": transformers_version,
    }


def _environment_fingerprint(**versions) -> str:
    serialized = json.dumps(
        _environment_payload(**versions),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _repository_relative(path: Path) -> str:
    return Path(path).resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _git_registry_identity() -> dict[str, object]:
    commit, clean = _git_provenance()
    branch = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff_sha256 = None
    if not clean:
        tracked = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256()
        digest.update(tracked)
        for raw_path in sorted(part for part in untracked.split(b"\0") if part):
            digest.update(raw_path)
            digest.update(b"\0")
            path = REPOSITORY_ROOT / raw_path.decode("utf-8")
            if path.is_file():
                digest.update(path.read_bytes())
            digest.update(b"\0")
        diff_sha256 = digest.hexdigest()
    return {
        "commit": commit,
        "branch": branch,
        "worktree_clean": clean,
        "worktree_diff_sha256": diff_sha256,
    }


def prepare_governed_contriever_run(
    *,
    runtime: ValidatedPubMedQARuntimeCorpus,
    local_index: LocalContrieverIndexIdentity,
    output_dir: Path,
    environment_sha256: str,
    runtime_sha256: str,
    hardware_summary: str,
    evidence_authority_path: Path,
    requested_sample_ids: tuple[str | int, ...] | None = None,
    max_samples: int | None = None,
    created_at: str | None = None,
    git_identity: Mapping[str, object] | None = None,
) -> tuple[CandidateProductionPlan, dict[str, object], dict[str, object]]:
    """Freeze the exact initially-missing run scope before registry mutation."""
    plan = plan_candidate_production(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=output_dir,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=local_index.retriever_provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=CANDIDATE_POOL_SIZE,
        requested_sample_ids=requested_sample_ids,
        max_samples=max_samples,
    )
    if not plan.scheduled_sample_ids:
        raise ValueError("requested Contriever scope contains no missing candidates")
    config_sha256 = hashlib.sha256(
        CONTRIEVER_CONFIG.scientific_json().encode("utf-8")
    ).hexdigest()
    plan_payload = candidate_production_plan_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=runtime.sample_manifest.manifest_id,
        corpus_manifest_id=runtime.corpus_manifest.corpus_manifest_id,
        retriever="contriever",
        retriever_config_sha256=config_sha256,
        index_artifact_id=f"index:sha256:{local_index.cache_fingerprint_sha256}",
        candidate_pool=CANDIDATE_POOL_SIZE,
        top_k=5,
        candidate_directory=_repository_relative(output_dir),
        scheduled_sample_ids=plan.scheduled_sample_ids,
    )
    data = {
        "dataset": "pubmedqa",
        "split": PUBMEDQA_SPLIT,
        "source": PUBMEDQA_SOURCE,
        "revision": PUBMEDQA_REVISION,
        "sample_manifest": artifact_ref(
            PUBMEDQA_SAMPLE_MANIFEST_PATH,
            repository_root=REPOSITORY_ROOT,
            artifact_id=runtime.sample_manifest.manifest_id,
        ),
        "corpus_manifest": artifact_ref(
            PUBMEDQA_OUTPUT_PATH,
            repository_root=REPOSITORY_ROOT,
            artifact_id=runtime.corpus_manifest.corpus_manifest_id,
        ),
    }
    planned = build_retrieval_planned_record(
        created_at=created_at or _utc_now(),
        plan_payload=plan_payload,
        git=_git_registry_identity() if git_identity is None else git_identity,
        data=data,
        retrieval_index=local_index.registry_index_reference(),
        environment_sha256=environment_sha256,
        runtime_sha256=runtime_sha256,
        hardware_summary=hardware_summary,
        output_directory=_repository_relative(output_dir),
        evidence_authority_path=evidence_authority_path,
    )
    return plan, plan_payload, planned


def run_governed_pubmedqa_contriever_candidates(
    *,
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever: ContrieverRetriever,
    local_index: LocalContrieverIndexIdentity,
    output_dir: Path,
    transformers_version: str,
    environment_fingerprint_sha256: str,
    runtime_fingerprint_sha256: str,
    plan: CandidateProductionPlan,
    plan_payload: Mapping[str, object],
    planned_record: Mapping[str, object],
    registry_path: Path,
    evidence_authority_path: Path,
    run_artifact_root: Path,
    candidate_set_path: Path,
    hardware_summary: str,
    resume_reason: str | None = None,
    timestamp: Callable[[], str] = _utc_now,
) -> GovernedContrieverRunSummary:
    """Register, execute, inventory, and close one exact missing-subset run."""
    run_directory = Path(run_artifact_root) / planned_record["run_id"]
    write_candidate_production_plan(
        run_directory / "plan.json",
        plan_payload,
        run_id=planned_record["run_id"],
    )
    existing = [
        record
        for record in read_registry(
            registry_path, evidence_authority_path=evidence_authority_path
        )
        if record["run_id"] == planned_record["run_id"]
    ]
    if not existing:
        append_run_record(
            registry_path,
            planned_record,
            evidence_authority_path=evidence_authority_path,
        )
        latest = dict(planned_record)
    else:
        latest = existing[-1]
        if latest["execution"]["status"] in {"COMPLETE", "FAILED"}:
            raise ValueError("candidate production run is already terminal")

    previous_status = latest["execution"]["status"]
    if previous_status == "PLANNED":
        attempt = 1
        prior_failure = None
    else:
        attempt = latest["execution"]["attempt_count"] + 1
        if not resume_reason:
            raise ValueError("resuming a RUNNING run requires resume_reason")
        prior_failure = resume_reason
    if attempt == 1:
        rechecked = plan_candidate_production(
            sample_manifest=runtime.sample_manifest,
            ordered_queries=runtime.ordered_queries,
            candidate_directory=output_dir,
            dataset_provenance=runtime.dataset_provenance,
            corpus_provenance=runtime.corpus_provenance,
            retriever_provenance=local_index.retriever_provenance,
            corpus_records=runtime.corpus_records,
            candidate_pool=CANDIDATE_POOL_SIZE,
            requested_sample_ids=tuple(plan_payload["scheduled_sample_ids"]),
        )
        if rechecked.skipped_valid_sample_ids:
            raise CandidateArtifactConflictError(
                "initial missing run scope gained existing artifacts before RUNNING"
            )
    running = running_candidate_record(
        latest,
        started_at=latest["execution"]["started_at"] or timestamp(),
        attempt_count=attempt,
        environment_sha256=environment_fingerprint_sha256,
        runtime_sha256=runtime_fingerprint_sha256,
        hardware_summary=hardware_summary,
        prior_failure_reason=prior_failure,
        evidence_authority_path=evidence_authority_path,
    )
    append_run_record(
        registry_path,
        running,
        evidence_authority_path=evidence_authority_path,
    )

    # Everything above happens before index loading or retrieval.
    summary = run_pubmedqa_contriever_candidates(
        runtime=runtime,
        retriever=retriever,
        output_dir=output_dir,
        producing_git_commit=planned_record["git"]["commit"],
        worktree_clean=planned_record["git"]["worktree_clean"],
        environment_fingerprint_sha256=environment_fingerprint_sha256,
        transformers_version=transformers_version,
        sample_ids=tuple(plan_payload["scheduled_sample_ids"]),
        expected_retriever_provenance=local_index.retriever_provenance,
    )
    current = inspect_candidate_directory(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=output_dir,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=local_index.retriever_provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=CANDIDATE_POOL_SIZE,
    )
    current.require_runnable()
    scheduled_keys = {
        canonical_json(value) for value in plan_payload["scheduled_sample_ids"]
    }
    initial_preexisting = tuple(
        entry
        for entry in plan.inspection.valid_entries
        if canonical_json(entry.sample_id) not in scheduled_keys
    )
    failures = tuple(
        {
            "sample_id": failure.sample_id,
            "error_type": failure.error_type,
            "message": failure.message,
        }
        for failure in summary.failures
    )
    output_payload = candidate_production_output_inventory(
        run_id=planned_record["run_id"],
        producer_identity=(
            "scripts.run_pubmedqa_contriever_candidates."
            "run_pubmedqa_contriever_candidates"
        ),
        retriever="contriever",
        repository_root=REPOSITORY_ROOT,
        scheduled_sample_ids=plan_payload["scheduled_sample_ids"],
        initial_preexisting_entries=initial_preexisting,
        current_inspection=current,
        failures=failures,
    )
    output_inventory_path = (
        run_directory / f"attempt_{attempt:04d}_terminal_output.json"
    )
    write_candidate_production_output(output_inventory_path, output_payload)
    complete = (
        output_payload["newly_produced_artifact_count"]
        == output_payload["requested_sample_count"]
        and output_payload["failed_count"] == 0
    )
    final_candidate_set_path = None
    if complete:
        materialize_candidate_set_inventory(
            sample_manifest=runtime.sample_manifest,
            ordered_queries=runtime.ordered_queries,
            candidate_directory=output_dir,
            dataset_provenance=runtime.dataset_provenance,
            corpus_provenance=runtime.corpus_provenance,
            retriever_provenance=local_index.retriever_provenance,
            corpus_records=runtime.corpus_records,
            candidate_pool=CANDIDATE_POOL_SIZE,
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            retriever="contriever",
            output_path=candidate_set_path,
            provenance={"run_id": planned_record["run_id"]},
        )
        final_candidate_set_path = candidate_set_path
    terminal = terminal_candidate_record(
        running,
        completed_at=timestamp(),
        output_inventory_path=output_inventory_path,
        repository_root=REPOSITORY_ROOT,
        successful_count=output_payload["newly_produced_artifact_count"],
        failed_count=output_payload["failed_count"],
        status_counts=output_payload["status_counts"],
        candidate_set_path=final_candidate_set_path,
        failure_reason=(
            None if complete else "candidate subset contains unresolved failures"
        ),
        evidence_authority_path=evidence_authority_path,
    )
    append_run_record(
        registry_path,
        terminal,
        evidence_authority_path=evidence_authority_path,
    )
    return GovernedContrieverRunSummary(
        run_id=planned_record["run_id"],
        status=terminal["execution"]["status"],
        attempt_count=attempt,
        candidate_summary=summary,
        output_inventory_path=output_inventory_path,
        candidate_set_path=final_candidate_set_path,
    )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_provenance() -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _sample_id_argument(value: str) -> str | int:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "sample ID must be a JSON string or integer"
        ) from error
    if isinstance(parsed, bool) or not isinstance(parsed, (str, int)):
        raise argparse.ArgumentTypeError(
            "sample ID must be a JSON string or integer"
        )
    if isinstance(parsed, str) and not parsed.strip():
        raise argparse.ArgumentTypeError("sample ID string must be non-empty")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=_positive_integer,
        help="Process only the first N queries in frozen manifest order.",
    )
    parser.add_argument(
        "--sample-id",
        dest="sample_ids",
        action="append",
        type=_sample_id_argument,
        help=(
            "Exact canonical sample ID as a JSON integer or string; repeat for "
            "multiple IDs. Output is normalized to manifest order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run dataset loading, Contriever cache/index setup, retrieval, and "
            "producer validation, but do not write CandidateArtifact files. "
            "Validated Contriever cache artifacts may still be created or updated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for deterministic per-query CandidateArtifact files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face dataset cache directory.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate local index/candidates and print the exact missing plan; "
            "do not register, load the model, retrieve, or write artifacts."
        ),
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
    )
    parser.add_argument(
        "--evidence-authority-path",
        type=Path,
        default=DEFAULT_EVIDENCE_AUTHORITY_PATH,
    )
    parser.add_argument(
        "--run-artifact-root",
        type=Path,
        default=DEFAULT_RUN_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--candidate-set-path",
        type=Path,
        default=DEFAULT_CANDIDATE_SET_PATH,
    )
    parser.add_argument(
        "--resume-run-id",
        help="Resume the exact active run from its immutable plan artifact.",
    )
    parser.add_argument(
        "--resume-reason",
        help="Prior infrastructure interruption reason for a RUNNING retry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import load_dataset
    import faiss
    import numpy as np
    import torch
    from retrievers.contriever_retriever import ContrieverRetriever

    rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
    )
    runtime = load_validated_pubmedqa_runtime_corpus(tuple(rows))
    transformers_version = importlib_metadata.version("transformers")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    environment_values = dict(
        transformers_version=transformers_version,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        faiss_version=getattr(faiss, "__version__", None)
        or _package_version("faiss-cpu"),
        device=device,
    )
    environment_fingerprint = _environment_fingerprint(**environment_values)
    runtime_fingerprint = stable_json_sha256(
        {
            "candidate_production_module_sha256": file_sha256(
                REPOSITORY_ROOT
                / "src/retrieval_artifacts/candidate_production.py"
            ),
            "contriever_config_sha256": hashlib.sha256(
                CONTRIEVER_CONFIG.scientific_json().encode("utf-8")
            ).hexdigest(),
            "runner_sha256": file_sha256(Path(__file__)),
        }
    )
    hardware_summary = f"device={device}; platform={platform.platform()}"
    local_index = validate_local_contriever_index_identity(
        runtime, transformers_version=transformers_version
    )

    if args.resume_run_id:
        plan_path = args.run_artifact_root / args.resume_run_id / "plan.json"
        stored_plan = read_candidate_production_plan(plan_path)
        if stored_plan["run_id"] != args.resume_run_id:
            raise ValueError("resume plan run_id mismatch")
        plan_payload = stored_plan["scientific_payload"]
        plan = plan_candidate_production(
            sample_manifest=runtime.sample_manifest,
            ordered_queries=runtime.ordered_queries,
            candidate_directory=args.output_dir,
            dataset_provenance=runtime.dataset_provenance,
            corpus_provenance=runtime.corpus_provenance,
            retriever_provenance=local_index.retriever_provenance,
            corpus_records=runtime.corpus_records,
            candidate_pool=CANDIDATE_POOL_SIZE,
            requested_sample_ids=tuple(plan_payload["scheduled_sample_ids"]),
        )
        records = [
            record
            for record in read_registry(
                args.registry_path,
                evidence_authority_path=args.evidence_authority_path,
            )
            if record["run_id"] == args.resume_run_id
        ]
        if not records:
            raise ValueError("resume run_id is absent from registry")
        planned_record = records[-1]
        current_payload = candidate_production_plan_scientific_payload(
            dataset="pubmedqa",
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            sample_manifest_id=runtime.sample_manifest.manifest_id,
            corpus_manifest_id=runtime.corpus_manifest.corpus_manifest_id,
            retriever="contriever",
            retriever_config_sha256=hashlib.sha256(
                CONTRIEVER_CONFIG.scientific_json().encode("utf-8")
            ).hexdigest(),
            index_artifact_id=(
                f"index:sha256:{local_index.cache_fingerprint_sha256}"
            ),
            candidate_pool=CANDIDATE_POOL_SIZE,
            top_k=5,
            candidate_directory=_repository_relative(args.output_dir),
            scheduled_sample_ids=tuple(plan_payload["scheduled_sample_ids"]),
        )
        if current_payload != plan_payload:
            raise ValueError("resume scientific plan identity mismatch")
    else:
        plan, plan_payload, planned_record = prepare_governed_contriever_run(
            runtime=runtime,
            local_index=local_index,
            output_dir=args.output_dir,
            environment_sha256=environment_fingerprint,
            runtime_sha256=runtime_fingerprint,
            hardware_summary=hardware_summary,
            evidence_authority_path=args.evidence_authority_path,
            requested_sample_ids=(
                None if args.sample_ids is None else tuple(args.sample_ids)
            ),
            max_samples=args.max_samples,
        )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "run_id": planned_record["run_id"],
                    "valid_existing_count": len(plan.inspection.valid_entries),
                    "scheduled_count": len(plan.scheduled_sample_ids),
                    "scheduled_sample_ids": plan.scheduled_sample_ids,
                    "index_fingerprint_sha256": (
                        local_index.cache_fingerprint_sha256
                    ),
                    "registry_written": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.dry_run:
        raise ValueError(
            "--dry-run is legacy retriever execution; use --preflight-only for "
            "governed no-write planning"
        )

    governed = run_governed_pubmedqa_contriever_candidates(
        runtime=runtime,
        retriever=ContrieverRetriever(CONTRIEVER_CONFIG),
        local_index=local_index,
        output_dir=args.output_dir,
        transformers_version=transformers_version,
        environment_fingerprint_sha256=environment_fingerprint,
        runtime_fingerprint_sha256=runtime_fingerprint,
        plan=plan,
        plan_payload=plan_payload,
        planned_record=planned_record,
        registry_path=args.registry_path,
        evidence_authority_path=args.evidence_authority_path,
        run_artifact_root=args.run_artifact_root,
        candidate_set_path=args.candidate_set_path,
        hardware_summary=hardware_summary,
        resume_reason=args.resume_reason,
    )
    print(json.dumps(governed, default=lambda value: value.__dict__, sort_keys=True))
    return 0 if governed.status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
