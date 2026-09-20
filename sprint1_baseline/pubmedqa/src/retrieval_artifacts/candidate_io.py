"""Deterministic storage for validated CandidateArtifact objects."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from evaluation.metric_registry import DatasetId
from retrieval_artifacts.contracts import (
    CandidateArtifact,
    CandidateEntry,
    CorpusProvenance,
    DatasetProvenance,
    RetrieverProvenance,
)


CANDIDATE_ARTIFACT_FORMAT = "sprint3.candidate-artifact.v1"


class CandidateArtifactConflictError(ValueError):
    """An existing path contains a different valid artifact."""


def candidate_artifact_payload(artifact: CandidateArtifact) -> dict[str, Any]:
    return {
        "artifact_format": CANDIDATE_ARTIFACT_FORMAT,
        "artifact_id": artifact.artifact_id,
        "production_fingerprint_sha256": artifact.production_fingerprint_sha256,
        "production_provenance": artifact.production_provenance_payload(),
        "scientific_payload": artifact.scientific_payload(),
        "scientific_sha256": artifact.sha256,
    }


def _artifact_from_payload(payload: Mapping[str, Any]) -> CandidateArtifact:
    dataset = payload["dataset"]
    corpus = payload["corpus"]
    retriever = payload["retriever"]
    production = payload["production_provenance"]
    return CandidateArtifact(
        schema_version=payload["schema_version"],
        dataset=DatasetProvenance(
            dataset_id=DatasetId(dataset["dataset_id"]),
            source=dataset["source"],
            config=dataset["config"],
            revision=dataset["revision"],
            split=dataset["split"],
            sample_manifest_id=dataset["sample_manifest_id"],
            sample_manifest_sha256=dataset["sample_manifest_sha256"],
            sampling_seed=dataset["sampling_seed"],
            sampling_algorithm=dataset["sampling_algorithm"],
        ),
        corpus=CorpusProvenance(**corpus),
        sample_id=payload["sample_id"],
        query_text=payload["query_text"],
        retriever=RetrieverProvenance(**retriever),
        requested_top_n=payload["requested_top_n"],
        candidates=tuple(
            CandidateEntry(
                rank=entry["rank"],
                document_id=entry["document_id"],
                source_document_id=entry["source_document_id"],
                corpus_position=entry["corpus_position"],
                native_score=float.fromhex(entry["native_score_hex"]),
                document_content_sha256=entry["document_content_sha256"],
            )
            for entry in payload["candidates"]
        ),
        producing_git_commit=production["producing_git_commit"],
        worktree_clean=production["worktree_clean"],
        environment_fingerprint_sha256=production[
            "environment_fingerprint_sha256"
        ],
    )


def read_candidate_artifact(path: Path) -> CandidateArtifact:
    """Read and verify scientific and production identities."""
    with path.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if not isinstance(stored, Mapping):
        raise ValueError("candidate artifact wrapper must be an object")
    if stored.get("artifact_format") != CANDIDATE_ARTIFACT_FORMAT:
        raise ValueError("unexpected candidate artifact format")
    scientific = stored.get("scientific_payload")
    production = stored.get("production_provenance")
    if not isinstance(scientific, Mapping) or not isinstance(production, Mapping):
        raise ValueError("candidate artifact payloads must be objects")
    artifact = _artifact_from_payload(
        {**scientific, "production_provenance": production}
    )
    if artifact.scientific_payload() != scientific:
        raise ValueError("stored candidate scientific payload is not canonical")
    if artifact.production_provenance_payload() != production:
        raise ValueError("stored candidate production provenance is not canonical")
    checks = {
        "artifact_id": artifact.artifact_id,
        "scientific_sha256": artifact.sha256,
        "production_fingerprint_sha256": artifact.production_fingerprint_sha256,
    }
    for name, expected in checks.items():
        if stored.get(name) != expected:
            raise ValueError(f"stored {name} does not match candidate payload")
    return artifact


def write_candidate_artifact(artifact: CandidateArtifact, path: Path) -> None:
    """Atomically create an artifact, safely skipping an exact existing copy."""
    if not isinstance(artifact, CandidateArtifact):
        raise TypeError("artifact must be a CandidateArtifact")
    if path.exists():
        existing = read_candidate_artifact(path)
        if existing != artifact:
            raise CandidateArtifactConflictError(
                f"existing candidate artifact differs: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        candidate_artifact_payload(artifact),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
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
        # A concurrent producer must not be silently overwritten.
        os.link(temporary_path, path)
    except FileExistsError:
        existing = read_candidate_artifact(path)
        if existing != artifact:
            raise CandidateArtifactConflictError(
                f"existing candidate artifact differs: {path}"
            )
    finally:
        temporary_path.unlink(missing_ok=True)
