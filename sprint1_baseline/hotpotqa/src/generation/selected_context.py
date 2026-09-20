"""Content-addressed relevance-only Top-5 context artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from generation._io import (
    canonical_json,
    file_sha256,
    sha256_text,
    stable_json_sha256,
    write_immutable_json,
)
from generation.prompts import render_context_block
from retrieval_artifacts import CandidateArtifact, CorpusRecord, read_candidate_artifact
from retrieval_artifacts.contracts import canonical_stable_id
from run_registry import read_candidate_set_artifact


SELECTED_CONTEXT_SCHEMA_VERSION = "sprint3.selected-context.v1"
SELECTED_CONTEXT_ARTIFACT_FORMAT = "sprint3.selected-context-artifact.v1"
SELECTED_CONTEXT_SET_SCHEMA_VERSION = "sprint3.selected-context-set.v1"
SELECTED_CONTEXT_SET_ARTIFACT_FORMAT = "sprint3.selected-context-set-artifact.v1"
_SELECTED_CONTEXT_ID_RE = re.compile(r"^selected-context:sha256:[0-9a-f]{64}$")
_CANDIDATE_SET_ID_RE = re.compile(r"^candidate-set:sha256:[0-9a-f]{64}$")


class SelectedContextConflictError(ValueError):
    """An existing selected-context artifact has conflicting content."""


@dataclass(frozen=True)
class SelectedPassage:
    rank: int
    document_id: str | int
    source_document_id: str | int
    corpus_position: int
    candidate_document_content_sha256: str
    passage_body: str
    passage_body_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise TypeError("rank must be a positive non-boolean integer")
        for name in ("document_id", "source_document_id"):
            canonical_stable_id(getattr(self, name), name)
        if (
            isinstance(self.corpus_position, bool)
            or not isinstance(self.corpus_position, int)
            or self.corpus_position < 0
        ):
            raise TypeError("corpus_position must be a nonnegative integer")
        if not isinstance(self.passage_body, str) or not self.passage_body:
            raise ValueError("passage_body must be non-empty")
        if self.passage_body != self.passage_body.strip():
            raise ValueError("passage_body must have canonical outer whitespace")
        if sha256_text(self.passage_body) != self.passage_body_sha256:
            raise ValueError("passage_body_sha256 does not match passage_body")

    def payload(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "document_id": canonical_stable_id(self.document_id, "document_id"),
            "source_document_id": canonical_stable_id(
                self.source_document_id, "source_document_id"
            ),
            "corpus_position": self.corpus_position,
            "candidate_document_content_sha256": (
                self.candidate_document_content_sha256
            ),
            "passage_body": self.passage_body,
            "passage_body_sha256": self.passage_body_sha256,
        }


@dataclass(frozen=True)
class SelectedContextArtifact:
    dataset: str
    evidence_role: str
    sample_manifest_id: str
    corpus_manifest_id: str
    sample_id: str | int
    query_text: str
    retriever: str
    candidate_set_id: str
    candidate_artifact_id: str
    passages: tuple[SelectedPassage, ...]
    context_block: str

    def __post_init__(self) -> None:
        for name in (
            "dataset",
            "evidence_role",
            "sample_manifest_id",
            "corpus_manifest_id",
            "query_text",
            "retriever",
            "candidate_artifact_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        canonical_stable_id(self.sample_id, "sample_id")
        if _CANDIDATE_SET_ID_RE.fullmatch(self.candidate_set_id) is None:
            raise ValueError("candidate_set_id is invalid")
        if not isinstance(self.passages, tuple) or len(self.passages) != 5:
            raise ValueError("selected context must contain exactly five passages")
        if tuple(item.rank for item in self.passages) != (1, 2, 3, 4, 5):
            raise ValueError("selected passage ranks must be exactly 1..5 in order")
        document_keys = [canonical_json(item.document_id) for item in self.passages]
        if len(set(document_keys)) != 5:
            raise ValueError("selected passages must have unique document IDs")
        expected_context = render_context_block(
            tuple(item.passage_body for item in self.passages)
        )
        if self.context_block != expected_context:
            raise ValueError("context rendering differs from selected passage bodies")

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTED_CONTEXT_SCHEMA_VERSION,
            "dataset": self.dataset,
            "evidence_role": self.evidence_role,
            "sample_manifest_id": self.sample_manifest_id,
            "corpus_manifest_id": self.corpus_manifest_id,
            "sample_id": canonical_stable_id(self.sample_id, "sample_id"),
            "query_text": self.query_text,
            "query_text_sha256": sha256_text(self.query_text),
            "retriever": self.retriever,
            "method": "none",
            "candidate_pool": 20,
            "top_k": 5,
            "candidate_set_id": self.candidate_set_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "passages": [item.payload() for item in self.passages],
            "ordered_document_ids": [
                canonical_stable_id(item.document_id, "document_id")
                for item in self.passages
            ],
            "ordered_body_sha256": [
                item.passage_body_sha256 for item in self.passages
            ],
            "context_block": self.context_block,
            "context_block_sha256": sha256_text(self.context_block),
        }

    @property
    def sha256(self) -> str:
        return stable_json_sha256(self.scientific_payload())

    @property
    def artifact_id(self) -> str:
        return f"selected-context:sha256:{self.sha256}"

    def wrapper(self) -> dict[str, Any]:
        return {
            "artifact_format": SELECTED_CONTEXT_ARTIFACT_FORMAT,
            "selected_context_id": self.artifact_id,
            "scientific_sha256": self.sha256,
            "scientific_payload": self.scientific_payload(),
        }


def _record_by_position(
    corpus_records: tuple[CorpusRecord, ...], position: int
) -> CorpusRecord:
    if position >= len(corpus_records):
        raise ValueError("candidate corpus position is outside corpus")
    record = corpus_records[position]
    if record.corpus_position != position:
        raise ValueError("corpus tuple order does not match corpus positions")
    return record


def _candidate_set_entry(
    candidate_set: Mapping[str, Any], sample_id: str | int
) -> Mapping[str, Any]:
    scientific = candidate_set["scientific_payload"]
    target = canonical_json(canonical_stable_id(sample_id, "sample_id"))
    matches = [
        entry
        for entry in scientific["entries"]
        if canonical_json(entry["sample_id"]) == target
    ]
    if len(matches) != 1:
        raise ValueError("candidate set does not contain exactly one sample entry")
    return matches[0]


def build_relevance_selected_context(
    *,
    candidate_artifact: CandidateArtifact,
    candidate_set: Mapping[str, Any],
    corpus_records: tuple[CorpusRecord, ...],
) -> SelectedContextArtifact:
    """Select exact ranks 1..5 without retrieval or score modification."""
    if candidate_artifact.requested_top_n != 20 or len(candidate_artifact.candidates) != 20:
        raise ValueError("canonical selected context requires an exact Top-20 artifact")
    scientific_set = candidate_set["scientific_payload"]
    if scientific_set["dataset"] != candidate_artifact.dataset.dataset_id.value:
        raise ValueError("candidate set dataset mismatch")
    if scientific_set["sample_manifest_id"] != candidate_artifact.dataset.sample_manifest_id:
        raise ValueError("candidate set sample manifest mismatch")
    if scientific_set["retriever"] != candidate_artifact.retriever.retriever_name:
        raise ValueError("candidate set retriever mismatch")
    set_entry = _candidate_set_entry(candidate_set, candidate_artifact.sample_id)
    if set_entry["candidate_artifact_id"] != candidate_artifact.artifact_id:
        raise ValueError("candidate set points to a different candidate artifact")

    selected: list[SelectedPassage] = []
    for expected_rank, entry in enumerate(candidate_artifact.candidates[:5], start=1):
        if entry.rank != expected_rank or entry.corpus_position is None:
            raise ValueError("candidate ranks 1..5 are not canonical")
        record = _record_by_position(corpus_records, entry.corpus_position)
        if canonical_json(record.document_id) != canonical_json(entry.document_id):
            raise ValueError("candidate document ID does not match corpus")
        if entry.source_document_id is None or canonical_json(
            record.source_document_id
        ) != canonical_json(entry.source_document_id):
            raise ValueError("candidate source document ID does not match corpus")
        from retrieval_artifacts import document_content_sha256

        if document_content_sha256(record.retrieval_content) != entry.document_content_sha256:
            raise ValueError("candidate document content hash does not match corpus")
        body = record.text
        if candidate_artifact.dataset.dataset_id.value == "pubmedqa" and (
            body != record.retrieval_content
            or document_content_sha256(body) != entry.document_content_sha256
        ):
            raise ValueError("PubMedQA passage body differs from validated candidate content")
        selected.append(
            SelectedPassage(
                rank=expected_rank,
                document_id=entry.document_id,
                source_document_id=entry.source_document_id,
                corpus_position=entry.corpus_position,
                candidate_document_content_sha256=entry.document_content_sha256,
                passage_body=body,
                passage_body_sha256=sha256_text(body),
            )
        )
    bodies = tuple(item.passage_body for item in selected)
    return SelectedContextArtifact(
        dataset=candidate_artifact.dataset.dataset_id.value,
        evidence_role=scientific_set["evidence_role"],
        sample_manifest_id=candidate_artifact.dataset.sample_manifest_id,
        corpus_manifest_id=candidate_artifact.corpus.corpus_id,
        sample_id=candidate_artifact.sample_id,
        query_text=candidate_artifact.query_text,
        retriever=candidate_artifact.retriever.retriever_name,
        candidate_set_id=candidate_set["candidate_set_id"],
        candidate_artifact_id=candidate_artifact.artifact_id,
        passages=tuple(selected),
        context_block=render_context_block(bodies),
    )


def _from_scientific_payload(payload: Mapping[str, Any]) -> SelectedContextArtifact:
    if payload.get("schema_version") != SELECTED_CONTEXT_SCHEMA_VERSION:
        raise ValueError("selected-context schema version mismatch")
    passages = tuple(
        SelectedPassage(
            rank=item["rank"],
            document_id=item["document_id"],
            source_document_id=item["source_document_id"],
            corpus_position=item["corpus_position"],
            candidate_document_content_sha256=item[
                "candidate_document_content_sha256"
            ],
            passage_body=item["passage_body"],
            passage_body_sha256=item["passage_body_sha256"],
        )
        for item in payload["passages"]
    )
    artifact = SelectedContextArtifact(
        dataset=payload["dataset"],
        evidence_role=payload["evidence_role"],
        sample_manifest_id=payload["sample_manifest_id"],
        corpus_manifest_id=payload["corpus_manifest_id"],
        sample_id=payload["sample_id"],
        query_text=payload["query_text"],
        retriever=payload["retriever"],
        candidate_set_id=payload["candidate_set_id"],
        candidate_artifact_id=payload["candidate_artifact_id"],
        passages=passages,
        context_block=payload["context_block"],
    )
    if artifact.scientific_payload() != dict(payload):
        raise ValueError("selected-context scientific payload is not canonical")
    return artifact


def read_selected_context(path: Path) -> SelectedContextArtifact:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping):
        raise ValueError("selected-context artifact must be an object")
    if set(stored) != {
        "artifact_format",
        "selected_context_id",
        "scientific_sha256",
        "scientific_payload",
    }:
        raise ValueError("selected-context wrapper keys mismatch")
    if stored["artifact_format"] != SELECTED_CONTEXT_ARTIFACT_FORMAT:
        raise ValueError("selected-context artifact format mismatch")
    if not isinstance(stored["scientific_payload"], Mapping):
        raise ValueError("selected-context scientific payload must be an object")
    artifact = _from_scientific_payload(stored["scientific_payload"])
    if stored != artifact.wrapper():
        raise ValueError("selected-context artifact identity mismatch")
    return artifact


def write_selected_context(artifact: SelectedContextArtifact, path: Path) -> None:
    write_immutable_json(
        path,
        artifact.wrapper(),
        conflict_error=SelectedContextConflictError,
    )


def selected_context_set_payload(
    *,
    dataset: str,
    evidence_role: str,
    sample_manifest_id: str,
    retriever: str,
    candidate_set_id: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in entries:
        sample_id = canonical_stable_id(raw["sample_id"], "sample_id")
        key = canonical_json(sample_id)
        if key in seen:
            raise ValueError("selected-context set sample IDs must be unique")
        seen.add(key)
        artifact_id = raw["selected_context_id"]
        if not isinstance(artifact_id, str) or _SELECTED_CONTEXT_ID_RE.fullmatch(artifact_id) is None:
            raise ValueError("selected context ID is invalid")
        canonical_entries.append(
            {"sample_id": sample_id, "selected_context_id": artifact_id}
        )
    canonical_entries.sort(key=lambda item: canonical_json(item["sample_id"]))
    if not canonical_entries:
        raise ValueError("selected-context set must not be empty")
    return {
        "schema_version": SELECTED_CONTEXT_SET_SCHEMA_VERSION,
        "dataset": dataset,
        "evidence_role": evidence_role,
        "sample_manifest_id": sample_manifest_id,
        "retriever": retriever,
        "method": "none",
        "candidate_pool": 20,
        "top_k": 5,
        "candidate_set_id": candidate_set_id,
        "expected_query_count": len(canonical_entries),
        "entries": canonical_entries,
    }


def selected_context_set_wrapper(
    scientific_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    scientific = selected_context_set_payload(
        dataset=scientific_payload["dataset"],
        evidence_role=scientific_payload["evidence_role"],
        sample_manifest_id=scientific_payload["sample_manifest_id"],
        retriever=scientific_payload["retriever"],
        candidate_set_id=scientific_payload["candidate_set_id"],
        entries=scientific_payload["entries"],
    )
    digest = stable_json_sha256(scientific)
    canonical_json(provenance)
    return {
        "artifact_format": SELECTED_CONTEXT_SET_ARTIFACT_FORMAT,
        "selected_context_set_id": f"selected-context:sha256:{digest}",
        "scientific_sha256": digest,
        "scientific_payload": scientific,
        "provenance": dict(provenance),
    }


def write_selected_context_set(
    path: Path,
    scientific_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> None:
    write_immutable_json(
        path,
        selected_context_set_wrapper(scientific_payload, provenance=provenance),
        conflict_error=SelectedContextConflictError,
    )


def read_selected_context_set(path: Path) -> dict[str, Any]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping):
        raise ValueError("selected-context set must be an object")
    if set(stored) != {
        "artifact_format",
        "selected_context_set_id",
        "scientific_sha256",
        "scientific_payload",
        "provenance",
    }:
        raise ValueError("selected-context set wrapper keys mismatch")
    rebuilt = selected_context_set_wrapper(
        stored["scientific_payload"], provenance=stored["provenance"]
    )
    if dict(stored) != rebuilt:
        raise ValueError("selected-context set identity mismatch")
    return rebuilt


def selected_context_path(output_directory: Path, position: int) -> Path:
    return Path(output_directory) / f"sample_{position:04d}.json"


def materialize_relevance_selected_contexts(
    *,
    runtime: Any,
    candidate_directory: Path,
    candidate_set_path: Path,
    output_directory: Path,
    output_inventory_path: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Create only missing Top-5 contexts and require a complete valid set."""
    repository_root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    candidate_set = read_candidate_set_artifact(candidate_set_path)
    scientific_set = candidate_set["scientific_payload"]
    if scientific_set["expected_query_count"] != len(runtime.ordered_queries):
        raise ValueError("candidate set does not cover the runtime query manifest")
    entries: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for query in runtime.ordered_queries:
        candidate_path = Path(candidate_directory) / f"sample_{query.position:04d}.json"
        candidate = read_candidate_artifact(candidate_path)
        expected = build_relevance_selected_context(
            candidate_artifact=candidate,
            candidate_set=candidate_set,
            corpus_records=runtime.corpus_records,
        )
        path = selected_context_path(output_directory, query.position)
        if path.exists():
            existing = read_selected_context(path)
            if existing != expected:
                raise SelectedContextConflictError(
                    f"existing selected context differs: {path}"
                )
        else:
            write_selected_context(expected, path)
        entries.append(
            {"sample_id": query.sample_id, "selected_context_id": expected.artifact_id}
        )
        files.append(
            {
                "sample_id": canonical_stable_id(query.sample_id, "sample_id"),
                "path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
                "sha256": file_sha256(path),
                "selected_context_id": expected.artifact_id,
            }
        )
    scientific = selected_context_set_payload(
        dataset=scientific_set["dataset"],
        evidence_role=scientific_set["evidence_role"],
        sample_manifest_id=scientific_set["sample_manifest_id"],
        retriever=scientific_set["retriever"],
        candidate_set_id=candidate_set["candidate_set_id"],
        entries=entries,
    )
    provenance = {
        "candidate_set_path": Path(candidate_set_path).resolve().relative_to(
            repository_root.resolve()
        ).as_posix(),
        "artifact_files": files,
    }
    write_selected_context_set(
        output_inventory_path, scientific, provenance=provenance
    )
    return read_selected_context_set(output_inventory_path)
