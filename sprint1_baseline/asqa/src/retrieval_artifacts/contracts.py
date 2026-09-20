"""Immutable provenance contracts for frozen Sprint 3 candidate pools.

These contracts validate provenance carried by an artifact, but cannot prove that
candidate identifiers or positions belong to the declared corpus. That check is
the responsibility of the future artifact producer, which has access to the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import re
from typing import Any

import numpy as np

from evaluation.metric_registry import DatasetId


CANDIDATE_SCHEMA_VERSION = "sprint3.candidate.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ARTIFACT_ID_RE = re.compile(r"^candidate:sha256:[0-9a-f]{64}$")


def _require_text(value: object, name: str, *, exact_nonempty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if (not value) if exact_nonempty else (not value.strip()):
        raise ValueError(f"{name} must be non-empty")


def _require_optional_text(value: object, name: str) -> None:
    if value is not None:
        _require_text(value, name)


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")


def _require_integer(value: object, name: str, minimum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _require_stable_id(value: object, name: str) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a string or non-boolean integer")


def _canonical_id(value: str | int) -> str | int:
    return value if isinstance(value, str) else int(value)


def canonical_stable_id(value: object, name: str = "stable_id") -> str | int:
    """Return the canonical scientific representation of a stable artifact ID.

    Exact strings remain strings. Non-boolean integral values, including NumPy
    integer scalars, become Python ``int`` values. This preserves the existing
    retrieval-artifact distinction between integer ``1`` and string ``"1"``.
    """
    _require_stable_id(value, name)
    return _canonical_id(value)


@dataclass(frozen=True)
class CandidateEntry:
    """One ordered, unnormalized entry returned by a retriever."""

    rank: int
    document_id: str | int
    source_document_id: str | int | None
    corpus_position: int | None
    native_score: float
    document_content_sha256: str

    def __post_init__(self) -> None:
        _require_integer(self.rank, "rank", 1)
        _require_stable_id(self.document_id, "document_id")
        if self.source_document_id is not None:
            _require_stable_id(self.source_document_id, "source_document_id")
        if self.corpus_position is not None:
            _require_integer(self.corpus_position, "corpus_position", 0)
        if isinstance(self.native_score, (bool, np.bool_)) or not isinstance(
            self.native_score, Real
        ):
            raise TypeError("native_score must be a non-boolean real number")
        if not math.isfinite(float(self.native_score)):
            raise ValueError("native_score must be finite")
        _require_sha256(self.document_content_sha256, "document_content_sha256")


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_id: DatasetId
    source: str
    config: str | None
    revision: str
    split: str
    sample_manifest_id: str
    sample_manifest_sha256: str
    sampling_seed: int | None
    sampling_algorithm: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be a DatasetId")
        for name in ("source", "revision", "split", "sample_manifest_id"):
            _require_text(getattr(self, name), name)
        _require_optional_text(self.config, "config")
        _require_sha256(self.sample_manifest_sha256, "sample_manifest_sha256")
        if self.sampling_seed is not None:
            _require_integer(self.sampling_seed, "sampling_seed", 0)
            _require_text(self.sampling_algorithm, "sampling_algorithm")
        else:
            _require_optional_text(self.sampling_algorithm, "sampling_algorithm")


@dataclass(frozen=True)
class CorpusProvenance:
    """Compact CandidateArtifact projection of a validated CorpusManifest.

    ``manifest_sha256`` is the scientific ``CorpusManifest.sha256``. Concrete
    ordered-record integrity is a separate producer-layer fingerprint.
    ``preprocessing_version`` currently projects the manifest's versioned
    corpus construction/preparation algorithm.
    """

    corpus_id: str
    source: str
    revision: str
    document_count: int
    manifest_sha256: str
    document_id_map_sha256: str
    preprocessing_version: str

    def __post_init__(self) -> None:
        for name in ("corpus_id", "source", "revision", "preprocessing_version"):
            _require_text(getattr(self, name), name)
        _require_integer(self.document_count, "document_count", 1)
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(self.document_id_map_sha256, "document_id_map_sha256")


@dataclass(frozen=True)
class RetrieverProvenance:
    retriever_name: str
    implementation: str
    library_name: str
    library_version: str
    model_id: str | None
    model_revision: str | None
    tokenizer_id: str | None
    tokenizer_revision: str | None
    query_preprocessing: str
    document_preprocessing: str
    normalization: str
    score_semantics: str
    index_type: str
    index_config: str
    index_fingerprint_sha256: str
    index_artifact_sha256: str | None

    def __post_init__(self) -> None:
        for name in (
            "retriever_name", "implementation", "library_name", "library_version",
            "query_preprocessing", "document_preprocessing", "normalization",
            "score_semantics", "index_type", "index_config",
        ):
            _require_text(getattr(self, name), name)
        for identifier, revision in (
            ("model_id", "model_revision"),
            ("tokenizer_id", "tokenizer_revision"),
        ):
            identifier_value = getattr(self, identifier)
            revision_value = getattr(self, revision)
            _require_optional_text(identifier_value, identifier)
            _require_optional_text(revision_value, revision)
            if identifier_value is None and revision_value is not None:
                raise ValueError(f"{revision} requires {identifier}")
            if identifier_value is not None and revision_value is None:
                raise ValueError(f"{identifier} requires {revision}")
        _require_sha256(self.index_fingerprint_sha256, "index_fingerprint_sha256")
        if self.index_artifact_sha256 is not None:
            _require_sha256(self.index_artifact_sha256, "index_artifact_sha256")


@dataclass(frozen=True)
class CandidateArtifact:
    """Successful raw retriever output before any diversification.

    ``artifact_id`` identifies the scientific retrieval content and configuration.
    ``production_fingerprint_sha256`` separately identifies the code, worktree,
    and environment state that produced that scientific artifact.
    """

    schema_version: str
    dataset: DatasetProvenance
    corpus: CorpusProvenance
    sample_id: str | int
    query_text: str
    retriever: RetrieverProvenance
    requested_top_n: int
    candidates: tuple[CandidateEntry, ...]
    producing_git_commit: str
    worktree_clean: bool
    environment_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {CANDIDATE_SCHEMA_VERSION!r}")
        if not isinstance(self.dataset, DatasetProvenance):
            raise TypeError("dataset must be DatasetProvenance")
        if not isinstance(self.corpus, CorpusProvenance):
            raise TypeError("corpus must be CorpusProvenance")
        if not isinstance(self.retriever, RetrieverProvenance):
            raise TypeError("retriever must be RetrieverProvenance")
        _require_stable_id(self.sample_id, "sample_id")
        _require_text(self.query_text, "query_text", exact_nonempty=True)
        _require_integer(self.requested_top_n, "requested_top_n", 1)
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be an immutable tuple")
        if not self.candidates:
            raise ValueError("candidates must contain at least one entry")
        if len(self.candidates) > self.requested_top_n:
            raise ValueError("candidate count cannot exceed requested_top_n")
        if not all(isinstance(entry, CandidateEntry) for entry in self.candidates):
            raise TypeError("candidates must contain CandidateEntry objects")
        if tuple(entry.rank for entry in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("candidate ranks must be contiguous and match tuple order")
        document_ids = [entry.document_id for entry in self.candidates]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("candidate document_id values must be unique")
        positions = [entry.corpus_position for entry in self.candidates]
        positions_present = [position is not None for position in positions]
        if any(positions_present) and not all(positions_present):
            raise ValueError("corpus_position must be present for all candidates or none")
        if all(positions_present) and len(set(positions)) != len(positions):
            raise ValueError("candidate corpus_position values must be unique")
        if not isinstance(self.producing_git_commit, str) or _GIT_SHA_RE.fullmatch(
            self.producing_git_commit
        ) is None:
            raise ValueError("producing_git_commit must be a full lowercase Git SHA")
        if type(self.worktree_clean) is not bool:
            raise TypeError("worktree_clean must be a bool")
        _require_sha256(
            self.environment_fingerprint_sha256, "environment_fingerprint_sha256"
        )

    def scientific_payload(self) -> dict[str, Any]:
        """Return scientific retrieval identity with exact float encodings."""
        return {
            "candidates": [
                {
                    "corpus_position": entry.corpus_position,
                    "document_content_sha256": entry.document_content_sha256,
                    "document_id": _canonical_id(entry.document_id),
                    "native_score_hex": float(entry.native_score).hex(),
                    "rank": int(entry.rank),
                    "source_document_id": (
                        None
                        if entry.source_document_id is None
                        else _canonical_id(entry.source_document_id)
                    ),
                }
                for entry in self.candidates
            ],
            "corpus": {
                "corpus_id": self.corpus.corpus_id,
                "document_count": int(self.corpus.document_count),
                "document_id_map_sha256": self.corpus.document_id_map_sha256,
                "manifest_sha256": self.corpus.manifest_sha256,
                "preprocessing_version": self.corpus.preprocessing_version,
                "revision": self.corpus.revision,
                "source": self.corpus.source,
            },
            "dataset": {
                "config": self.dataset.config,
                "dataset_id": self.dataset.dataset_id.value,
                "revision": self.dataset.revision,
                "sample_manifest_id": self.dataset.sample_manifest_id,
                "sample_manifest_sha256": self.dataset.sample_manifest_sha256,
                "sampling_algorithm": self.dataset.sampling_algorithm,
                "sampling_seed": (
                    None if self.dataset.sampling_seed is None else int(self.dataset.sampling_seed)
                ),
                "source": self.dataset.source,
                "split": self.dataset.split,
            },
            "query_text": self.query_text,
            "requested_top_n": int(self.requested_top_n),
            "retriever": {
                "document_preprocessing": self.retriever.document_preprocessing,
                "implementation": self.retriever.implementation,
                "index_artifact_sha256": self.retriever.index_artifact_sha256,
                "index_config": self.retriever.index_config,
                "index_fingerprint_sha256": self.retriever.index_fingerprint_sha256,
                "index_type": self.retriever.index_type,
                "library_name": self.retriever.library_name,
                "library_version": self.retriever.library_version,
                "model_id": self.retriever.model_id,
                "model_revision": self.retriever.model_revision,
                "normalization": self.retriever.normalization,
                "query_preprocessing": self.retriever.query_preprocessing,
                "retriever_name": self.retriever.retriever_name,
                "score_semantics": self.retriever.score_semantics,
                "tokenizer_id": self.retriever.tokenizer_id,
                "tokenizer_revision": self.retriever.tokenizer_revision,
            },
            "sample_id": _canonical_id(self.sample_id),
            "schema_version": self.schema_version,
        }

    def scientific_json(self) -> str:
        return json.dumps(
            self.scientific_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.scientific_json().encode("utf-8")).hexdigest()

    @property
    def artifact_id(self) -> str:
        return f"candidate:sha256:{self.sha256}"

    def production_provenance_payload(self) -> dict[str, Any]:
        """Return provenance identifying one production of this artifact."""
        return {
            "candidate_artifact_id": self.artifact_id,
            "environment_fingerprint_sha256": self.environment_fingerprint_sha256,
            "producing_git_commit": self.producing_git_commit,
            "worktree_clean": self.worktree_clean,
        }

    def production_provenance_json(self) -> str:
        return json.dumps(
            self.production_provenance_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def production_fingerprint_sha256(self) -> str:
        return hashlib.sha256(
            self.production_provenance_json().encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CandidateEmbeddingArtifactRef:
    """Binding from one candidate artifact to external representation vectors."""

    representation_name: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    pooling: str
    truncation: str
    normalization: str
    embedding_dimension: int
    dtype: str
    artifact_sha256: str
    candidate_artifact_id: str

    def __post_init__(self) -> None:
        for name in (
            "representation_name", "model_id", "model_revision", "tokenizer_revision",
            "pooling", "truncation", "normalization", "dtype",
        ):
            _require_text(getattr(self, name), name)
        _require_integer(self.embedding_dimension, "embedding_dimension", 1)
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        if not isinstance(self.candidate_artifact_id, str) or _ARTIFACT_ID_RE.fullmatch(
            self.candidate_artifact_id
        ) is None:
            raise ValueError("candidate_artifact_id must be a valid CandidateArtifact ID")
