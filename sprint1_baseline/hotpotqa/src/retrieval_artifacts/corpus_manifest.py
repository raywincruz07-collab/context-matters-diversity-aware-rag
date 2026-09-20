"""Immutable, content-addressed corpus manifests for Sprint 3.

The manifest identifies scientific corpus contents and construction inputs. It
does not record production details such as Git state, hardware, cache paths, or
timestamps, and it does not by itself prove that a producer followed the
declared construction procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import re
from typing import Any

import numpy as np

from evaluation.metric_registry import DatasetId


CORPUS_MANIFEST_SCHEMA_VERSION = "sprint3.corpus-manifest.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_MANIFEST_ID_RE = re.compile(r"^sample-manifest:sha256:([0-9a-f]{64})$")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_optional_text(value: object, name: str) -> None:
    if value is not None:
        _require_text(value, name)


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")


def _require_stable_id(value: object, name: str) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a string or non-boolean integer")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _canonical_id(value: str | int) -> str | int:
    return value if isinstance(value, str) else int(value)


@dataclass(frozen=True)
class CorpusManifestEntry:
    """One ordered document identity without storing document text.

    ``retrieval_content_sha256`` identifies the canonical prepared document
    content supplied to retrievers before retriever-specific tokenization,
    truncation, encoding, normalization, or model processing. Those later
    transformations belong to RetrieverProvenance rather than corpus identity.
    """

    position: int
    doc_id: str | int
    source_document_id: str | int
    title_sha256: str | None
    text_sha256: str
    retrieval_content_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.position, "position")
        _require_stable_id(self.doc_id, "doc_id")
        _require_stable_id(self.source_document_id, "source_document_id")
        if self.title_sha256 is not None:
            _require_sha256(self.title_sha256, "title_sha256")
        _require_sha256(self.text_sha256, "text_sha256")
        _require_sha256(
            self.retrieval_content_sha256, "retrieval_content_sha256"
        )


@dataclass(frozen=True)
class CorpusConstructionDependency:
    """One pinned upstream input used to construct corpus membership/content."""

    role: str
    source: str
    config: str | None
    revision: str
    split: str

    def __post_init__(self) -> None:
        for name in ("role", "source", "revision", "split"):
            _require_text(getattr(self, name), name)
        _require_optional_text(self.config, "config")


@dataclass(frozen=True)
class CorpusManifest:
    """Exact ordered corpus and the scientific procedure that constructed it.

    Dependencies are declarative pinned inputs, so their supplied tuple order is
    not scientific. Serialization sorts them deterministically without mutating
    the stored tuple. Construction order, when relevant, must be represented
    explicitly by the construction algorithm. Negative-sampling fields are
    structured so global/per-query sampling, exclusion policy, replacement
    policy, and final ordering cannot drift behind a shared algorithm label.
    """

    schema_version: str
    dataset_id: DatasetId
    source: str
    config: str | None
    revision: str
    split: str
    construction_algorithm: str
    input_sample_manifest_id: str | None
    input_sample_manifest_sha256: str | None
    dependencies: tuple[CorpusConstructionDependency, ...]
    rng_family: str | None
    sampling_seed: int | None
    rng_state_semantics: str | None
    requested_negatives_per_query: int | None
    negative_sampling_scope: str | None
    negative_exclusion_scope: str | None
    negative_sampling_without_replacement: bool | None
    final_source_id_ordering: str | None
    entries: tuple[CorpusManifestEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {CORPUS_MANIFEST_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be a DatasetId")
        for name in ("source", "revision", "split", "construction_algorithm"):
            _require_text(getattr(self, name), name)
        _require_optional_text(self.config, "config")

        sample_binding = (
            self.input_sample_manifest_id,
            self.input_sample_manifest_sha256,
        )
        if any(value is not None for value in sample_binding):
            if any(value is None for value in sample_binding):
                raise ValueError(
                    "input sample manifest ID and SHA-256 must be provided together"
                )
            match = (
                _SAMPLE_MANIFEST_ID_RE.fullmatch(self.input_sample_manifest_id)
                if isinstance(self.input_sample_manifest_id, str)
                else None
            )
            if match is None:
                raise ValueError(
                    "input_sample_manifest_id must be a valid SampleManifest ID"
                )
            _require_sha256(
                self.input_sample_manifest_sha256,
                "input_sample_manifest_sha256",
            )
            if match.group(1) != self.input_sample_manifest_sha256:
                raise ValueError(
                    "input sample manifest ID and SHA-256 must identify the same manifest"
                )

        if not isinstance(self.dependencies, tuple):
            raise TypeError("dependencies must be an immutable tuple")
        if not all(
            isinstance(dependency, CorpusConstructionDependency)
            for dependency in self.dependencies
        ):
            raise TypeError(
                "dependencies must contain CorpusConstructionDependency objects"
            )
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must not contain duplicates")

        rng_values = (
            self.rng_family,
            self.sampling_seed,
            self.rng_state_semantics,
        )
        if any(value is not None for value in rng_values):
            if any(value is None for value in rng_values):
                raise ValueError(
                    "rng_family, sampling_seed, and rng_state_semantics must be "
                    "provided together"
                )
            _require_text(self.rng_family, "rng_family")
            _require_nonnegative_integer(self.sampling_seed, "sampling_seed")
            _require_text(self.rng_state_semantics, "rng_state_semantics")

        negative_values = (
            self.requested_negatives_per_query,
            self.negative_sampling_scope,
            self.negative_exclusion_scope,
            self.negative_sampling_without_replacement,
            self.final_source_id_ordering,
        )
        if any(value is not None for value in negative_values):
            if any(value is None for value in negative_values):
                raise ValueError(
                    "all negative-sampling fields must be provided together"
                )
            _require_nonnegative_integer(
                self.requested_negatives_per_query,
                "requested_negatives_per_query",
            )
            for name in (
                "negative_sampling_scope",
                "negative_exclusion_scope",
                "final_source_id_ordering",
            ):
                _require_text(getattr(self, name), name)
            if type(self.negative_sampling_without_replacement) is not bool:
                raise TypeError(
                    "negative_sampling_without_replacement must be a bool"
                )
            if self.rng_family is None:
                raise ValueError("negative sampling requires explicit RNG provenance")
            if self.input_sample_manifest_id is None:
                raise ValueError(
                    "negative sampling requires a SampleManifest binding"
                )

        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be an immutable tuple")
        if not self.entries:
            raise ValueError("entries must not be empty")
        if not all(isinstance(entry, CorpusManifestEntry) for entry in self.entries):
            raise TypeError("entries must contain CorpusManifestEntry objects")
        positions = tuple(entry.position for entry in self.entries)
        if positions != tuple(range(len(self.entries))):
            raise ValueError("entry positions must be contiguous and match tuple order")
        doc_ids = [entry.doc_id for entry in self.entries]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("entry doc_id values must be unique")
        source_ids = [entry.source_document_id for entry in self.entries]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("entry source_document_id values must be unique")

    @property
    def document_count(self) -> int:
        return len(self.entries)

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "construction_algorithm": self.construction_algorithm,
            "dataset_id": self.dataset_id.value,
            "dependencies": [
                {
                    "config": dependency.config,
                    "revision": dependency.revision,
                    "role": dependency.role,
                    "source": dependency.source,
                    "split": dependency.split,
                }
                for dependency in sorted(
                    self.dependencies,
                    key=lambda value: (
                        value.role,
                        value.source,
                        "" if value.config is None else value.config,
                        value.revision,
                        value.split,
                    ),
                )
            ],
            "entries": [
                {
                    "doc_id": _canonical_id(entry.doc_id),
                    "position": int(entry.position),
                    "retrieval_content_sha256": entry.retrieval_content_sha256,
                    "source_document_id": _canonical_id(entry.source_document_id),
                    "text_sha256": entry.text_sha256,
                    "title_sha256": entry.title_sha256,
                }
                for entry in self.entries
            ],
            "final_source_id_ordering": self.final_source_id_ordering,
            "input_sample_manifest_id": self.input_sample_manifest_id,
            "input_sample_manifest_sha256": self.input_sample_manifest_sha256,
            "negative_exclusion_scope": self.negative_exclusion_scope,
            "negative_sampling_scope": self.negative_sampling_scope,
            "negative_sampling_without_replacement": (
                self.negative_sampling_without_replacement
            ),
            "requested_negatives_per_query": (
                None
                if self.requested_negatives_per_query is None
                else int(self.requested_negatives_per_query)
            ),
            "revision": self.revision,
            "rng_family": self.rng_family,
            "rng_state_semantics": self.rng_state_semantics,
            "sampling_seed": (
                None if self.sampling_seed is None else int(self.sampling_seed)
            ),
            "schema_version": self.schema_version,
            "source": self.source,
            "split": self.split,
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
    def corpus_manifest_id(self) -> str:
        return f"corpus-manifest:sha256:{self.sha256}"
