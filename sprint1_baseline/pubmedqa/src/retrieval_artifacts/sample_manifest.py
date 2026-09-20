"""Immutable, content-addressed query-population manifests for Sprint 3.

Real manifest creation must resolve and freeze an immutable upstream dataset
revision before final experiments. This data-only contract validates a supplied
revision but cannot prove that a remote revision identifier is immutable.
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
from retrieval_artifacts.contracts import DatasetProvenance


SAMPLE_MANIFEST_SCHEMA_VERSION = "sprint3.sample-manifest.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


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


def query_text_sha256(text: str) -> str:
    """Hash exact, unnormalized query text as UTF-8 bytes."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SampleManifestEntry:
    """One ordered query identity in a frozen experimental population."""

    position: int
    sample_id: str | int
    source_sample_id: str | int | None
    query_text_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.position, "position")
        _require_stable_id(self.sample_id, "sample_id")
        if self.source_sample_id is not None:
            _require_stable_id(self.source_sample_id, "source_sample_id")
        if (
            not isinstance(self.query_text_sha256, str)
            or _SHA256_RE.fullmatch(self.query_text_sha256) is None
        ):
            raise ValueError(
                "query_text_sha256 must be a lowercase 64-character SHA-256"
            )


@dataclass(frozen=True)
class SampleSelectionDependency:
    """One upstream source that determines sample eligibility or selection."""

    role: str
    source: str
    config: str | None
    revision: str
    split: str

    def __post_init__(self) -> None:
        for name in ("role", "source", "revision", "split"):
            _require_text(getattr(self, name), name)
        if self.config is not None:
            _require_text(self.config, "config")


@dataclass(frozen=True)
class SampleManifest:
    """Exact ordered sample population for one dataset split or subset."""

    schema_version: str
    dataset_id: DatasetId
    source: str
    config: str | None
    revision: str
    split: str
    sampling_algorithm: str
    sampling_seed: int | None
    requested_sample_size: int | None
    selection_dependencies: tuple[SampleSelectionDependency, ...]
    entries: tuple[SampleManifestEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SAMPLE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SAMPLE_MANIFEST_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be a DatasetId")
        for name in ("source", "revision", "split", "sampling_algorithm"):
            _require_text(getattr(self, name), name)
        if self.config is not None:
            _require_text(self.config, "config")
        if self.sampling_seed is not None:
            _require_nonnegative_integer(self.sampling_seed, "sampling_seed")
        if self.requested_sample_size is not None:
            _require_nonnegative_integer(
                self.requested_sample_size, "requested_sample_size"
            )
            if self.requested_sample_size == 0:
                raise ValueError("requested_sample_size must be positive")
        if not isinstance(self.selection_dependencies, tuple):
            raise TypeError("selection_dependencies must be an immutable tuple")
        if not all(
            isinstance(dependency, SampleSelectionDependency)
            for dependency in self.selection_dependencies
        ):
            raise TypeError(
                "selection_dependencies must contain SampleSelectionDependency objects"
            )
        if len(set(self.selection_dependencies)) != len(self.selection_dependencies):
            raise ValueError("selection_dependencies must not contain duplicates")
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be an immutable tuple")
        if not self.entries:
            raise ValueError("entries must not be empty")
        if not all(isinstance(entry, SampleManifestEntry) for entry in self.entries):
            raise TypeError("entries must contain SampleManifestEntry objects")
        positions = tuple(entry.position for entry in self.entries)
        if positions != tuple(range(len(self.entries))):
            raise ValueError("entry positions must be contiguous and match tuple order")
        sample_ids = [entry.sample_id for entry in self.entries]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("entry sample_id values must be unique")
        source_ids = [
            entry.source_sample_id
            for entry in self.entries
            if entry.source_sample_id is not None
        ]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("present source_sample_id values must be unique")

    @property
    def actual_sample_size(self) -> int:
        return len(self.entries)

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "dataset_id": self.dataset_id.value,
            "entries": [
                {
                    "position": int(entry.position),
                    "query_text_sha256": entry.query_text_sha256,
                    "sample_id": _canonical_id(entry.sample_id),
                    "source_sample_id": (
                        None
                        if entry.source_sample_id is None
                        else _canonical_id(entry.source_sample_id)
                    ),
                }
                for entry in self.entries
            ],
            "requested_sample_size": (
                None
                if self.requested_sample_size is None
                else int(self.requested_sample_size)
            ),
            "revision": self.revision,
            "sampling_algorithm": self.sampling_algorithm,
            "sampling_seed": (
                None if self.sampling_seed is None else int(self.sampling_seed)
            ),
            "schema_version": self.schema_version,
            "selection_dependencies": [
                {
                    "config": dependency.config,
                    "revision": dependency.revision,
                    "role": dependency.role,
                    "source": dependency.source,
                    "split": dependency.split,
                }
                for dependency in self.selection_dependencies
            ],
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
    def manifest_id(self) -> str:
        return f"sample-manifest:sha256:{self.sha256}"


def dataset_provenance_from_sample_manifest(
    manifest: SampleManifest,
) -> DatasetProvenance:
    """Build DatasetProvenance without duplicating manifest identity fields."""
    if not isinstance(manifest, SampleManifest):
        raise TypeError("manifest must be a SampleManifest")
    return DatasetProvenance(
        dataset_id=manifest.dataset_id,
        source=manifest.source,
        config=manifest.config,
        revision=manifest.revision,
        split=manifest.split,
        sample_manifest_id=manifest.manifest_id,
        sample_manifest_sha256=manifest.sha256,
        sampling_seed=manifest.sampling_seed,
        sampling_algorithm=manifest.sampling_algorithm,
    )


def verify_manifest_sample(
    manifest: SampleManifest,
    *,
    sample_id: str | int,
    query_text: str,
) -> SampleManifestEntry:
    """Return the matching entry only when exact query text remains unchanged."""
    if not isinstance(manifest, SampleManifest):
        raise TypeError("manifest must be a SampleManifest")
    _require_stable_id(sample_id, "sample_id")
    for entry in manifest.entries:
        if entry.sample_id == sample_id:
            if entry.query_text_sha256 != query_text_sha256(query_text):
                raise ValueError("query_text does not match the frozen manifest")
            return entry
    raise KeyError(f"sample_id is absent from the manifest: {sample_id!r}")
