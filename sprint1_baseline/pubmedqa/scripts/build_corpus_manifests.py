#!/usr/bin/env python3
"""Build deterministic Sprint 3 corpus-manifest artifacts.

Only PubMedQA is supported in this milestone. Importing this module performs no
dataset access and writes no artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import json
from numbers import Integral
import os
from pathlib import Path
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    SampleManifest,
    CorpusProvenance,
    DatasetProvenance,
    corpus_provenance_from_corpus_manifest,
    dataset_provenance_from_sample_manifest,
    document_content_sha256,
    query_text_sha256,
    validate_corpus_records_against_manifest,
    verify_manifest_sample,
)
from scripts.build_sample_manifests import read_and_verify_manifest_artifact


PUBMEDQA_SOURCE = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
PUBMEDQA_SPLIT = "train"
PUBMEDQA_EXPECTED_SAMPLE_COUNT = 1000
PUBMEDQA_EXPECTED_DOCUMENT_COUNT = 3358
PUBMEDQA_SAMPLE_MANIFEST_ID = (
    "sample-manifest:sha256:"
    "4704a5250f52144a38b7f4734f887fe11f5bad18198c285fb048b21d414387ff"
)
PUBMEDQA_SAMPLE_MANIFEST_SHA256 = (
    "4704a5250f52144a38b7f4734f887fe11f5bad18198c285fb048b21d414387ff"
)
PUBMEDQA_CONSTRUCTION_ALGORITHM = (
    "pubmedqa_context_sections_source_order_strip.v1"
)
PUBMEDQA_SECTION_SOURCE_ID_PROTOCOL = "pubmedqa_pubid_context_ordinal.v1"
CORPUS_MANIFEST_ARTIFACT_FORMAT = "sprint3.corpus-manifest-artifact.v1"
PUBMEDQA_PROVENANCE_NOTE = (
    "This artifact freezes a pinned immutable source revision for the clean "
    "Sprint 3 rerun and reconstructs the documented historical PubMedQA corpus "
    "construction semantics; it does not prove the exact upstream revision or "
    "exact source text bytes used historically in Sprint 1."
)
PUBMEDQA_SAMPLE_MANIFEST_PATH = REPOSITORY_ROOT / (
    "artifacts/sample_manifests/pubmedqa_sample_manifest_v2.json"
)
PUBMEDQA_HISTORICAL_ARTIFACT = REPOSITORY_ROOT / (
    "results/sprint1/raw/fullrag_bm25_top5.csv"
)
PUBMEDQA_OUTPUT_PATH = REPOSITORY_ROOT / (
    "artifacts/corpus_manifests/pubmedqa_corpus_manifest_v1.json"
)
PUBMEDQA_CORPUS_MANIFEST_SHA256 = (
    "e4a9d368c9b3d4270e6bcc721e2e0e8d807d47891ac6039d866ea1c6e44b2a43"
)
PUBMEDQA_CORPUS_MANIFEST_ID = (
    f"corpus-manifest:sha256:{PUBMEDQA_CORPUS_MANIFEST_SHA256}"
)
_CANONICAL_DIGITS = re.compile(r"^(?:0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class PubMedQACorpusBuild:
    corpus_manifest: CorpusManifest
    corpus_records: tuple[CorpusRecord, ...]
    gold_document_ids_by_sample: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class HistoricalCompatibility:
    sample_count: int
    document_count: int
    gold_assignment_count: int


@dataclass(frozen=True)
class ValidatedPubMedQAQuery:
    position: int
    sample_id: str | int
    query_text: str


@dataclass(frozen=True)
class ValidatedPubMedQARuntimeCorpus:
    sample_manifest: SampleManifest
    corpus_manifest: CorpusManifest
    corpus_records: tuple[CorpusRecord, ...]
    dataset_provenance: DatasetProvenance
    corpus_provenance: CorpusProvenance
    ordered_queries: tuple[ValidatedPubMedQAQuery, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ordered_queries, tuple):
            raise TypeError("ordered_queries must be an immutable tuple")
        if len(self.ordered_queries) != self.sample_manifest.actual_sample_size:
            raise ValueError("ordered queries do not cover the SampleManifest")
        for entry, query in zip(
            self.sample_manifest.entries, self.ordered_queries, strict=True
        ):
            if not isinstance(query, ValidatedPubMedQAQuery):
                raise TypeError(
                    "ordered_queries must contain ValidatedPubMedQAQuery objects"
                )
            if query.position != entry.position or query.sample_id != entry.sample_id:
                raise ValueError("ordered query identity does not match SampleManifest")
            verify_manifest_sample(
                self.sample_manifest,
                sample_id=query.sample_id,
                query_text=query.query_text,
            )
        expected_dataset = dataset_provenance_from_sample_manifest(
            self.sample_manifest
        )
        if self.dataset_provenance != expected_dataset:
            raise ValueError("runtime DatasetProvenance does not match SampleManifest")
        expected_corpus = corpus_provenance_from_corpus_manifest(
            corpus_manifest=self.corpus_manifest,
            corpus_records=self.corpus_records,
            dataset_provenance=self.dataset_provenance,
        )
        if self.corpus_provenance != expected_corpus:
            raise ValueError("runtime CorpusProvenance does not match corpus records")


def pubmedqa_section_source_document_id(
    pubid: str | int, context_index: int
) -> str:
    """Encode a canonical PubMedQA parent ID and zero-based section ordinal."""
    if isinstance(pubid, (bool, np.bool_)):
        raise TypeError("pubid must be an integer or canonical digit string")
    if isinstance(pubid, Integral):
        if pubid < 0:
            raise ValueError("pubid integer must be nonnegative")
        encoded_pubid = str(int(pubid))
    elif isinstance(pubid, str) and _CANONICAL_DIGITS.fullmatch(pubid):
        encoded_pubid = pubid
    else:
        raise ValueError("pubid must be an integer or canonical digit string")
    if isinstance(context_index, bool) or not isinstance(context_index, int):
        raise TypeError("context_index must be a non-boolean integer")
    if context_index < 0:
        raise ValueError("context_index must be nonnegative")
    return f"pubmedqa:pubid:{encoded_pubid}:context:{context_index}"


def _pubmedqa_source_ids_equal(left: object, right: object) -> bool:
    """Compare IDs by stable string or non-boolean Integral family."""
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return False
    if isinstance(left, Integral) and isinstance(right, Integral):
        return int(left) == int(right)
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    return False


def _pubmedqa_source_id_key(value: str | int) -> tuple[str, str | int]:
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return ("integral", int(value))
    return ("string", value)


def load_frozen_pubmedqa_sample_manifest(path: Path) -> SampleManifest:
    """Load and cryptographically verify the exact frozen PubMedQA manifest."""
    manifest = read_and_verify_manifest_artifact(path)
    expected = {
        "dataset_id": DatasetId.PUBMEDQA,
        "source": PUBMEDQA_SOURCE,
        "config": PUBMEDQA_CONFIG,
        "revision": PUBMEDQA_REVISION,
        "split": PUBMEDQA_SPLIT,
        "manifest_id": PUBMEDQA_SAMPLE_MANIFEST_ID,
        "sha256": PUBMEDQA_SAMPLE_MANIFEST_SHA256,
    }
    for name, value in expected.items():
        if getattr(manifest, name) != value:
            raise ValueError(f"frozen PubMedQA SampleManifest {name} mismatch")
    return manifest


def _require_pubmedqa_manifest_identity(manifest: SampleManifest) -> None:
    if not isinstance(manifest, SampleManifest):
        raise TypeError("sample_manifest must be a SampleManifest")
    expected = (
        DatasetId.PUBMEDQA,
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        PUBMEDQA_REVISION,
        PUBMEDQA_SPLIT,
    )
    actual = (
        manifest.dataset_id,
        manifest.source,
        manifest.config,
        manifest.revision,
        manifest.split,
    )
    if actual != expected:
        raise ValueError("SampleManifest does not identify pinned PubMedQA")


def build_pubmedqa_corpus_manifest(
    rows: Sequence[Mapping[str, Any]],
    sample_manifest: SampleManifest,
    *,
    expected_sample_count: int = PUBMEDQA_EXPECTED_SAMPLE_COUNT,
    expected_document_count: int = PUBMEDQA_EXPECTED_DOCUMENT_COUNT,
) -> PubMedQACorpusBuild:
    """Construct exact stripped PubMedQA context sections in source order."""
    _require_pubmedqa_manifest_identity(sample_manifest)
    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError("rows must be an ordered iterable of mappings")
    try:
        ordered_rows = tuple(rows)
    except TypeError as error:
        raise TypeError("rows must be an ordered iterable of mappings") from error
    if len(ordered_rows) != expected_sample_count:
        raise ValueError(
            f"expected {expected_sample_count} PubMedQA rows, found {len(ordered_rows)}"
        )
    if len(ordered_rows) != sample_manifest.actual_sample_size:
        raise ValueError("dataset row count does not match SampleManifest")

    records: list[CorpusRecord] = []
    entries: list[CorpusManifestEntry] = []
    grouped_ids: list[tuple[int, ...]] = []
    seen_pubids: set[tuple[str, str | int]] = set()
    for position, (row, sample_entry) in enumerate(
        zip(ordered_rows, sample_manifest.entries, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise TypeError(f"dataset row {position} must be a mapping")
        pubid = row.get("pubid")
        # Validate canonical encodability before comparing its stable type/value.
        pubmedqa_section_source_document_id(pubid, 0)
        typed_pubid = _pubmedqa_source_id_key(pubid)
        if typed_pubid in seen_pubids:
            raise ValueError(f"duplicate PubMedQA pubid at position {position}")
        seen_pubids.add(typed_pubid)
        if not _pubmedqa_source_ids_equal(sample_entry.source_sample_id, pubid):
            raise ValueError(f"PubMedQA pubid mismatch at position {position}")
        question = row.get("question")
        if not isinstance(question, str):
            raise TypeError(f"question at position {position} must be a string")
        if query_text_sha256(question) != sample_entry.query_text_sha256:
            raise ValueError(f"PubMedQA question mismatch at position {position}")

        context = row.get("context")
        if not isinstance(context, Mapping):
            raise TypeError(f"context at position {position} must be a mapping")
        contexts = context.get("contexts")
        labels = context.get("labels")
        if (
            isinstance(contexts, (str, bytes))
            or not isinstance(contexts, Sequence)
            or isinstance(labels, (str, bytes))
            or not isinstance(labels, Sequence)
        ):
            raise TypeError("contexts and labels must be sequences")
        if len(contexts) != len(labels):
            raise ValueError(
                f"contexts and labels length mismatch at position {position}"
            )
        sample_doc_ids: list[int] = []
        for context_index, context_text in enumerate(contexts):
            if not isinstance(context_text, str):
                raise TypeError("PubMedQA context text must be a string")
            retrieval_text = context_text.strip()
            if not retrieval_text:
                raise ValueError("PubMedQA context text must be non-empty after strip")
            document_id = len(records)
            source_id = pubmedqa_section_source_document_id(pubid, context_index)
            record = CorpusRecord(
                document_id=document_id,
                source_document_id=source_id,
                title=None,
                text=retrieval_text,
                retrieval_content=retrieval_text,
                corpus_position=document_id,
            )
            records.append(record)
            sample_doc_ids.append(document_id)
            content_hash = document_content_sha256(retrieval_text)
            entries.append(
                CorpusManifestEntry(
                    position=document_id,
                    doc_id=document_id,
                    source_document_id=source_id,
                    title_sha256=None,
                    text_sha256=content_hash,
                    retrieval_content_sha256=content_hash,
                )
            )
        grouped_ids.append(tuple(sample_doc_ids))

    if len(records) != expected_document_count:
        raise ValueError(
            f"expected {expected_document_count} PubMedQA documents, "
            f"found {len(records)}"
        )
    corpus_manifest = CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source=PUBMEDQA_SOURCE,
        config=PUBMEDQA_CONFIG,
        revision=PUBMEDQA_REVISION,
        split=PUBMEDQA_SPLIT,
        construction_algorithm=PUBMEDQA_CONSTRUCTION_ALGORITHM,
        input_sample_manifest_id=sample_manifest.manifest_id,
        input_sample_manifest_sha256=sample_manifest.sha256,
        dependencies=(),
        rng_family=None,
        sampling_seed=None,
        rng_state_semantics=None,
        requested_negatives_per_query=None,
        negative_sampling_scope=None,
        negative_exclusion_scope=None,
        negative_sampling_without_replacement=None,
        final_source_id_ordering=None,
        entries=tuple(entries),
    )
    corpus_records = tuple(records)
    validate_corpus_records_against_manifest(corpus_manifest, corpus_records)
    corpus_provenance_from_corpus_manifest(
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        dataset_provenance=dataset_provenance_from_sample_manifest(sample_manifest),
    )
    return PubMedQACorpusBuild(
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        gold_document_ids_by_sample=tuple(grouped_ids),
    )


def verify_pubmedqa_historical_gold_assignments(
    build: PubMedQACorpusBuild,
    historical_artifact: Path,
    *,
    expected_sample_count: int = PUBMEDQA_EXPECTED_SAMPLE_COUNT,
) -> HistoricalCompatibility:
    """Check only the sequential gold-ID evidence retained by Sprint 1 CSV."""
    with historical_artifact.open("r", encoding="utf-8", newline="") as handle:
        historical_rows = list(csv.DictReader(handle))
    if len(historical_rows) != expected_sample_count:
        raise ValueError("historical PubMedQA sample count mismatch")
    if len(build.gold_document_ids_by_sample) != expected_sample_count:
        raise ValueError("constructed PubMedQA sample count mismatch")
    flattened: list[int] = []
    for position, (historical, expected_ids) in enumerate(
        zip(historical_rows, build.gold_document_ids_by_sample, strict=True)
    ):
        if historical.get("qa_id") != str(position):
            raise ValueError(f"historical qa_id mismatch at position {position}")
        try:
            historical_ids = ast.literal_eval(historical["gold_doc_ids"])
        except (KeyError, SyntaxError, ValueError) as error:
            raise ValueError("historical gold_doc_ids is malformed") from error
        if not isinstance(historical_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in historical_ids
        ):
            raise ValueError("historical gold_doc_ids must be a list of integers")
        if tuple(historical_ids) != expected_ids:
            raise ValueError(
                f"historical gold document grouping mismatch at position {position}"
            )
        flattened.extend(historical_ids)
    if flattened != list(range(len(build.corpus_records))):
        raise ValueError("historical gold IDs do not cover the sequential corpus")
    return HistoricalCompatibility(
        sample_count=len(historical_rows),
        document_count=len(build.corpus_records),
        gold_assignment_count=len(flattened),
    )


def corpus_manifest_from_scientific_payload(payload: Mapping[str, Any]) -> CorpusManifest:
    """Reconstruct and validate a CorpusManifest from its scientific payload."""
    return CorpusManifest(
        schema_version=payload["schema_version"],
        dataset_id=DatasetId(payload["dataset_id"]),
        source=payload["source"],
        config=payload["config"],
        revision=payload["revision"],
        split=payload["split"],
        construction_algorithm=payload["construction_algorithm"],
        input_sample_manifest_id=payload["input_sample_manifest_id"],
        input_sample_manifest_sha256=payload["input_sample_manifest_sha256"],
        dependencies=(),
        rng_family=payload["rng_family"],
        sampling_seed=payload["sampling_seed"],
        rng_state_semantics=payload["rng_state_semantics"],
        requested_negatives_per_query=payload["requested_negatives_per_query"],
        negative_sampling_scope=payload["negative_sampling_scope"],
        negative_exclusion_scope=payload["negative_exclusion_scope"],
        negative_sampling_without_replacement=payload[
            "negative_sampling_without_replacement"
        ],
        final_source_id_ordering=payload["final_source_id_ordering"],
        entries=tuple(CorpusManifestEntry(**entry) for entry in payload["entries"]),
    )


def corpus_manifest_artifact_payload(manifest: CorpusManifest) -> dict[str, Any]:
    return {
        "artifact_format": CORPUS_MANIFEST_ARTIFACT_FORMAT,
        "corpus_manifest_id": manifest.corpus_manifest_id,
        "corpus_manifest_sha256": manifest.sha256,
        "provenance_note": PUBMEDQA_PROVENANCE_NOTE,
        "scientific_payload": manifest.scientific_payload(),
    }


def write_corpus_manifest_artifact(manifest: CorpusManifest, path: Path) -> None:
    """Atomically write deterministic corpus-manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        corpus_manifest_artifact_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def read_and_verify_corpus_manifest_artifact(path: Path) -> CorpusManifest:
    with path.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored.get("artifact_format") != CORPUS_MANIFEST_ARTIFACT_FORMAT:
        raise ValueError("unexpected corpus-manifest artifact format")
    if stored.get("provenance_note") != PUBMEDQA_PROVENANCE_NOTE:
        raise ValueError("unexpected corpus-manifest provenance note")
    payload = stored.get("scientific_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("scientific_payload must be an object")
    manifest = corpus_manifest_from_scientific_payload(payload)
    if manifest.scientific_payload() != payload:
        raise ValueError("stored scientific payload is not canonical")
    if stored.get("corpus_manifest_id") != manifest.corpus_manifest_id:
        raise ValueError("stored corpus_manifest_id does not match payload")
    if stored.get("corpus_manifest_sha256") != manifest.sha256:
        raise ValueError("stored corpus_manifest_sha256 does not match payload")
    return manifest


def load_frozen_pubmedqa_corpus_manifest(path: Path) -> CorpusManifest:
    """Load and verify the exact committed PubMedQA CorpusManifest artifact."""
    manifest = read_and_verify_corpus_manifest_artifact(path)
    expected = {
        "dataset_id": DatasetId.PUBMEDQA,
        "source": PUBMEDQA_SOURCE,
        "config": PUBMEDQA_CONFIG,
        "revision": PUBMEDQA_REVISION,
        "split": PUBMEDQA_SPLIT,
        "construction_algorithm": PUBMEDQA_CONSTRUCTION_ALGORITHM,
        "corpus_manifest_id": PUBMEDQA_CORPUS_MANIFEST_ID,
        "sha256": PUBMEDQA_CORPUS_MANIFEST_SHA256,
        "document_count": PUBMEDQA_EXPECTED_DOCUMENT_COUNT,
    }
    for name, value in expected.items():
        if getattr(manifest, name) != value:
            raise ValueError(f"frozen PubMedQA CorpusManifest {name} mismatch")
    return manifest


def prepare_pubmedqa_runtime_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_manifest: SampleManifest,
    frozen_corpus_manifest: CorpusManifest,
    expected_sample_count: int = PUBMEDQA_EXPECTED_SAMPLE_COUNT,
    expected_document_count: int = PUBMEDQA_EXPECTED_DOCUMENT_COUNT,
) -> ValidatedPubMedQARuntimeCorpus:
    """Reconstruct PubMedQA and require exact frozen corpus identity."""
    ordered_rows = tuple(rows)
    build = build_pubmedqa_corpus_manifest(
        ordered_rows,
        sample_manifest,
        expected_sample_count=expected_sample_count,
        expected_document_count=expected_document_count,
    )
    if build.corpus_manifest != frozen_corpus_manifest:
        raise ValueError(
            "reconstructed PubMedQA corpus does not match frozen CorpusManifest"
        )
    validate_corpus_records_against_manifest(
        frozen_corpus_manifest, build.corpus_records
    )
    dataset_provenance = dataset_provenance_from_sample_manifest(sample_manifest)
    corpus_provenance = corpus_provenance_from_corpus_manifest(
        corpus_manifest=frozen_corpus_manifest,
        corpus_records=build.corpus_records,
        dataset_provenance=dataset_provenance,
    )
    ordered_queries = []
    for entry, row in zip(
        sample_manifest.entries, ordered_rows, strict=True
    ):
        query_text = row["question"]
        verify_manifest_sample(
            sample_manifest,
            sample_id=entry.sample_id,
            query_text=query_text,
        )
        ordered_queries.append(
            ValidatedPubMedQAQuery(entry.position, entry.sample_id, query_text)
        )
    return ValidatedPubMedQARuntimeCorpus(
        sample_manifest=sample_manifest,
        corpus_manifest=frozen_corpus_manifest,
        corpus_records=build.corpus_records,
        dataset_provenance=dataset_provenance,
        corpus_provenance=corpus_provenance,
        ordered_queries=tuple(ordered_queries),
    )


def load_validated_pubmedqa_runtime_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_manifest_path: Path = PUBMEDQA_SAMPLE_MANIFEST_PATH,
    corpus_manifest_path: Path = PUBMEDQA_OUTPUT_PATH,
) -> ValidatedPubMedQARuntimeCorpus:
    """Load both committed manifests and validate already-loaded pinned rows."""
    sample_manifest = load_frozen_pubmedqa_sample_manifest(sample_manifest_path)
    corpus_manifest = load_frozen_pubmedqa_corpus_manifest(corpus_manifest_path)
    return prepare_pubmedqa_runtime_corpus(
        rows,
        sample_manifest=sample_manifest,
        frozen_corpus_manifest=corpus_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, default=PUBMEDQA_SAMPLE_MANIFEST_PATH)
    parser.add_argument("--historical-artifact", type=Path, default=PUBMEDQA_HISTORICAL_ARTIFACT)
    parser.add_argument("--output", type=Path, default=PUBMEDQA_OUTPUT_PATH)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    manifest = load_frozen_pubmedqa_sample_manifest(args.sample_manifest)
    rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
    )
    build = build_pubmedqa_corpus_manifest(rows, manifest)
    compatibility = verify_pubmedqa_historical_gold_assignments(
        build, args.historical_artifact
    )
    write_corpus_manifest_artifact(build.corpus_manifest, args.output)
    reconstructed = read_and_verify_corpus_manifest_artifact(args.output)
    if reconstructed != build.corpus_manifest:
        raise ValueError("round-trip CorpusManifest differs")
    print(f"corpus_manifest_id={reconstructed.corpus_manifest_id}")
    print(f"corpus_manifest_sha256={reconstructed.sha256}")
    print(f"actual_document_count={reconstructed.document_count}")
    print(f"historical_gold_assignments={compatibility.gold_assignment_count}")


if __name__ == "__main__":
    main()
