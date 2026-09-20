#!/usr/bin/env python3
"""Build deterministic Sprint 3 sample-manifest artifacts.

The resulting manifests freeze revisions for the clean Sprint 3 rerun; they do
not identify the exact upstream revisions used historically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts.sample_manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleManifestEntry,
    SampleSelectionDependency,
    query_text_sha256,
)


PUBMEDQA_SOURCE = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
PUBMEDQA_SPLIT = "train"
PUBMEDQA_EXPECTED_SIZE = 1000
FULL_SPLIT_SOURCE_ORDER_ALGORITHM = "full_split_source_order.v1"
HOTPOTQA_SOURCE = "BeIR/hotpotqa"
HOTPOTQA_CONFIG = "queries"
HOTPOTQA_REVISION = "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b"
HOTPOTQA_SPLIT = "queries"
HOTPOTQA_QUERY_COUNT = 97852
HOTPOTQA_QRELS_SOURCE = "BeIR/hotpotqa-qrels"
HOTPOTQA_QRELS_CONFIG = "default"
HOTPOTQA_QRELS_REVISION = "b15429e9244c8ec966985d7778427c3b1543b314"
HOTPOTQA_QRELS_SPLIT = "test"
HOTPOTQA_SAMPLE_SIZE = 500
HOTPOTQA_SEED = 42
HOTPOTQA_SAMPLING_ALGORITHM = (
    "qrels_score_gte_1_lexsorted_random_sample_then_lexsort.v1"
)
ARTIFACT_FORMAT = "sprint3.sample-manifest-artifact.v1"
PUBMEDQA_PROVENANCE_NOTE = (
    "This manifest freezes an immutable revision for the clean Sprint 3 rerun; "
    "it does not identify the exact upstream revision used historically in Sprint 1."
)
HOTPOTQA_PROVENANCE_NOTE = (
    "This manifest freezes immutable query and qrels revisions for the clean "
    "Sprint 3 rerun and reproduces the exact historical evaluated query population; "
    "it does not identify the exact upstream revisions used historically in Sprint 2."
)
PUBMEDQA_HISTORICAL_ARTIFACT = (
    REPOSITORY_ROOT / "results/sprint1/raw/fullrag_bm25_top5.csv"
)
PUBMEDQA_OUTPUT = (
    REPOSITORY_ROOT / "artifacts/sample_manifests/pubmedqa_sample_manifest_v2.json"
)
HOTPOTQA_HISTORICAL_ARTIFACT = (
    REPOSITORY_ROOT / "results/sprint2/raw/sprint2_beir_bm25_none_top5.csv"
)
HOTPOTQA_OUTPUT = (
    REPOSITORY_ROOT / "artifacts/sample_manifests/hotpotqa_sample_manifest_v2.json"
)


@dataclass(frozen=True)
class HotpotSelection:
    selected_qids: tuple[str, ...]
    selected_query_texts: tuple[str, ...]
    qrels_row_count: int
    relevant_qrel_row_count: int
    eligible_qid_count: int


def build_full_split_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: DatasetId,
    source: str,
    config: str | None,
    revision: str,
    split: str,
    expected_size: int,
) -> SampleManifest:
    """Build a full-split manifest while preserving supplied row order exactly."""
    if len(rows) != expected_size:
        raise ValueError(f"expected {expected_size} rows, found {len(rows)}")
    entries = tuple(
        SampleManifestEntry(
            position=position,
            sample_id=position,
            source_sample_id=row["pubid"],
            query_text_sha256=query_text_sha256(row["question"]),
        )
        for position, row in enumerate(rows)
    )
    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        source=source,
        config=config,
        revision=revision,
        split=split,
        sampling_algorithm=FULL_SPLIT_SOURCE_ORDER_ALGORITHM,
        sampling_seed=None,
        requested_sample_size=None,
        selection_dependencies=(),
        entries=entries,
    )


def verify_historical_questions(
    rows: Sequence[Mapping[str, Any]],
    historical_artifact: Path,
    *,
    expected_size: int,
) -> None:
    """Require exact historical IDs and question strings in source order."""
    with historical_artifact.open("r", encoding="utf-8", newline="") as handle:
        historical_rows = list(csv.DictReader(handle))
    if len(historical_rows) != expected_size:
        raise ValueError(
            f"historical artifact must contain {expected_size} rows, "
            f"found {len(historical_rows)}"
        )
    if len(rows) != expected_size:
        raise ValueError(f"pinned dataset must contain {expected_size} rows, found {len(rows)}")
    for position, (historical, pinned) in enumerate(
        zip(historical_rows, rows, strict=True)
    ):
        if historical.get("qa_id") != str(position):
            raise ValueError(f"historical qa_id mismatch at position {position}")
        if historical.get("question") != pinned["question"]:
            raise ValueError(f"historical question mismatch at position {position}")


def select_hotpotqa_population(
    query_rows: Sequence[Mapping[str, Any]],
    qrel_rows: Sequence[Mapping[str, Any]],
    *,
    requested_size: int,
    seed: int,
    expected_query_count: int | None = None,
) -> HotpotSelection:
    """Reproduce the verified HotpotQA qrels-driven selection exactly."""
    if expected_query_count is not None and len(query_rows) != expected_query_count:
        raise ValueError(
            f"expected {expected_query_count} query rows, found {len(query_rows)}"
        )
    query_text_by_id: dict[str, str] = {}
    for row in query_rows:
        query_id = row["_id"]
        query_text = row["text"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise TypeError("pinned query _id must be a non-empty string")
        if query_id in query_text_by_id:
            raise ValueError(f"duplicate pinned query ID: {query_id!r}")
        if not isinstance(query_text, str):
            raise TypeError(f"query text must be a string for query ID {query_id!r}")
        query_text_by_id[query_id] = query_text

    relevant_qids = sorted(
        set(
            str(row["query-id"])
            for row in qrel_rows
            if row["score"] >= 1
        )
    )
    rng = random.Random(seed)
    sampled_qids = rng.sample(
        relevant_qids,
        min(requested_size, len(relevant_qids)),
    )
    sampled_qids = sorted(sampled_qids)
    missing_qids = [qid for qid in sampled_qids if qid not in query_text_by_id]
    if missing_qids:
        raise ValueError(f"selected query ID is absent from pinned queries: {missing_qids[0]!r}")
    return HotpotSelection(
        selected_qids=tuple(sampled_qids),
        selected_query_texts=tuple(query_text_by_id[qid] for qid in sampled_qids),
        qrels_row_count=len(qrel_rows),
        relevant_qrel_row_count=sum(row["score"] >= 1 for row in qrel_rows),
        eligible_qid_count=len(relevant_qids),
    )


def build_hotpotqa_manifest(
    selection: HotpotSelection,
    *,
    source: str,
    config: str,
    revision: str,
    split: str,
    qrels_dependency: SampleSelectionDependency,
    requested_size: int,
    seed: int,
) -> SampleManifest:
    entries = tuple(
        SampleManifestEntry(
            position=position,
            sample_id=position,
            source_sample_id=query_id,
            query_text_sha256=query_text_sha256(query_text),
        )
        for position, (query_id, query_text) in enumerate(
            zip(selection.selected_qids, selection.selected_query_texts, strict=True)
        )
    )
    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.HOTPOTQA,
        source=source,
        config=config,
        revision=revision,
        split=split,
        sampling_algorithm=HOTPOTQA_SAMPLING_ALGORITHM,
        sampling_seed=seed,
        requested_sample_size=requested_size,
        selection_dependencies=(qrels_dependency,),
        entries=entries,
    )


def verify_hotpotqa_historical_population(
    selection: HotpotSelection,
    historical_artifact: Path,
    *,
    expected_size: int,
) -> None:
    with historical_artifact.open("r", encoding="utf-8", newline="") as handle:
        historical_rows = list(csv.DictReader(handle))
    if len(historical_rows) != expected_size:
        raise ValueError(
            f"historical artifact must contain {expected_size} rows, "
            f"found {len(historical_rows)}"
        )
    if len(selection.selected_qids) != expected_size:
        raise ValueError(
            f"selected population must contain {expected_size} rows, "
            f"found {len(selection.selected_qids)}"
        )
    for position, (historical, query_id, query_text) in enumerate(
        zip(
            historical_rows,
            selection.selected_qids,
            selection.selected_query_texts,
            strict=True,
        )
    ):
        if historical.get("qa_id") != str(position):
            raise ValueError(f"historical qa_id mismatch at position {position}")
        if historical.get("beir_query_id") != query_id:
            raise ValueError(f"historical query ID mismatch at position {position}")
        if historical.get("question") != query_text:
            raise ValueError(f"historical question mismatch at position {position}")


def manifest_from_scientific_payload(payload: Mapping[str, Any]) -> SampleManifest:
    """Reconstruct a manifest from its deterministic scientific payload."""
    dependencies = tuple(
        SampleSelectionDependency(**dependency)
        for dependency in payload["selection_dependencies"]
    )
    entries = tuple(SampleManifestEntry(**entry) for entry in payload["entries"])
    return SampleManifest(
        schema_version=payload["schema_version"],
        dataset_id=DatasetId(payload["dataset_id"]),
        source=payload["source"],
        config=payload["config"],
        revision=payload["revision"],
        split=payload["split"],
        sampling_algorithm=payload["sampling_algorithm"],
        sampling_seed=payload["sampling_seed"],
        requested_sample_size=payload["requested_sample_size"],
        selection_dependencies=dependencies,
        entries=entries,
    )


def artifact_payload(
    manifest: SampleManifest,
    *,
    provenance_note: str = PUBMEDQA_PROVENANCE_NOTE,
) -> dict[str, Any]:
    return {
        "artifact_format": ARTIFACT_FORMAT,
        "manifest_id": manifest.manifest_id,
        "provenance_note": provenance_note,
        "scientific_payload": manifest.scientific_payload(),
        "sha256": manifest.sha256,
    }


def write_manifest_artifact(
    manifest: SampleManifest,
    output_path: Path,
    *,
    provenance_note: str = PUBMEDQA_PROVENANCE_NOTE,
) -> None:
    """Write deterministic JSON atomically after all compatibility checks pass."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        artifact_payload(manifest, provenance_note=provenance_note),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
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
    os.replace(temporary_path, output_path)


def read_and_verify_manifest_artifact(
    output_path: Path,
    *,
    expected_provenance_note: str | None = None,
) -> SampleManifest:
    """Round-trip and verify the stored manifest identity and population."""
    with output_path.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError("unexpected sample-manifest artifact format")
    if (
        expected_provenance_note is not None
        and stored.get("provenance_note") != expected_provenance_note
    ):
        raise ValueError("unexpected sample-manifest provenance note")
    manifest = manifest_from_scientific_payload(stored["scientific_payload"])
    if stored.get("manifest_id") != manifest.manifest_id:
        raise ValueError("stored manifest_id does not match scientific payload")
    if stored.get("sha256") != manifest.sha256:
        raise ValueError("stored sha256 does not match scientific payload")
    return manifest


def build_verify_and_write(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: DatasetId,
    source: str,
    config: str | None,
    revision: str,
    split: str,
    expected_size: int,
    historical_artifact: Path,
    output_path: Path,
) -> SampleManifest:
    """Apply the historical gate before writing and verify the written artifact."""
    verify_historical_questions(
        rows,
        historical_artifact,
        expected_size=expected_size,
    )
    manifest = build_full_split_manifest(
        rows,
        dataset_id=dataset_id,
        source=source,
        config=config,
        revision=revision,
        split=split,
        expected_size=expected_size,
    )
    write_manifest_artifact(manifest, output_path)
    reconstructed = read_and_verify_manifest_artifact(output_path)
    if reconstructed != manifest:
        raise ValueError("round-trip manifest differs from constructed manifest")
    if reconstructed.actual_sample_size != expected_size:
        raise ValueError("round-trip manifest has an unexpected sample size")
    return reconstructed


def build_verify_and_write_hotpotqa(
    query_rows: Sequence[Mapping[str, Any]],
    qrel_rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    config: str,
    revision: str,
    split: str,
    qrels_dependency: SampleSelectionDependency,
    requested_size: int,
    seed: int,
    expected_query_count: int | None,
    historical_artifact: Path,
    output_path: Path,
    provenance_note: str,
) -> tuple[SampleManifest, HotpotSelection]:
    selection = select_hotpotqa_population(
        query_rows,
        qrel_rows,
        requested_size=requested_size,
        seed=seed,
        expected_query_count=expected_query_count,
    )
    verify_hotpotqa_historical_population(
        selection,
        historical_artifact,
        expected_size=requested_size,
    )
    manifest = build_hotpotqa_manifest(
        selection,
        source=source,
        config=config,
        revision=revision,
        split=split,
        qrels_dependency=qrels_dependency,
        requested_size=requested_size,
        seed=seed,
    )
    write_manifest_artifact(
        manifest,
        output_path,
        provenance_note=provenance_note,
    )
    reconstructed = read_and_verify_manifest_artifact(
        output_path,
        expected_provenance_note=provenance_note,
    )
    if reconstructed != manifest:
        raise ValueError("round-trip manifest differs from constructed manifest")
    if reconstructed.actual_sample_size != requested_size:
        raise ValueError("round-trip manifest has an unexpected sample size")
    source_ids = tuple(entry.source_sample_id for entry in reconstructed.entries)
    if not all(type(source_id) is str for source_id in source_ids):
        raise TypeError("round-trip HotpotQA source IDs must all be strings")
    if len(set(source_ids)) != requested_size:
        raise ValueError("round-trip HotpotQA source IDs must be unique")
    if reconstructed.selection_dependencies != (qrels_dependency,):
        raise ValueError("round-trip HotpotQA selection dependency differs")
    return reconstructed, selection


def physical_file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("pubmedqa", "hotpotqa"),
        default="pubmedqa",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--historical-artifact",
        type=Path,
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    cache_dir = None if args.cache_dir is None else str(args.cache_dir)
    if args.dataset == "pubmedqa":
        output_path = args.output or PUBMEDQA_OUTPUT
        historical_artifact = (
            args.historical_artifact or PUBMEDQA_HISTORICAL_ARTIFACT
        )
        rows = load_dataset(
            PUBMEDQA_SOURCE,
            PUBMEDQA_CONFIG,
            split=PUBMEDQA_SPLIT,
            revision=PUBMEDQA_REVISION,
            cache_dir=cache_dir,
        )
        manifest = build_verify_and_write(
            rows,
            dataset_id=DatasetId.PUBMEDQA,
            source=PUBMEDQA_SOURCE,
            config=PUBMEDQA_CONFIG,
            revision=PUBMEDQA_REVISION,
            split=PUBMEDQA_SPLIT,
            expected_size=PUBMEDQA_EXPECTED_SIZE,
            historical_artifact=historical_artifact,
            output_path=output_path,
        )
        compatibility = "1000/1000 exact ordered questions"
        selection = None
    else:
        output_path = args.output or HOTPOTQA_OUTPUT
        historical_artifact = (
            args.historical_artifact or HOTPOTQA_HISTORICAL_ARTIFACT
        )
        query_rows = load_dataset(
            HOTPOTQA_SOURCE,
            HOTPOTQA_CONFIG,
            split=HOTPOTQA_SPLIT,
            revision=HOTPOTQA_REVISION,
            cache_dir=cache_dir,
        )
        qrel_rows = load_dataset(
            HOTPOTQA_QRELS_SOURCE,
            HOTPOTQA_QRELS_CONFIG,
            split=HOTPOTQA_QRELS_SPLIT,
            revision=HOTPOTQA_QRELS_REVISION,
            cache_dir=cache_dir,
        )
        dependency = SampleSelectionDependency(
            role="eligibility_qrels",
            source=HOTPOTQA_QRELS_SOURCE,
            config=HOTPOTQA_QRELS_CONFIG,
            revision=HOTPOTQA_QRELS_REVISION,
            split=HOTPOTQA_QRELS_SPLIT,
        )
        manifest, selection = build_verify_and_write_hotpotqa(
            query_rows,
            qrel_rows,
            source=HOTPOTQA_SOURCE,
            config=HOTPOTQA_CONFIG,
            revision=HOTPOTQA_REVISION,
            split=HOTPOTQA_SPLIT,
            qrels_dependency=dependency,
            requested_size=HOTPOTQA_SAMPLE_SIZE,
            seed=HOTPOTQA_SEED,
            expected_query_count=HOTPOTQA_QUERY_COUNT,
            historical_artifact=historical_artifact,
            output_path=output_path,
            provenance_note=HOTPOTQA_PROVENANCE_NOTE,
        )
        compatibility = "500/500 exact ordered IDs and questions"
    source_ids = tuple(entry.source_sample_id for entry in manifest.entries)
    print(f"manifest_path={output_path}")
    print(f"manifest_id={manifest.manifest_id}")
    print(f"scientific_sha256={manifest.sha256}")
    print(f"physical_artifact_sha256={physical_file_sha256(output_path)}")
    print(f"actual_sample_size={manifest.actual_sample_size}")
    print(f"unique_source_sample_ids={len(set(source_ids))}")
    print(f"source_sample_id_types={sorted({type(value).__name__ for value in source_ids})}")
    if selection is not None:
        print(f"qrels_row_count={selection.qrels_row_count}")
        print(f"relevant_qrel_row_count={selection.relevant_qrel_row_count}")
        print(f"eligible_unique_qid_count={selection.eligible_qid_count}")
    print(f"historical_compatibility={compatibility}")
    print("claim_boundary=clean Sprint 3 frozen revisions, not proven historical revisions")


if __name__ == "__main__":
    main()
