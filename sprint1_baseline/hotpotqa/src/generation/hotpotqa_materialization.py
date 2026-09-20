"""Compatibility identities for full-corpus HotpotQA generation materialization.

The canonical full-corpus HotpotQA retrieval runners emit aggregate JSONL
candidate files rather than the repository's older per-query CandidateArtifact
files.  This module creates deterministic scientific identities for those
validated aggregate rows without rerunning retrieval or fabricating legacy
production provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from generation._io import stable_json_sha256
from retrieval_artifacts.contracts import canonical_stable_id
from run_registry import (
    candidate_set_artifact_payload,
    candidate_set_scientific_payload,
)


HOTPOTQA_AGGREGATE_CANDIDATE_SCHEMA_VERSION = (
    "sprint3.hotpotqa-aggregate-candidate-compat.v1"
)

HOTPOTQA_DATASET = "hotpotqa"
HOTPOTQA_EVIDENCE_ROLE = "OFFICIAL_TEST_FULL"

HOTPOTQA_QUERY_MANIFEST_SHA256 = (
    "efedb99b0bf844896611f37d19bebcf710b171a375c8c9fb932c055b7aa16e9c"
)
HOTPOTQA_SAMPLE_MANIFEST_ID = (
    "sample-manifest:sha256:"
    + HOTPOTQA_QUERY_MANIFEST_SHA256
)

HOTPOTQA_STREAMING_CORPUS_SCIENTIFIC_SHA256 = (
    "ac98c20e24668bb886df75382baff8edf3944f294d1fd0dc8609e58f53f739c9"
)
HOTPOTQA_CORPUS_MANIFEST_ID = (
    "corpus-manifest:sha256:"
    + HOTPOTQA_STREAMING_CORPUS_SCIENTIFIC_SHA256
)

HOTPOTQA_EXPECTED_QUERY_COUNT = 7_405
HOTPOTQA_CANDIDATE_POOL = 20
HOTPOTQA_TOP_K = 5

CANONICAL_HOTPOTQA_RETRIEVERS = (
    "bm25",
    "dpr",
    "contriever",
    "colbertv2",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def canonical_hotpotqa_retriever(value: str) -> str:
    """Return the canonical generation/registry retriever name."""
    if value == "colbert":
        return "colbertv2"
    if value not in CANONICAL_HOTPOTQA_RETRIEVERS:
        raise ValueError(f"unsupported HotpotQA retriever: {value!r}")
    return value


def aggregate_candidate_scientific_payload(
    *,
    sample_id: str | int,
    source_query_id: str,
    query_text_sha256: str,
    retriever: str,
    retriever_config_sha256: str,
    aggregate_candidate_file_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one stable scientific identity for an aggregate retrieval row."""
    sample = canonical_stable_id(sample_id, "sample_id")

    if not isinstance(source_query_id, str) or not source_query_id:
        raise ValueError("source_query_id must be a non-empty string")

    _require_sha256(query_text_sha256, "query_text_sha256")
    _require_sha256(
        retriever_config_sha256,
        "retriever_config_sha256",
    )
    _require_sha256(
        aggregate_candidate_file_sha256,
        "aggregate_candidate_file_sha256",
    )

    retriever_name = canonical_hotpotqa_retriever(retriever)

    values = [dict(item) for item in candidates]

    if len(values) != HOTPOTQA_CANDIDATE_POOL:
        raise ValueError(
            "HotpotQA aggregate candidate row must contain exactly 20 candidates"
        )

    canonical_candidates: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()
    seen_positions: set[int] = set()

    for expected_rank, item in enumerate(values, start=1):
        required = {
            "rank",
            "document_id",
            "corpus_position",
            "native_score_hex",
            "document_content_sha256",
        }
        if set(item) != required:
            raise ValueError(
                "aggregate candidate fields differ from frozen HotpotQA schema"
            )

        if item["rank"] != expected_rank:
            raise ValueError("aggregate candidate ranks must be exactly 1..20")

        document_id = canonical_stable_id(
            item["document_id"],
            "document_id",
        )

        corpus_position = item["corpus_position"]
        if (
            isinstance(corpus_position, bool)
            or not isinstance(corpus_position, int)
            or corpus_position < 0
        ):
            raise ValueError(
                "candidate corpus_position must be a nonnegative integer"
            )

        document_key = repr(document_id)
        if document_key in seen_document_ids:
            raise ValueError("candidate document IDs must be unique")
        seen_document_ids.add(document_key)

        if corpus_position in seen_positions:
            raise ValueError("candidate corpus positions must be unique")
        seen_positions.add(corpus_position)

        native_score_hex = item["native_score_hex"]
        if not isinstance(native_score_hex, str):
            raise TypeError("native_score_hex must be a string")

        # Validate that it is a finite Python float encoding.
        score = float.fromhex(native_score_hex)
        if not (-float("inf") < score < float("inf")):
            raise ValueError("candidate native score must be finite")

        content_sha = _require_sha256(
            item["document_content_sha256"],
            "document_content_sha256",
        )

        canonical_candidates.append(
            {
                "rank": expected_rank,
                "document_id": document_id,
                "corpus_position": corpus_position,
                "native_score_hex": native_score_hex,
                "document_content_sha256": content_sha,
            }
        )

    return {
        "schema_version": HOTPOTQA_AGGREGATE_CANDIDATE_SCHEMA_VERSION,
        "dataset": HOTPOTQA_DATASET,
        "evidence_role": HOTPOTQA_EVIDENCE_ROLE,
        "sample_manifest_id": HOTPOTQA_SAMPLE_MANIFEST_ID,
        "corpus_manifest_id": HOTPOTQA_CORPUS_MANIFEST_ID,
        "sample_id": sample,
        "source_query_id": source_query_id,
        "query_text_sha256": query_text_sha256,
        "retriever": retriever_name,
        "retriever_config_sha256": retriever_config_sha256,
        "aggregate_candidate_file_sha256": (
            aggregate_candidate_file_sha256
        ),
        "candidate_pool": HOTPOTQA_CANDIDATE_POOL,
        "candidates": canonical_candidates,
    }


def aggregate_candidate_artifact_id(
    *,
    sample_id: str | int,
    source_query_id: str,
    query_text_sha256: str,
    retriever: str,
    retriever_config_sha256: str,
    aggregate_candidate_file_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    """Return the stable compatibility candidate ID for one aggregate row."""
    payload = aggregate_candidate_scientific_payload(
        sample_id=sample_id,
        source_query_id=source_query_id,
        query_text_sha256=query_text_sha256,
        retriever=retriever,
        retriever_config_sha256=retriever_config_sha256,
        aggregate_candidate_file_sha256=aggregate_candidate_file_sha256,
        candidates=candidates,
    )
    return f"candidate:sha256:{stable_json_sha256(payload)}"


def build_hotpotqa_candidate_set(
    *,
    retriever: str,
    entries: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical aggregate candidate-set wrapper."""
    retriever_name = canonical_hotpotqa_retriever(retriever)

    scientific = candidate_set_scientific_payload(
        dataset=HOTPOTQA_DATASET,
        evidence_role=HOTPOTQA_EVIDENCE_ROLE,
        sample_manifest_id=HOTPOTQA_SAMPLE_MANIFEST_ID,
        retriever=retriever_name,
        expected_query_count=HOTPOTQA_EXPECTED_QUERY_COUNT,
        entries=entries,
    )

    return candidate_set_artifact_payload(
        scientific,
        provenance=provenance,
    )


def build_hotpotqa_relevance_selected_context(
    *,
    sample_id: str | int,
    source_query_id: str,
    query_text: str,
    query_text_sha256: str,
    retriever: str,
    retriever_config_sha256: str,
    aggregate_candidate_file_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
    candidate_set_id: str,
    passage_bodies: Sequence[str],
):
    """Build relevance-only HotpotQA Top-5 context from a validated aggregate row.

    ``passage_bodies`` must be the canonical BEIR ``text`` fields for ranks 1..5.
    Titles are deliberately excluded from generation context.
    """
    from generation._io import sha256_text
    from generation.prompts import render_context_block
    from generation.selected_context import (
        SelectedContextArtifact,
        SelectedPassage,
    )

    if not isinstance(query_text, str) or not query_text:
        raise ValueError("query_text must be a non-empty string")

    if sha256_text(query_text) != query_text_sha256:
        raise ValueError("query_text_sha256 does not match query_text")

    if len(passage_bodies) != HOTPOTQA_TOP_K:
        raise ValueError("HotpotQA selected context requires exactly five bodies")

    compatibility_candidate_id = aggregate_candidate_artifact_id(
        sample_id=sample_id,
        source_query_id=source_query_id,
        query_text_sha256=query_text_sha256,
        retriever=retriever,
        retriever_config_sha256=retriever_config_sha256,
        aggregate_candidate_file_sha256=aggregate_candidate_file_sha256,
        candidates=candidates,
    )

    canonical_candidates = aggregate_candidate_scientific_payload(
        sample_id=sample_id,
        source_query_id=source_query_id,
        query_text_sha256=query_text_sha256,
        retriever=retriever,
        retriever_config_sha256=retriever_config_sha256,
        aggregate_candidate_file_sha256=aggregate_candidate_file_sha256,
        candidates=candidates,
    )["candidates"]

    selected = []

    for expected_rank, (candidate, raw_body) in enumerate(
        zip(
            canonical_candidates[:HOTPOTQA_TOP_K],
            passage_bodies,
            strict=True,
        ),
        start=1,
    ):
        if candidate["rank"] != expected_rank:
            raise ValueError("Top-5 candidate ranks must be exactly 1..5")

        if not isinstance(raw_body, str):
            raise TypeError("passage body must be a string")

        body = raw_body.strip()

        if not body:
            raise ValueError("passage body must be non-empty")

        if body != raw_body:
            raise ValueError(
                "passage body must already have canonical outer whitespace"
            )

        # Full-corpus HotpotQA retrieval defines doc_id = source BEIR _id.
        document_id = candidate["document_id"]

        selected.append(
            SelectedPassage(
                rank=expected_rank,
                document_id=document_id,
                source_document_id=document_id,
                corpus_position=candidate["corpus_position"],
                candidate_document_content_sha256=(
                    candidate["document_content_sha256"]
                ),
                passage_body=body,
                passage_body_sha256=sha256_text(body),
            )
        )

    bodies = tuple(item.passage_body for item in selected)

    return SelectedContextArtifact(
        dataset=HOTPOTQA_DATASET,
        evidence_role=HOTPOTQA_EVIDENCE_ROLE,
        sample_manifest_id=HOTPOTQA_SAMPLE_MANIFEST_ID,
        corpus_manifest_id=HOTPOTQA_CORPUS_MANIFEST_ID,
        sample_id=sample_id,
        query_text=query_text,
        retriever=canonical_hotpotqa_retriever(retriever),
        candidate_set_id=candidate_set_id,
        candidate_artifact_id=compatibility_candidate_id,
        passages=tuple(selected),
        context_block=render_context_block(bodies),
    )


def build_hotpotqa_selected_context_set(
    *,
    retriever: str,
    candidate_set_id: str,
    entries: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical HotpotQA relevance-only selected-context set."""
    from generation.selected_context import (
        selected_context_set_payload,
        selected_context_set_wrapper,
    )

    retriever_name = canonical_hotpotqa_retriever(retriever)

    scientific = selected_context_set_payload(
        dataset=HOTPOTQA_DATASET,
        evidence_role=HOTPOTQA_EVIDENCE_ROLE,
        sample_manifest_id=HOTPOTQA_SAMPLE_MANIFEST_ID,
        retriever=retriever_name,
        candidate_set_id=candidate_set_id,
        entries=entries,
    )

    if scientific["expected_query_count"] != HOTPOTQA_EXPECTED_QUERY_COUNT:
        raise ValueError(
            "HotpotQA selected-context set must contain exactly 7,405 queries"
        )

    return selected_context_set_wrapper(
        scientific,
        provenance=provenance,
    )
