#!/usr/bin/env python3
"""Materialize the frozen HotpotQA 220-query resource-pilot manifest.

The BEIR train qrels are used only to identify official DEVELOPMENT query
membership through the ``query-id`` field. Corpus IDs and relevance scores are
never used by this resource-pilot selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

QUERY_SOURCE = "BeIR/hotpotqa"
QUERY_CONFIG = "queries"
QUERY_REVISION = "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b"
QUERY_SPLIT = "queries"

DEVELOPMENT_MEMBERSHIP_SOURCE = "BeIR/hotpotqa-qrels"
DEVELOPMENT_MEMBERSHIP_CONFIG = "default"
DEVELOPMENT_MEMBERSHIP_REVISION = (
    "b15429e9244c8ec966985d7778427c3b1543b314"
)
DEVELOPMENT_MEMBERSHIP_SPLIT = "train"

EXPECTED_DEVELOPMENT_QUERY_COUNT = 85_000
RESOURCE_QUERY_COUNT = 220
WARMUP_QUERY_COUNT = 20
MEASURED_QUERY_COUNT = 200

SELECTION_NAMESPACE = "hotpotqa-resource-pilot-v1"

DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "artifacts/resource_pilots/hotpotqa_resource_queries_v1.json"
)


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _selection_digest(query_id: str) -> bytes:
    return hashlib.sha256(
        f"{SELECTION_NAMESPACE}|{query_id}".encode("utf-8")
    ).digest()


def _selection_key(query_id: str) -> tuple[bytes, str]:
    return _selection_digest(query_id), query_id


def select_resource_queries(
    query_rows: Iterable[Mapping[str, Any]],
    membership_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the frozen 220 DEVELOPMENT queries using query ID only."""

    query_text_by_id: dict[str, str] = {}

    for row in query_rows:
        if "_id" not in row or "text" not in row:
            raise ValueError("pinned query row must contain _id and text")

        query_id = _require_nonempty_string(row["_id"], "query _id")
        question = _require_nonempty_string(
            row["text"],
            f"question text for query ID {query_id!r}",
        )

        if query_id in query_text_by_id:
            raise ValueError(f"duplicate pinned query ID: {query_id!r}")

        query_text_by_id[query_id] = question

    development_ids: set[str] = set()

    for row in membership_rows:
        if "query-id" not in row:
            raise ValueError(
                "pinned DEVELOPMENT membership row must contain query-id"
            )

        development_ids.add(
            _require_nonempty_string(
                row["query-id"],
                "DEVELOPMENT membership query-id",
            )
        )

    if len(development_ids) != EXPECTED_DEVELOPMENT_QUERY_COUNT:
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_DEVELOPMENT_QUERY_COUNT:,} unique DEVELOPMENT "
            f"query IDs, found {len(development_ids):,}"
        )

    missing = development_ids.difference(query_text_by_id)
    if missing:
        raise ValueError(
            f"{len(missing)} DEVELOPMENT query IDs are missing from "
            "the pinned query source"
        )

    selected_ids = sorted(development_ids, key=_selection_key)[
        :RESOURCE_QUERY_COUNT
    ]

    if len(selected_ids) != RESOURCE_QUERY_COUNT:
        raise RuntimeError("failed to select exactly 220 resource queries")

    entries: list[dict[str, Any]] = []

    for position, query_id in enumerate(selected_ids):
        role = "warmup" if position < WARMUP_QUERY_COUNT else "measured"

        entries.append(
            {
                "position": position,
                "query_id": query_id,
                "role": role,
                "question": query_text_by_id[query_id],
                "selection_sha256": _selection_digest(query_id).hex(),
            }
        )

    if sum(e["role"] == "warmup" for e in entries) != WARMUP_QUERY_COUNT:
        raise RuntimeError("warm-up query count mismatch")

    if sum(e["role"] == "measured" for e in entries) != MEASURED_QUERY_COUNT:
        raise RuntimeError("measured query count mismatch")

    return entries


def build_manifest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(entries) != RESOURCE_QUERY_COUNT:
        raise ValueError("manifest requires exactly 220 query entries")

    return {
        "artifact_format": "hotpotqa.resource-query-manifest.v1",
        "dataset": "hotpotqa",
        "evidence_role": "DEVELOPMENT",
        "query_source": QUERY_SOURCE,
        "query_config": QUERY_CONFIG,
        "query_revision": QUERY_REVISION,
        "query_split": QUERY_SPLIT,
        "development_membership_source": DEVELOPMENT_MEMBERSHIP_SOURCE,
        "development_membership_config": DEVELOPMENT_MEMBERSHIP_CONFIG,
        "development_membership_revision": DEVELOPMENT_MEMBERSHIP_REVISION,
        "development_membership_split": DEVELOPMENT_MEMBERSHIP_SPLIT,
        "development_membership_semantics": (
            "query-id field only; corpus-id and score ignored"
        ),
        "development_query_count": EXPECTED_DEVELOPMENT_QUERY_COUNT,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_rule": (
            'ascending SHA256(namespace + "|" + query_id), '
            "query_id deterministic tie-break"
        ),
        "selected_query_count": RESOURCE_QUERY_COUNT,
        "warmup_count": WARMUP_QUERY_COUNT,
        "measured_count": MEASURED_QUERY_COUNT,
        "candidate_pool": 20,
        "entries": entries,
    }


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def write_manifest(
    output_path: Path,
    payload: Mapping[str, Any],
) -> str:
    data = _canonical_bytes(payload)

    if output_path.exists():
        existing = output_path.read_bytes()
        if existing != data:
            raise FileExistsError(
                f"existing immutable artifact differs: {output_path}"
            )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)

    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    cache_dir = None if args.cache_dir is None else str(args.cache_dir)

    query_rows = load_dataset(
        QUERY_SOURCE,
        QUERY_CONFIG,
        revision=QUERY_REVISION,
        split=QUERY_SPLIT,
        cache_dir=cache_dir,
    )

    membership_rows = load_dataset(
        DEVELOPMENT_MEMBERSHIP_SOURCE,
        DEVELOPMENT_MEMBERSHIP_CONFIG,
        revision=DEVELOPMENT_MEMBERSHIP_REVISION,
        split=DEVELOPMENT_MEMBERSHIP_SPLIT,
        cache_dir=cache_dir,
    )

    entries = select_resource_queries(query_rows, membership_rows)
    manifest = build_manifest(entries)
    artifact_sha256 = write_manifest(args.output, manifest)

    print("output:", args.output)
    print("selected:", len(entries))
    print("warmup:", WARMUP_QUERY_COUNT)
    print("measured:", MEASURED_QUERY_COUNT)
    print("manifest_sha256:", artifact_sha256)


if __name__ == "__main__":
    main()
