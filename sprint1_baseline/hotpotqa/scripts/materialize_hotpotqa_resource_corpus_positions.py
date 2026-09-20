#!/usr/bin/env python3
"""Materialize deterministic nested HotpotQA resource-pilot document subsets.

This is RESOURCE FEASIBILITY ONLY. It never loads qrels, answers, protected
queries, correctness metrics, or retrieval-effectiveness information.

Selection:
  1. Start from the complete pinned BEIR HotpotQA corpus.
  2. Rank documents by ascending
       SHA256("hotpotqa-resource-pilot-doc-v1|" + source_document_id)
     with source_document_id as deterministic tie-break.
  3. Take prefixes of 100k, 500k, and 1M documents.
  4. Restore original canonical corpus order inside each selected subset.

Therefore the subsets are nested:
    100k ⊂ 500k ⊂ 1M
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Iterable, Mapping, Any

import numpy as np


SOURCE = "BeIR/hotpotqa"
CONFIG = "corpus"
REVISION = "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b"
SPLIT = "corpus"

EXPECTED_FULL_DOCUMENT_COUNT = 5_233_329

SELECTION_NAMESPACE = "hotpotqa-resource-pilot-doc-v1"
PILOT_SIZES = (100_000, 500_000, 1_000_000)
MAX_PILOT_SIZE = max(PILOT_SIZES)


def selection_digest(source_document_id: str) -> bytes:
    if not isinstance(source_document_id, str) or not source_document_id:
        raise ValueError("source_document_id must be a non-empty string")
    return hashlib.sha256(
        f"{SELECTION_NAMESPACE}|{source_document_id}".encode("utf-8")
    ).digest()


def selection_key(item: tuple[int, str]) -> tuple[bytes, str]:
    _, source_document_id = item
    return selection_digest(source_document_id), source_document_id


def select_largest_pilot(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
    max_pilot_size: int = MAX_PILOT_SIZE,
) -> list[tuple[int, str]]:
    """Return max pilot documents in deterministic hash-rank order."""

    seen = 0

    def items():
        nonlocal seen
        for position, row in enumerate(rows):
            if "_id" not in row:
                raise ValueError(
                    f"corpus row {position} is missing _id"
                )

            source_document_id = row["_id"]

            if not isinstance(source_document_id, str):
                raise TypeError(
                    f"corpus _id at position {position} must be a string"
                )
            if not source_document_id:
                raise ValueError(
                    f"corpus _id at position {position} must be non-empty"
                )

            seen += 1
            yield position, source_document_id

    selected = heapq.nsmallest(
        max_pilot_size,
        items(),
        key=selection_key,
    )

    if seen != expected_count:
        raise ValueError(
            f"full corpus count mismatch: expected {expected_count:,}, "
            f"found {seen:,}"
        )

    if len(selected) != max_pilot_size:
        raise RuntimeError(
            f"expected {max_pilot_size:,} selected documents, "
            f"found {len(selected):,}"
        )

    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_position_stream_sha256(
    selected: list[tuple[int, str]],
) -> str:
    digest = hashlib.sha256()
    for position, _ in selected:
        digest.update(f"{position}\n".encode("utf-8"))
    return digest.hexdigest()


def _canonical_id_stream_sha256(
    selected: list[tuple[int, str]],
) -> str:
    digest = hashlib.sha256()
    for position, source_document_id in selected:
        digest.update(
            json.dumps(
                {
                    "source_position": position,
                    "source_document_id": source_document_id,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_rank_stream_sha256(
    selected_in_hash_order: list[tuple[int, str]],
) -> str:
    digest = hashlib.sha256()
    for position, source_document_id in selected_in_hash_order:
        digest.update(
            json.dumps(
                {
                    "selection_sha256": selection_digest(
                        source_document_id
                    ).hex(),
                    "source_document_id": source_document_id,
                    "source_position": position,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(
                f"existing immutable artifact differs: {path}"
            )
        return

    path.write_bytes(data)


def materialize(
    rows,
    output_dir: Path,
) -> None:
    selected_max = select_largest_pilot(
        rows,
        expected_count=EXPECTED_FULL_DOCUMENT_COUNT,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for size in PILOT_SIZES:
        hash_rank_subset = selected_max[:size]

        # Retrieval corpus must use original source corpus order.
        canonical_subset = sorted(
            hash_rank_subset,
            key=lambda item: item[0],
        )

        positions = np.asarray(
            [position for position, _ in canonical_subset],
            dtype=np.int64,
        )

        positions_path = (
            output_dir
            / f"hotpotqa_resource_corpus_{size}_positions.npy"
        )

        import io

        buffer = io.BytesIO()
        np.save(buffer, positions, allow_pickle=False)
        _write_immutable(positions_path, buffer.getvalue())

        manifest = {
            "artifact_format":
                "hotpotqa.resource-corpus-position-manifest.v1",
            "dataset": "hotpotqa",
            "evidence_role": "RESOURCE_PILOT",
            "source": SOURCE,
            "config": CONFIG,
            "revision": REVISION,
            "split": SPLIT,
            "full_document_count":
                EXPECTED_FULL_DOCUMENT_COUNT,
            "selected_document_count": size,
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_rule": (
                'ascending SHA256(namespace + "|" + '
                "source_document_id), source_document_id tie-break"
            ),
            "nested_prefix_rule": (
                "100k, 500k and 1M are prefixes of one global "
                "hash-ranked document ordering"
            ),
            "retriever_order_rule": (
                "selected documents restored to original canonical "
                "source corpus position order"
            ),
            "canonical_position_stream_sha256":
                _canonical_position_stream_sha256(
                    canonical_subset
                ),
            "canonical_source_id_stream_sha256":
                _canonical_id_stream_sha256(
                    canonical_subset
                ),
            "selection_rank_stream_sha256":
                _hash_rank_stream_sha256(
                    hash_rank_subset
                ),
            "selection_cutoff_sha256":
                selection_digest(
                    hash_rank_subset[-1][1]
                ).hex(),
            "positions_artifact": {
                "filename": positions_path.name,
                "dtype": "int64",
                "shape": [size],
                "physical_sha256":
                    _sha256_file(positions_path),
            },
        }

        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

        manifest_path = (
            output_dir
            / f"hotpotqa_resource_corpus_{size}_manifest.json"
        )

        _write_immutable(
            manifest_path,
            manifest_bytes,
        )

        print(
            f"{size}:",
            manifest_path,
            manifest["canonical_position_stream_sha256"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()

    rows = load_dataset(
        SOURCE,
        CONFIG,
        revision=REVISION,
        split=SPLIT,
        cache_dir=str(args.cache_dir),
    )

    if len(rows) != EXPECTED_FULL_DOCUMENT_COUNT:
        raise ValueError(
            "loaded HotpotQA corpus does not have exactly "
            f"{EXPECTED_FULL_DOCUMENT_COUNT:,} documents"
        )

    materialize(
        rows,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
