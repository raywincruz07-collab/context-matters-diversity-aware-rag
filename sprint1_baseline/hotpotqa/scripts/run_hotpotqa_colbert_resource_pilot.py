#!/usr/bin/env python3
"""Governed ColBERTv2 HotpotQA resource-feasibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

for path in (REPO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from colbert import Indexer, Searcher
from colbert.infra import (
    ColBERTConfig as StanfordColBERTConfig,
    Run,
    RunConfig,
)
from datasets import load_dataset

from retrieval_artifacts.hotpotqa_streaming_corpus_manifest import (
    HOTPOTQA_CONFIG,
    HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
    HOTPOTQA_REVISION,
    HOTPOTQA_SOURCE,
    HOTPOTQA_SPLIT,
)
from retrievers.colbert_config import COLBERT_CONFIG

from scripts.run_hotpotqa_resource_pilot import (
    CANDIDATE_POOL,
    EXPECTED_MEASURED,
    EXPECTED_WARMUP,
    build_runtime_corpus,
    directory_size,
    git_commit,
    gpu_stats,
    load_positions,
    load_queries,
    peak_rss_kib,
    percentile,
    sha256_file,
)


ALLOWED_SIZES = (100_000, 500_000, 1_000_000)

EXPECTED_CHECKPOINT_REVISION = (
    "0855eac81381e0323a846f1ed7d8452d4c648b50"
)

EXPECTED_CHECKPOINT_WEIGHTS_SHA256 = (
    "26e4c2f9f95a3da4442252bb40d99e4f"
    "bfd098e733edac1d785c9937b8a278da"
)

EXPECTED_CHECKPOINT_METADATA_SHA256 = (
    "0ddc5a54234cff6d13bc9411250a5479"
    "d9d96f3ffbace76d8a1884144377e434"
)


def validate_frozen_config() -> None:
    expected = {
        "checkpoint_id": "colbert-ir/colbertv2.0",
        "checkpoint_revision": EXPECTED_CHECKPOINT_REVISION,
        "dim": 128,
        "query_maxlen": 32,
        "doc_maxlen": 180,
        "nbits": 2,
        "kmeans_niters": 4,
        "nranks": 1,
        "index_bsize": 64,
        "amp": True,
        "search_ncells": 2,
        "search_centroid_score_threshold": 0.45,
        "search_ndocs": 1024,
        "candidate_pool_size": 20,
    }

    for field, expected_value in expected.items():
        actual = getattr(COLBERT_CONFIG, field)

        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"frozen ColBERT config mismatch for {field}: "
                f"{actual!r} != {expected_value!r}"
            )


def validate_checkpoint(checkpoint: Path) -> dict[str, str]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    weights = checkpoint / "pytorch_model.bin"
    metadata = checkpoint / "artifact.metadata"

    if not weights.is_file():
        raise FileNotFoundError(weights)

    if not metadata.is_file():
        raise FileNotFoundError(metadata)

    weights_sha = sha256_file(weights)
    metadata_sha = sha256_file(metadata)

    if weights_sha != EXPECTED_CHECKPOINT_WEIGHTS_SHA256:
        raise ValueError(
            "ColBERTv2 weights SHA256 mismatch"
        )

    if metadata_sha != EXPECTED_CHECKPOINT_METADATA_SHA256:
        raise ValueError(
            "ColBERTv2 artifact.metadata SHA256 mismatch"
        )

    return {
        "weights_sha256": weights_sha,
        "metadata_sha256": metadata_sha,
    }


def make_stanford_config(root: Path) -> StanfordColBERTConfig:
    return StanfordColBERTConfig(
        root=str(root),
        dim=COLBERT_CONFIG.dim,
        query_maxlen=COLBERT_CONFIG.query_maxlen,
        doc_maxlen=COLBERT_CONFIG.doc_maxlen,
        nbits=COLBERT_CONFIG.nbits,
        kmeans_niters=COLBERT_CONFIG.kmeans_niters,
        nranks=COLBERT_CONFIG.nranks,
        index_bsize=COLBERT_CONFIG.index_bsize,
        amp=COLBERT_CONFIG.amp,
        ncells=COLBERT_CONFIG.search_ncells,
        centroid_score_threshold=(
            COLBERT_CONFIG.search_centroid_score_threshold
        ),
        ndocs=COLBERT_CONFIG.search_ndocs,
    )


def run_queries(
    *,
    searcher: Searcher,
    queries: list[dict[str, Any]],
    pid_to_document_id: tuple[str | int, ...],
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    measured_latencies: list[float] = []
    write_seconds = 0.0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for query in queries:
            started = time.perf_counter()

            pids, _ranks, scores = searcher.search(
                query["question"],
                k=CANDIDATE_POOL,
            )

            latency = time.perf_counter() - started

            if len(pids) != CANDIDATE_POOL:
                raise ValueError(
                    "ColBERT did not return exactly 20 candidates"
                )

            if len(scores) != CANDIDATE_POOL:
                raise ValueError(
                    "ColBERT score count mismatch"
                )

            normalized_pids = tuple(
                int(pid) for pid in pids
            )

            if len(set(normalized_pids)) != CANDIDATE_POOL:
                raise ValueError(
                    "ColBERT returned duplicate PIDs"
                )

            if any(
                pid < 0 or pid >= len(pid_to_document_id)
                for pid in normalized_pids
            ):
                raise ValueError(
                    "ColBERT PID outside pilot corpus"
                )

            numeric_scores = tuple(
                float(score) for score in scores
            )

            if any(
                not np.isfinite(score)
                for score in numeric_scores
            ):
                raise ValueError(
                    "ColBERT returned non-finite score"
                )

            candidates = [
                {
                    "rank": rank,
                    "document_id": pid_to_document_id[pid],
                    "native_score": score,
                }
                for rank, (pid, score) in enumerate(
                    zip(
                        normalized_pids,
                        numeric_scores,
                        strict=True,
                    ),
                    start=1,
                )
            ]

            if query["role"] == "measured":
                measured_latencies.append(latency)

            row = {
                "position": query["position"],
                "query_id": query["query_id"],
                "role": query["role"],
                "candidates": candidates,
            }

            write_started = time.perf_counter()

            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

            handle.flush()

            write_seconds += (
                time.perf_counter() - write_started
            )

    if len(measured_latencies) != EXPECTED_MEASURED:
        raise ValueError(
            "expected exactly 200 measured ColBERT queries"
        )

    measured_seconds = sum(measured_latencies)
    candidate_bytes = output_path.stat().st_size

    return {
        "measured_query_count": len(
            measured_latencies
        ),
        "query_latency_p50_seconds": percentile(
            measured_latencies,
            50,
        ),
        "query_latency_p95_seconds": percentile(
            measured_latencies,
            95,
        ),
        "measured_query_seconds": measured_seconds,
        "measured_qps": (
            len(measured_latencies)
            / measured_seconds
        ),
        "candidate_file_bytes": candidate_bytes,
        "candidate_bytes_per_query": (
            candidate_bytes / len(queries)
        ),
        "candidate_write_seconds": write_seconds,
        "candidate_write_bytes_per_second": (
            candidate_bytes / write_seconds
            if write_seconds > 0
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--corpus-size",
        type=int,
        choices=ALLOWED_SIZES,
        required=True,
    )

    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--positions-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_started = time.perf_counter()

    validate_frozen_config()

    checkpoint_info = validate_checkpoint(
        args.checkpoint
    )

    positions, _position_manifest = load_positions(
        args.positions_dir,
        args.corpus_size,
    )

    source_started = time.perf_counter()

    rows = load_dataset(
        HOTPOTQA_SOURCE,
        HOTPOTQA_CONFIG,
        revision=HOTPOTQA_REVISION,
        split=HOTPOTQA_SPLIT,
        cache_dir=str(args.dataset_cache_dir),
    )

    if len(rows) != HOTPOTQA_EXPECTED_DOCUMENT_COUNT:
        raise ValueError(
            "canonical HotpotQA corpus count mismatch"
        )

    source_load_seconds = (
        time.perf_counter() - source_started
    )

    corpus_started = time.perf_counter()

    (
        corpus_manifest,
        corpus_records,
        corpus_stats,
    ) = build_runtime_corpus(
        rows=rows,
        positions=positions,
        corpus_size=args.corpus_size,
    )

    corpus_materialization_seconds = (
        time.perf_counter() - corpus_started
    )

    queries = load_queries(
        args.query_manifest
    )

    if sum(
        query["role"] == "warmup"
        for query in queries
    ) != EXPECTED_WARMUP:
        raise ValueError(
            "ColBERT warm-up count mismatch"
        )

    collection = [
        record.retrieval_content
        for record in corpus_records
    ]

    pid_to_document_id = tuple(
        record.document_id
        for record in corpus_records
    )

    root = (
        args.cache_root
        / "colbert"
        / str(args.corpus_size)
    )

    experiment = "hotpotqa_resource_pilot"
    index_name = "colbertv2.nbits2"

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    disk_before = directory_size(root)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    index_started = time.perf_counter()

    with Run().context(
        RunConfig(
            nranks=COLBERT_CONFIG.nranks,
            experiment=experiment,
            root=str(root),
        )
    ):
        config = make_stanford_config(root)

        indexer = Indexer(
            checkpoint=str(
                args.checkpoint.resolve()
            ),
            config=config,
        )

        index_path = Path(
            indexer.index(
                name=index_name,
                collection=collection,
                overwrite="resume",
            )
        )

    index_seconds = (
        time.perf_counter() - index_started
    )

    if not index_path.is_dir():
        raise RuntimeError(
            f"ColBERT index directory missing: {index_path}"
        )

    metadata_path = index_path / "metadata.json"

    if not metadata_path.is_file():
        raise RuntimeError(
            "ColBERT metadata.json missing"
        )

    index_size_bytes = directory_size(
        index_path
    )

    disk_after_index = directory_size(
        root
    )

    output_dir = (
        args.output_root
        / "colbert"
        / str(args.corpus_size)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = (
        output_dir / "candidates_top20.jsonl"
    )

    search_load_started = time.perf_counter()

    with Run().context(
        RunConfig(
            nranks=1,
            experiment=experiment,
            root=str(root),
        )
    ):
        search_config = make_stanford_config(
            root
        )

        searcher = Searcher(
            index=index_name,
            config=search_config,
            collection=collection,
        )

        search_reload_seconds = (
            time.perf_counter()
            - search_load_started
        )

        query_stats = run_queries(
            searcher=searcher,
            queries=queries,
            pid_to_document_id=pid_to_document_id,
            output_path=candidate_path,
        )

    summary = {
        "artifact_format":
            "hotpotqa.resource-pilot-result.v1",
        "status": "PASS",
        "dataset": "hotpotqa",
        "evidence_role": "RESOURCE_PILOT",
        "git_commit": git_commit(),
        "retriever": "colbert",
        "corpus_size": args.corpus_size,
        "candidate_pool": CANDIDATE_POOL,
        "warmup_queries": EXPECTED_WARMUP,
        "measured_queries": EXPECTED_MEASURED,
        "corpus_manifest_id":
            corpus_manifest.corpus_manifest_id,
        "corpus_manifest_sha256":
            corpus_manifest.sha256,
        "source_load_seconds":
            source_load_seconds,
        "corpus_materialization_seconds":
            corpus_materialization_seconds,
        "corpus": corpus_stats,
        "checkpoint": {
            "path": str(
                args.checkpoint.resolve()
            ),
            "id": COLBERT_CONFIG.checkpoint_id,
            "revision":
                COLBERT_CONFIG.checkpoint_revision,
            **checkpoint_info,
        },
        "colbert_config": {
            "dim": COLBERT_CONFIG.dim,
            "query_maxlen":
                COLBERT_CONFIG.query_maxlen,
            "doc_maxlen":
                COLBERT_CONFIG.doc_maxlen,
            "nbits": COLBERT_CONFIG.nbits,
            "kmeans_niters":
                COLBERT_CONFIG.kmeans_niters,
            "nranks": COLBERT_CONFIG.nranks,
            "index_bsize":
                COLBERT_CONFIG.index_bsize,
            "amp": COLBERT_CONFIG.amp,
            "search_ncells":
                COLBERT_CONFIG.search_ncells,
            "search_centroid_score_threshold":
                COLBERT_CONFIG.search_centroid_score_threshold,
            "search_ndocs":
                COLBERT_CONFIG.search_ndocs,
        },
        "index": {
            "index_path": str(index_path),
            "index_build_seconds":
                index_seconds,
            "index_size_bytes":
                index_size_bytes,
            "search_reload_seconds":
                search_reload_seconds,
            "cache_disk_bytes_before":
                disk_before,
            "cache_disk_bytes_after":
                disk_after_index,
            "cache_disk_bytes_created":
                disk_after_index
                - disk_before,
        },
        "queries": query_stats,
        "runtime": {
            "peak_rss_kib":
                peak_rss_kib(),
            "gpu": gpu_stats(),
            "total_seconds":
                time.perf_counter()
                - total_started,
        },
        "candidate_artifact": {
            "path": str(candidate_path),
            "sha256":
                sha256_file(candidate_path),
        },
    }

    summary_path = (
        output_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    print(
        "HOTPOTQA COLBERT RESOURCE PILOT: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
