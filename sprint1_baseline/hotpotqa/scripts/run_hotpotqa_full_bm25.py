#!/usr/bin/env python3
"""Run canonical full-corpus HotpotQA BM25 retrieval in parallel.

Canonical experiment:
- BEIR HotpotQA full corpus: 5,233,329 documents
- OFFICIAL_TEST_FULL: all 7,405 official BEIR test queries
- candidate pool: top-20
- frozen rank_bm25.BM25Okapi configuration
- parallelism only across independent queries
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

for path in (REPO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from scripts.run_hotpotqa_full_dense import (
    CANDIDATE_POOL,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_QUERY_COUNT,
    EXPECTED_QUERY_MANIFEST_SHA256,
    canonical_json,
    completed_query_count,
    git_commit,
    load_official_queries,
    load_query_manifest,
    peak_rss_kib,
    sha256_file,
    sha256_text,
    build_verified_runtime_corpus,
)

from retrieval_artifacts.hotpotqa_streaming_corpus_manifest import (
    HOTPOTQA_CONFIG,
    HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
    HOTPOTQA_REVISION,
    HOTPOTQA_SOURCE,
    HOTPOTQA_SPLIT,
    verify_canonical_hotpotqa_streaming_corpus_manifest,
)

from retrievers.bm25_config import BM25_CONFIG


RESULT_SCHEMA = "hotpotqa.full-bm25-retrieval-result.v1"

_WORKER_RETRIEVER = None
_WORKER_CANDIDATE_POOL = CANDIDATE_POOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--query-manifest",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=CANDIDATE_POOL,
    )

    return parser.parse_args()


def configure_bm25(cache_root: Path):
    import retrievers.bm25_retriever as module

    index_dir = cache_root / "bm25" / "full" / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)

    module.INDEX_DIR = str(index_dir)

    return module.BM25Retriever(BM25_CONFIG), index_dir


def convert_dense_runtime_to_bm25(
    runtime_documents: list[dict[str, Any]],
) -> None:
    """Convert in place without duplicating the 5.23M-document list."""
    for document in runtime_documents:
        document["text"] = document.pop("retrieval_content")


def _retrieve_one(query: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_RETRIEVER is None:
        raise RuntimeError("BM25 worker retriever is not initialized")

    retrieved = _WORKER_RETRIEVER.retrieve(
        query["question"],
        top_k=_WORKER_CANDIDATE_POOL,
    )

    if len(retrieved) != _WORKER_CANDIDATE_POOL:
        raise ValueError(
            f"BM25 returned {len(retrieved)} candidates instead of "
            f"{_WORKER_CANDIDATE_POOL}"
        )

    candidates: list[dict[str, Any]] = []

    for rank, (document, score) in enumerate(retrieved, start=1):
        candidates.append(
            {
                "rank": rank,
                "document_id": document["doc_id"],
                "corpus_position": document["corpus_position"],
                "native_score_hex": float(score).hex(),
                "document_content_sha256": sha256_text(
                    document["text"]
                ),
            }
        )

    return {
        "position": query["position"],
        "query_id": query["query_id"],
        "query_text_sha256": query["query_text_sha256"],
        "evidence_role": "OFFICIAL_TEST_FULL",
        "candidate_pool": _WORKER_CANDIDATE_POOL,
        "candidates": candidates,
    }


def run_parallel_queries(
    *,
    retriever,
    queries: list[dict[str, Any]],
    candidate_path: Path,
    workers: int,
    candidate_pool: int,
) -> dict[str, Any]:
    global _WORKER_RETRIEVER
    global _WORKER_CANDIDATE_POOL

    if workers < 1:
        raise ValueError("workers must be >= 1")

    available_cpus = os.cpu_count() or 1

    if workers > available_cpus:
        raise ValueError(
            f"workers={workers} exceeds available CPUs={available_cpus}"
        )

    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    start_position = completed_query_count(candidate_path, candidate_pool)

    if start_position:
        print(
            f"resuming BM25 retrieval at "
            f"{start_position:,}/{EXPECTED_QUERY_COUNT:,}",
            flush=True,
        )

    remaining = queries[start_position:]

    if not remaining:
        return {
            "completed_queries": EXPECTED_QUERY_COUNT,
            "query_seconds_this_invocation": 0.0,
            "candidate_file_bytes": candidate_path.stat().st_size,
            "candidate_sha256": sha256_file(candidate_path),
            "workers": workers,
        }

    if sys.platform != "linux":
        raise RuntimeError(
            "parallel full BM25 requires Linux fork semantics"
        )

    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be positive")

    _WORKER_RETRIEVER = retriever
    _WORKER_CANDIDATE_POOL = candidate_pool
    context = mp.get_context("fork")

    mode = "a" if start_position else "w"
    started = time.perf_counter()

    print(
        f"running {len(remaining):,} BM25 queries "
        f"with {workers} fork workers...",
        flush=True,
    )

    completed = start_position

    with candidate_path.open(mode, encoding="utf-8") as handle:
        with context.Pool(processes=workers) as pool:
            for payload in pool.imap(
                _retrieve_one,
                remaining,
                chunksize=1,
            ):
                handle.write(canonical_json(payload) + "\n")
                handle.flush()

                completed += 1

                if completed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    finished_this_run = completed - start_position
                    rate = (
                        finished_this_run / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    remaining_count = EXPECTED_QUERY_COUNT - completed
                    eta_seconds = (
                        remaining_count / rate
                        if rate > 0
                        else None
                    )

                    print(
                        f"retrieved: {completed:,}/"
                        f"{EXPECTED_QUERY_COUNT:,}; "
                        f"rate={rate:.3f} q/s; "
                        f"eta_hours="
                        f"{None if eta_seconds is None else round(eta_seconds / 3600, 2)}",
                        flush=True,
                    )

    total_rows = completed_query_count(candidate_path, candidate_pool)

    if total_rows != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"candidate row count mismatch: {total_rows}"
        )

    return {
        "completed_queries": total_rows,
        "query_seconds_this_invocation": (
            time.perf_counter() - started
        ),
        "candidate_file_bytes": candidate_path.stat().st_size,
        "candidate_sha256": sha256_file(candidate_path),
        "workers": workers,
    }


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    total_started = time.perf_counter()

    print("=== HOTPOTQA FULL BM25 PREFLIGHT ===", flush=True)

    corpus_manifest = (
        verify_canonical_hotpotqa_streaming_corpus_manifest(
            args.corpus_manifest
        )
    )

    if (
        corpus_manifest["scientific_sha256"]
        != EXPECTED_CORPUS_SHA256
    ):
        raise ValueError(
            "canonical HotpotQA streaming corpus SHA mismatch"
        )

    print(
        "corpus manifest: PASS "
        f"({HOTPOTQA_EXPECTED_DOCUMENT_COUNT:,} docs)",
        flush=True,
    )

    query_manifest = load_query_manifest(args.query_manifest)

    if (
        query_manifest["sha256"]
        != EXPECTED_QUERY_MANIFEST_SHA256
    ):
        raise ValueError("official query manifest SHA mismatch")

    print(
        f"official query manifest: PASS "
        f"({EXPECTED_QUERY_COUNT:,} queries)",
        flush=True,
    )

    print("loading pinned HotpotQA corpus...", flush=True)

    source_started = time.perf_counter()

    rows = load_dataset(
        HOTPOTQA_SOURCE,
        HOTPOTQA_CONFIG,
        revision=HOTPOTQA_REVISION,
        split=HOTPOTQA_SPLIT,
        cache_dir=str(args.dataset_cache_dir),
    )

    source_load_seconds = time.perf_counter() - source_started

    print(
        "verifying corpus and materializing BM25 runtime corpus...",
        flush=True,
    )

    corpus_started = time.perf_counter()

    runtime_documents = build_verified_runtime_corpus(
        rows=rows,
        manifest_path=args.corpus_manifest,
        manifest=corpus_manifest,
    )

    convert_dense_runtime_to_bm25(runtime_documents)

    corpus_materialization_seconds = (
        time.perf_counter() - corpus_started
    )

    print(
        f"runtime corpus: PASS "
        f"({len(runtime_documents):,} docs)",
        flush=True,
    )

    print("loading official test questions...", flush=True)

    queries = load_official_queries(
        query_manifest=query_manifest,
        dataset_cache_dir=args.dataset_cache_dir,
    )

    print(
        f"official queries: PASS ({len(queries):,})",
        flush=True,
    )

    retriever, index_dir = configure_bm25(args.cache_root)

    print("building/loading frozen BM25 index...", flush=True)

    index_started = time.perf_counter()
    retriever.index(runtime_documents)
    index_seconds = time.perf_counter() - index_started

    print(
        f"BM25 index ready in {index_seconds:.1f}s",
        flush=True,
    )

    output_dir = args.output_root / "bm25" / "full"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate_pool <= 0:
        raise ValueError("--candidate-pool must be positive")

    candidate_path = (
        output_dir
        / f"candidates_top{args.candidate_pool}.jsonl"
    )

    query_stats = run_parallel_queries(
        retriever=retriever,
        queries=queries,
        candidate_path=candidate_path,
        workers=args.workers,
        candidate_pool=args.candidate_pool,
    )

    summary = {
        "artifact_format": RESULT_SCHEMA,
        "status": "PASS",
        "dataset": "hotpotqa",
        "evidence_role": "OFFICIAL_TEST_FULL",
        "git_commit": git_commit(),
        "retriever": "bm25",
        "corpus": {
            "document_count": HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
            "scientific_sha256": EXPECTED_CORPUS_SHA256,
            "document_id_map_sha256": (
                corpus_manifest["scientific_payload"][
                    "document_id_map_sha256"
                ]
            ),
            "retrieval_serialization": (
                corpus_manifest["scientific_payload"][
                    "retrieval_serialization"
                ]
            ),
        },
        "queries": {
            "query_count": EXPECTED_QUERY_COUNT,
            "sample_manifest_sha256": (
                EXPECTED_QUERY_MANIFEST_SHA256
            ),
        },
        "candidate_pool": args.candidate_pool,
        "retriever_config": BM25_CONFIG.scientific_payload(),
        "parallelism": {
            "method": (
                "multiprocessing fork; independent queries; "
                "shared read-only parent BM25 index"
            ),
            "workers": args.workers,
        },
        "index": {
            "index_dir": str(index_dir),
            "index_seconds": index_seconds,
            "runtime_index_fingerprint": (
                retriever.runtime_index_fingerprint
            ),
            "cache_path": retriever.cache_path,
        },
        "timing": {
            "source_load_seconds": source_load_seconds,
            "corpus_materialization_seconds": (
                corpus_materialization_seconds
            ),
            "total_seconds": (
                time.perf_counter() - total_started
            ),
        },
        "runtime": {
            "peak_rss_kib_parent": peak_rss_kib(),
        },
        "candidate_artifact": {
            "path": str(candidate_path),
            **query_stats,
        },
    }

    summary_path = (
        output_dir / "summary.json"
        if args.candidate_pool == CANDIDATE_POOL
        else output_dir / f"summary_top{args.candidate_pool}.json"
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
        "HOTPOTQA FULL BM25 RETRIEVAL: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
