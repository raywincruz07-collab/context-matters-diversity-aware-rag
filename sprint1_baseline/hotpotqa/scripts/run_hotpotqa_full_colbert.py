#!/usr/bin/env python3
"""Run canonical full-corpus HotpotQA ColBERTv2 retrieval."""

from __future__ import annotations

import argparse
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
from colbert.infra import Run, RunConfig

from retrieval_artifacts.hotpotqa_streaming_corpus_manifest import (
    HOTPOTQA_CONFIG,
    HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
    HOTPOTQA_REVISION,
    HOTPOTQA_SOURCE,
    HOTPOTQA_SPLIT,
    verify_canonical_hotpotqa_streaming_corpus_manifest,
)
from retrievers.colbert_config import COLBERT_CONFIG

from scripts.run_hotpotqa_colbert_resource_pilot import (
    make_stanford_config,
    validate_checkpoint,
    validate_frozen_config,
)
from scripts.run_hotpotqa_resource_pilot import directory_size
from scripts.run_hotpotqa_full_dense import (
    CANDIDATE_POOL,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_QUERY_COUNT,
    EXPECTED_QUERY_MANIFEST_SHA256,
    build_verified_runtime_corpus,
    canonical_json,
    completed_query_count,
    git_commit,
    load_official_queries,
    load_query_manifest,
    peak_rss_kib,
    sha256_file,
    sha256_text,
)

RESULT_SCHEMA = "hotpotqa.full-colbert-retrieval-result.v1"
EXPERIMENT = "hotpotqa_official_test_full"
INDEX_NAME = "colbertv2.nbits2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=CANDIDATE_POOL,
    )
    return parser.parse_args()


def run_queries(
    *,
    searcher: Searcher,
    queries: list[dict[str, Any]],
    runtime_documents: list[dict[str, Any]],
    output_path: Path,
    candidate_pool: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be positive")

    start_position = completed_query_count(
        output_path,
        candidate_pool,
    )
    remaining = queries[start_position:]

    if start_position:
        print(
            f"resuming ColBERT retrieval at "
            f"{start_position:,}/{EXPECTED_QUERY_COUNT:,}",
            flush=True,
        )

    if not remaining:
        return {
            "completed_queries": EXPECTED_QUERY_COUNT,
            "query_seconds_this_invocation": 0.0,
            "candidate_file_bytes": output_path.stat().st_size,
            "candidate_sha256": sha256_file(output_path),
        }

    mode = "a" if start_position else "w"
    started = time.perf_counter()
    completed = start_position

    with output_path.open(mode, encoding="utf-8") as handle:
        for query in remaining:
            pids, _ranks, scores = searcher.search(
                query["question"],
                k=candidate_pool,
            )

            if len(pids) != candidate_pool:
                raise ValueError(
                    f"ColBERT returned {len(pids)} candidates instead of "
                    f"{candidate_pool}"
                )

            if len(scores) != candidate_pool:
                raise ValueError("ColBERT score count mismatch")

            normalized_pids = tuple(int(pid) for pid in pids)

            if len(set(normalized_pids)) != candidate_pool:
                raise ValueError("ColBERT returned duplicate PIDs")

            if any(
                pid < 0 or pid >= len(runtime_documents)
                for pid in normalized_pids
            ):
                raise ValueError("ColBERT PID outside canonical corpus")

            numeric_scores = tuple(float(score) for score in scores)

            if any(not np.isfinite(score) for score in numeric_scores):
                raise ValueError("ColBERT returned non-finite score")

            candidates = []

            for rank, (pid, score) in enumerate(
                zip(normalized_pids, numeric_scores, strict=True),
                start=1,
            ):
                document = runtime_documents[pid]

                candidates.append(
                    {
                        "rank": rank,
                        "document_id": document["doc_id"],
                        "corpus_position": document["corpus_position"],
                        "native_score_hex": float(score).hex(),
                        "document_content_sha256": sha256_text(
                            document["retrieval_content"]
                        ),
                    }
                )

            payload = {
                "position": query["position"],
                "query_id": query["query_id"],
                "query_text_sha256": query["query_text_sha256"],
                "evidence_role": "OFFICIAL_TEST_FULL",
                "candidate_pool": candidate_pool,
                "candidates": candidates,
            }

            handle.write(canonical_json(payload) + "\n")
            handle.flush()

            completed += 1

            if completed % 100 == 0:
                elapsed = time.perf_counter() - started
                done_now = completed - start_position
                rate = done_now / elapsed if elapsed > 0 else 0.0
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

    total_rows = completed_query_count(
        output_path,
        candidate_pool,
    )

    if total_rows != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"candidate row count mismatch: {total_rows}"
        )

    return {
        "completed_queries": total_rows,
        "query_seconds_this_invocation": (
            time.perf_counter() - started
        ),
        "candidate_file_bytes": output_path.stat().st_size,
        "candidate_sha256": sha256_file(output_path),
    }


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    total_started = time.perf_counter()

    print("=== HOTPOTQA FULL COLBERT PREFLIGHT ===", flush=True)

    validate_frozen_config()

    if COLBERT_CONFIG.nranks != 1:
        raise ValueError(
            "canonical HotpotQA ColBERT nranks must equal 1"
        )

    checkpoint_info = validate_checkpoint(args.checkpoint)

    corpus_manifest = (
        verify_canonical_hotpotqa_streaming_corpus_manifest(
            args.corpus_manifest
        )
    )

    if corpus_manifest["scientific_sha256"] != EXPECTED_CORPUS_SHA256:
        raise ValueError("canonical HotpotQA corpus SHA mismatch")

    print(
        f"corpus manifest: PASS "
        f"({HOTPOTQA_EXPECTED_DOCUMENT_COUNT:,} docs)",
        flush=True,
    )

    query_manifest = load_query_manifest(args.query_manifest)

    if query_manifest["sha256"] != EXPECTED_QUERY_MANIFEST_SHA256:
        raise ValueError("official query manifest SHA mismatch")

    print(
        f"official query manifest: PASS "
        f"({EXPECTED_QUERY_COUNT:,} queries)",
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly 1 visible GPU; "
            f"found {torch.cuda.device_count()}"
        )

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

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
        "verifying corpus rows and materializing runtime corpus...",
        flush=True,
    )

    corpus_started = time.perf_counter()

    runtime_documents = build_verified_runtime_corpus(
        rows=rows,
        manifest_path=args.corpus_manifest,
        manifest=corpus_manifest,
    )

    corpus_materialization_seconds = (
        time.perf_counter() - corpus_started
    )

    print(
        f"runtime corpus: PASS ({len(runtime_documents):,} docs)",
        flush=True,
    )

    collection = [
        document["retrieval_content"]
        for document in runtime_documents
    ]

    print("loading official test questions...", flush=True)

    queries = load_official_queries(
        query_manifest=query_manifest,
        dataset_cache_dir=args.dataset_cache_dir,
    )

    print(
        f"official queries: PASS ({len(queries):,})",
        flush=True,
    )

    root = args.cache_root / "colbert" / "full"
    root.mkdir(parents=True, exist_ok=True)

    disk_before = directory_size(root)

    torch.cuda.reset_peak_memory_stats()

    print(
        f"indexing {len(collection):,} documents with ColBERTv2...",
        flush=True,
    )

    index_started = time.perf_counter()

    with Run().context(
        RunConfig(
            nranks=1,
            experiment=EXPERIMENT,
            root=str(root),
        )
    ):
        config = make_stanford_config(root)

        indexer = Indexer(
            checkpoint=str(args.checkpoint.resolve()),
            config=config,
        )

        index_path = Path(
            indexer.index(
                name=INDEX_NAME,
                collection=collection,
                overwrite="resume",
            )
        )

    index_seconds = time.perf_counter() - index_started

    if not index_path.is_dir():
        raise RuntimeError(
            f"ColBERT index directory missing: {index_path}"
        )

    if not (index_path / "metadata.json").is_file():
        raise RuntimeError("ColBERT metadata.json missing")

    index_size_bytes = directory_size(index_path)
    disk_after_index = directory_size(root)

    print(
        f"index complete in {index_seconds:.1f}s",
        flush=True,
    )

    output_dir = args.output_root / "colbert" / "full"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate_pool <= 0:
        raise ValueError("--candidate-pool must be positive")

    candidate_path = (
        output_dir
        / f"candidates_top{args.candidate_pool}.jsonl"
    )

    print("loading ColBERT searcher...", flush=True)

    search_load_started = time.perf_counter()

    with Run().context(
        RunConfig(
            nranks=1,
            experiment=EXPERIMENT,
            root=str(root),
        )
    ):
        search_config = make_stanford_config(root)

        searcher = Searcher(
            index=INDEX_NAME,
            config=search_config,
            collection=collection,
        )

        search_reload_seconds = (
            time.perf_counter() - search_load_started
        )

        print(
            f"running {EXPECTED_QUERY_COUNT:,} official test queries...",
            flush=True,
        )

        query_stats = run_queries(
            searcher=searcher,
            queries=queries,
            runtime_documents=runtime_documents,
            output_path=candidate_path,
            candidate_pool=args.candidate_pool,
        )

    summary = {
        "artifact_format": RESULT_SCHEMA,
        "status": "PASS",
        "dataset": "hotpotqa",
        "evidence_role": "OFFICIAL_TEST_FULL",
        "git_commit": git_commit(),
        "retriever": "colbert",
        "candidate_pool": args.candidate_pool,
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
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "id": COLBERT_CONFIG.checkpoint_id,
            "revision": COLBERT_CONFIG.checkpoint_revision,
            **checkpoint_info,
        },
        "colbert_config": {
            **COLBERT_CONFIG.scientific_payload(),
            "candidate_pool_size": args.candidate_pool,
        },
        "index": {
            "index_path": str(index_path),
            "index_build_seconds": index_seconds,
            "index_size_bytes": index_size_bytes,
            "search_reload_seconds": search_reload_seconds,
            "cache_disk_bytes_before": disk_before,
            "cache_disk_bytes_after": disk_after_index,
            "cache_disk_bytes_created": (
                disk_after_index - disk_before
            ),
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
            "peak_rss_kib": peak_rss_kib(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated()
            ),
            "gpu_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved()
            ),
        },
        "candidate_artifact": {
            "path": str(candidate_path),
            **query_stats,
        },
    }

    summary_path = (
        output_dir / "summary.json"
        if args.candidate_pool == CANDIDATE_POOL
        else output_dir
        / f"summary_top{args.candidate_pool}.json"
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
        "HOTPOTQA FULL COLBERT RETRIEVAL: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
