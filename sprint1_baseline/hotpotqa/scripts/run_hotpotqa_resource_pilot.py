#!/usr/bin/env python3
"""Run one governed HotpotQA resource-feasibility pilot.

RESOURCE FEASIBILITY ONLY:
- deterministic 100k / 500k / 1M corpus
- frozen 220 DEVELOPMENT queries
- first 20 warm-up, next 200 measured
- candidate pool = 20
- no qrels, answers, correctness, or protected-final evaluation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

for path in (REPO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from evaluation.metric_registry import DatasetId
from retrieval_artifacts.corpus_manifest import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
)
from retrieval_artifacts.hotpotqa_streaming_corpus_manifest import (
    HOTPOTQA_CONFIG,
    HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
    HOTPOTQA_REVISION,
    HOTPOTQA_SOURCE,
    HOTPOTQA_SPLIT,
    canonical_retrieval_content,
)
from retrieval_artifacts.producer import (
    CorpusRecord,
    document_content_sha256,
    validate_corpus_records_against_manifest,
)
from retrieval_artifacts.runtime_corpus import (
    bm25_runtime_documents_from_corpus_records,
)


ALLOWED_SIZES = (100_000, 500_000, 1_000_000)
CANDIDATE_POOL = 20
EXPECTED_QUERY_COUNT = 220
EXPECTED_WARMUP = 20
EXPECTED_MEASURED = 200

DOC_NAMESPACE = "hotpotqa-resource-pilot-doc-v1"
QUERY_NAMESPACE = "hotpotqa-resource-pilot-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_positions(
    positions_dir: Path,
    corpus_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = (
        positions_dir
        / f"hotpotqa_resource_corpus_{corpus_size}_manifest.json"
    )

    positions_path = (
        positions_dir
        / f"hotpotqa_resource_corpus_{corpus_size}_positions.npy"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["selected_document_count"] != corpus_size:
        raise ValueError("pilot corpus size mismatch")

    if manifest["selection_namespace"] != DOC_NAMESPACE:
        raise ValueError("pilot document namespace mismatch")

    if manifest["full_document_count"] != HOTPOTQA_EXPECTED_DOCUMENT_COUNT:
        raise ValueError("full HotpotQA corpus count mismatch")

    expected_sha = manifest["positions_artifact"]["physical_sha256"]

    if sha256_file(positions_path) != expected_sha:
        raise ValueError("pilot position artifact SHA-256 mismatch")

    positions = np.load(
        positions_path,
        allow_pickle=False,
    )

    if positions.dtype != np.int64:
        raise ValueError("pilot positions must be int64")

    if positions.shape != (corpus_size,):
        raise ValueError("pilot positions shape mismatch")

    if np.any(positions[1:] <= positions[:-1]):
        raise ValueError(
            "pilot positions must be strictly increasing "
            "in canonical source order"
        )

    if int(positions[0]) < 0:
        raise ValueError("negative corpus position")

    if int(positions[-1]) >= HOTPOTQA_EXPECTED_DOCUMENT_COUNT:
        raise ValueError("pilot position outside full corpus")

    return positions, manifest


def build_runtime_corpus(
    *,
    rows,
    positions: np.ndarray,
    corpus_size: int,
) -> tuple[
    CorpusManifest,
    tuple[CorpusRecord, ...],
    dict[str, int],
]:
    records: list[CorpusRecord] = []
    entries: list[CorpusManifestEntry] = []

    retrieval_utf8_bytes = 0
    retrieval_whitespace_tokens = 0

    for local_position, source_position_raw in enumerate(positions):
        source_position = int(source_position_raw)
        row = rows[source_position]

        for field in ("_id", "title", "text"):
            if field not in row:
                raise ValueError(
                    f"source row {source_position} missing {field}"
                )
            if not isinstance(row[field], str):
                raise TypeError(
                    f"{field} at source position "
                    f"{source_position} must be a string"
                )

        source_id = row["_id"]
        title = row["title"]
        text = row["text"]

        if not source_id:
            raise ValueError("HotpotQA source document ID is empty")

        retrieval_content = canonical_retrieval_content(
            title,
            text,
        )

        record = CorpusRecord(
            document_id=source_id,
            source_document_id=source_id,
            title=title,
            text=text,
            retrieval_content=retrieval_content,
            corpus_position=local_position,
        )

        records.append(record)

        entries.append(
            CorpusManifestEntry(
                position=local_position,
                doc_id=source_id,
                source_document_id=source_id,
                title_sha256=document_content_sha256(title),
                text_sha256=document_content_sha256(text),
                retrieval_content_sha256=(
                    document_content_sha256(retrieval_content)
                ),
            )
        )

        retrieval_utf8_bytes += len(
            retrieval_content.encode("utf-8")
        )

        retrieval_whitespace_tokens += len(
            retrieval_content.split()
        )

    if len(records) != corpus_size:
        raise ValueError("runtime corpus count mismatch")

    manifest = CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.HOTPOTQA,
        source=HOTPOTQA_SOURCE,
        config=HOTPOTQA_CONFIG,
        revision=HOTPOTQA_REVISION,
        split=HOTPOTQA_SPLIT,
        construction_algorithm=(
            "RESOURCE_PILOT_ONLY: ascending "
            f'SHA256("{DOC_NAMESPACE}|" + source_document_id) '
            f"global prefix size={corpus_size}; selected documents "
            "restored to canonical BEIR source order"
        ),
        input_sample_manifest_id=None,
        input_sample_manifest_sha256=None,
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

    validate_corpus_records_against_manifest(
        manifest,
        corpus_records,
    )

    stats = {
        "document_count": len(corpus_records),
        "retrieval_utf8_bytes": retrieval_utf8_bytes,
        "retrieval_whitespace_tokens": (
            retrieval_whitespace_tokens
        ),
    }

    return manifest, corpus_records, stats


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload["selection_namespace"] != QUERY_NAMESPACE:
        raise ValueError("resource query namespace mismatch")

    entries = payload["entries"]

    if len(entries) != EXPECTED_QUERY_COUNT:
        raise ValueError("expected exactly 220 resource queries")

    if sum(
        item["role"] == "warmup"
        for item in entries
    ) != EXPECTED_WARMUP:
        raise ValueError("warm-up query count mismatch")

    if sum(
        item["role"] == "measured"
        for item in entries
    ) != EXPECTED_MEASURED:
        raise ValueError("measured query count mismatch")

    return entries


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty list")

    return float(np.percentile(np.asarray(values), q))


def peak_rss_kib() -> int:
    return int(
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
    )


def configure_retriever(
    *,
    retriever_name: str,
    cache_root: Path,
    corpus_size: int,
):
    root = (
        cache_root
        / retriever_name
        / str(corpus_size)
    )

    embeddings = root / "embeddings"
    indexes = root / "indices"

    embeddings.mkdir(parents=True, exist_ok=True)
    indexes.mkdir(parents=True, exist_ok=True)

    if retriever_name == "bm25":
        import retrievers.bm25_retriever as module

        module.INDEX_DIR = str(indexes)

        return module.BM25Retriever(), root

    if retriever_name == "dpr":
        import retrievers.dpr_original_retriever as module
        from retrievers.dpr_config import DPR_CONFIG

        module.EMBEDDINGS_DIR = str(embeddings)
        module.INDEX_DIR = str(indexes)

        return module.OriginalDPRRetriever(DPR_CONFIG), root

    if retriever_name == "contriever":
        import retrievers.contriever_retriever as module
        from retrievers.contriever_config import (
            CONTRIEVER_CONFIG,
        )

        module.EMBEDDINGS_DIR = str(embeddings)
        module.INDEX_DIR = str(indexes)

        return module.ContrieverRetriever(
            CONTRIEVER_CONFIG
        ), root

    raise ValueError(
        f"unsupported retriever: {retriever_name}"
    )


def build_index(
    *,
    retriever_name: str,
    retriever,
    corpus_manifest: CorpusManifest,
    corpus_records: tuple[CorpusRecord, ...],
) -> dict[str, Any]:
    encoding_seconds = None

    if retriever_name == "bm25":
        runtime_docs = (
            bm25_runtime_documents_from_corpus_records(
                corpus_manifest=corpus_manifest,
                corpus_records=corpus_records,
            )
        )

        started = time.perf_counter()
        retriever.index(runtime_docs)
        total = time.perf_counter() - started

        return {
            "index_total_seconds": total,
            "encoding_seconds": None,
            "post_encoding_index_cache_seconds": None,
        }

    if retriever_name == "dpr":
        original = retriever._encode_contexts

        def timed_encode(*args, **kwargs):
            nonlocal encoding_seconds
            started = time.perf_counter()
            result = original(*args, **kwargs)
            encoding_seconds = (
                time.perf_counter() - started
            )
            return result

        retriever._encode_contexts = timed_encode

    elif retriever_name == "contriever":
        original = retriever._encode_documents

        def timed_encode(*args, **kwargs):
            nonlocal encoding_seconds
            started = time.perf_counter()
            result = original(*args, **kwargs)
            encoding_seconds = (
                time.perf_counter() - started
            )
            return result

        retriever._encode_documents = timed_encode

    started = time.perf_counter()

    retriever.index_from_corpus_records(
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
    )

    total = time.perf_counter() - started

    return {
        "index_total_seconds": total,
        "encoding_seconds": encoding_seconds,
        "post_encoding_index_cache_seconds": (
            None
            if encoding_seconds is None
            else total - encoding_seconds
        ),
    }


def retrieve_top20(
    retriever_name: str,
    retriever,
    question: str,
) -> list[dict[str, Any]]:
    retrieved = retriever.retrieve(
        question,
        top_k=CANDIDATE_POOL,
    )

    if len(retrieved) != CANDIDATE_POOL:
        raise ValueError(
            f"{retriever_name} returned "
            f"{len(retrieved)} candidates, expected 20"
        )

    result: list[dict[str, Any]] = []

    for rank, (document, score) in enumerate(
        retrieved,
        start=1,
    ):
        result.append(
            {
                "rank": rank,
                "document_id": document["doc_id"],
                "native_score": float(score),
            }
        )

    return result


def run_queries(
    *,
    retriever_name: str,
    retriever,
    queries: list[dict[str, Any]],
    candidate_path: Path,
) -> dict[str, Any]:
    candidate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    measured_latencies: list[float] = []
    write_seconds = 0.0

    with candidate_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for query in queries:
            started = time.perf_counter()

            candidates = retrieve_top20(
                retriever_name,
                retriever,
                query["question"],
            )

            latency = time.perf_counter() - started

            if query["role"] == "measured":
                measured_latencies.append(latency)

            payload = {
                "position": query["position"],
                "query_id": query["query_id"],
                "role": query["role"],
                "candidates": candidates,
            }

            write_started = time.perf_counter()

            handle.write(
                json.dumps(
                    payload,
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
            "expected exactly 200 measured latencies"
        )

    measured_total = sum(measured_latencies)

    candidate_bytes = candidate_path.stat().st_size

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
        "measured_query_seconds": measured_total,
        "measured_qps": (
            len(measured_latencies)
            / measured_total
        ),
        "candidate_file_bytes": candidate_bytes,
        "candidate_bytes_per_query": (
            candidate_bytes
            / len(queries)
        ),
        "candidate_write_seconds": write_seconds,
        "candidate_write_bytes_per_second": (
            candidate_bytes / write_seconds
            if write_seconds > 0
            else None
        ),
    }


def gpu_stats_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_stats() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "gpu_name": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }

        return {
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        }

    except Exception as error:
        return {
            "cuda_available": None,
            "gpu_stats_error": str(error),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--retriever",
        choices=("bm25", "dpr", "contriever"),
        required=True,
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
    from datasets import load_dataset

    args = parse_args()

    started_total = time.perf_counter()

    positions, position_manifest = load_positions(
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
            "canonical full HotpotQA corpus count mismatch"
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

    retriever, retriever_cache_root = (
        configure_retriever(
            retriever_name=args.retriever,
            cache_root=args.cache_root,
            corpus_size=args.corpus_size,
        )
    )

    disk_before = directory_size(
        retriever_cache_root
    )

    gpu_stats_reset()

    index_stats = build_index(
        retriever_name=args.retriever,
        retriever=retriever,
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
    )

    disk_after_index = directory_size(
        retriever_cache_root
    )

    output_dir = (
        args.output_root
        / args.retriever
        / str(args.corpus_size)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = (
        output_dir / "candidates_top20.jsonl"
    )

    query_stats = run_queries(
        retriever_name=args.retriever,
        retriever=retriever,
        queries=queries,
        candidate_path=candidate_path,
    )

    summary = {
        "artifact_format":
            "hotpotqa.resource-pilot-result.v1",
        "status": "PASS",
        "dataset": "hotpotqa",
        "evidence_role": "RESOURCE_PILOT",
        "git_commit": git_commit(),
        "retriever": args.retriever,
        "corpus_size": args.corpus_size,
        "candidate_pool": CANDIDATE_POOL,
        "warmup_queries": EXPECTED_WARMUP,
        "measured_queries": EXPECTED_MEASURED,
        "position_manifest_sha256": (
            hashlib.sha256(
                json.dumps(
                    position_manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ),
        "corpus_manifest_id":
            corpus_manifest.corpus_manifest_id,
        "corpus_manifest_sha256":
            corpus_manifest.sha256,
        "source_load_seconds":
            source_load_seconds,
        "corpus_materialization_seconds":
            corpus_materialization_seconds,
        "corpus": corpus_stats,
        "index": {
            **index_stats,
            "cache_disk_bytes_before":
                disk_before,
            "cache_disk_bytes_after":
                disk_after_index,
            "cache_disk_bytes_created":
                disk_after_index - disk_before,
        },
        "queries": query_stats,
        "runtime": {
            "peak_rss_kib": peak_rss_kib(),
            "gpu": gpu_stats(),
            "total_seconds":
                time.perf_counter()
                - started_total,
        },
        "candidate_artifact": {
            "path": str(candidate_path),
            "sha256": sha256_file(
                candidate_path
            ),
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
        "HOTPOTQA RESOURCE PILOT: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
