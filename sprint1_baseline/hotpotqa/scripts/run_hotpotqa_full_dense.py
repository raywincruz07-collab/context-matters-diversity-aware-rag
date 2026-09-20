#!/usr/bin/env python3
"""Run canonical full-corpus HotpotQA DPR or Contriever retrieval.

Canonical experiment:
- BEIR HotpotQA full corpus: 5,233,329 documents
- OFFICIAL_TEST_FULL: all 7,405 official BEIR test queries
- candidate pool: top-20
- frozen DPR / Contriever semantics from the successful resource pilots
- verified streaming corpus manifest; no 5.23M-entry legacy CorpusManifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


from retrieval_artifacts.hotpotqa_streaming_corpus_manifest import (
    HOTPOTQA_CONFIG,
    HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
    HOTPOTQA_REVISION,
    HOTPOTQA_SOURCE,
    HOTPOTQA_SPLIT,
    canonical_retrieval_content,
    verify_canonical_hotpotqa_streaming_corpus_manifest,
)


EXPECTED_CORPUS_SHA256 = (
    "ac98c20e24668bb886df75382baff8edf3944f294d1fd0dc8609e58f53f739c9"
)

EXPECTED_QUERY_MANIFEST_SHA256 = (
    "efedb99b0bf844896611f37d19bebcf710b171a375c8c9fb932c055b7aa16e9c"
)

EXPECTED_QUERY_COUNT = 7_405
CANDIDATE_POOL = 20

QUERY_SOURCE = "BeIR/hotpotqa"
QUERY_CONFIG = "queries"
QUERY_REVISION = HOTPOTQA_REVISION
QUERY_SPLIT = "queries"

CACHE_SCHEMA = "hotpotqa.full-dense-cache.v1"
RESULT_SCHEMA = "hotpotqa.full-retrieval-result.v1"


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--retriever",
        choices=("dpr", "contriever"),
        required=True,
    )
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
        "--candidate-pool",
        type=int,
        default=CANDIDATE_POOL,
    )

    return parser.parse_args()


def load_query_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("sha256") != EXPECTED_QUERY_MANIFEST_SHA256:
        raise ValueError("official test manifest SHA-256 mismatch")

    if payload.get("manifest_id") != (
        "sample-manifest:sha256:" + EXPECTED_QUERY_MANIFEST_SHA256
    ):
        raise ValueError("official test manifest ID mismatch")

    scientific = payload.get("scientific_payload")
    if not isinstance(scientific, dict):
        raise ValueError("query scientific_payload must be an object")

    computed = sha256_text(canonical_json(scientific))
    if computed != EXPECTED_QUERY_MANIFEST_SHA256:
        raise ValueError("query scientific payload hash mismatch")

    entries = scientific.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_QUERY_COUNT:
        raise ValueError("expected exactly 7,405 official test queries")

    for position, entry in enumerate(entries):
        if entry.get("position") != position:
            raise ValueError("query manifest positions are not contiguous")
        if not isinstance(entry.get("source_sample_id"), str):
            raise TypeError("query source_sample_id must be a string")
        query_sha = entry.get("query_text_sha256")
        if (
            not isinstance(query_sha, str)
            or len(query_sha) != 64
        ):
            raise ValueError("invalid query_text_sha256")

    return payload


def load_official_queries(
    *,
    query_manifest: dict[str, Any],
    dataset_cache_dir: Path,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    source_rows = load_dataset(
        QUERY_SOURCE,
        QUERY_CONFIG,
        revision=QUERY_REVISION,
        split=QUERY_SPLIT,
        cache_dir=str(dataset_cache_dir),
    )

    source_by_id: dict[str, str] = {}

    for row in source_rows:
        query_id = row.get("_id")
        question = row.get("text")

        if not isinstance(query_id, str):
            raise TypeError("BEIR query _id must be a string")
        if not isinstance(question, str):
            raise TypeError("BEIR query text must be a string")

        source_by_id[query_id] = question

    entries = query_manifest["scientific_payload"]["entries"]
    queries: list[dict[str, Any]] = []

    for position, entry in enumerate(entries):
        query_id = entry["source_sample_id"]

        try:
            question = source_by_id[query_id]
        except KeyError as exc:
            raise ValueError(
                f"official query ID missing from pinned source: {query_id}"
            ) from exc

        if sha256_text(question) != entry["query_text_sha256"]:
            raise ValueError(
                f"official query text hash mismatch: {query_id}"
            )

        queries.append(
            {
                "position": position,
                "query_id": query_id,
                "question": question,
                "query_text_sha256": entry["query_text_sha256"],
            }
        )

    if len(queries) != EXPECTED_QUERY_COUNT:
        raise ValueError("official query count mismatch")

    return queries


def build_verified_runtime_corpus(
    *,
    rows,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(rows) != HOTPOTQA_EXPECTED_DOCUMENT_COUNT:
        raise ValueError("canonical HotpotQA corpus count mismatch")

    runtime_documents: list[dict[str, Any]] = []
    global_position = 0

    for shard in manifest["storage"]["manifest_shards"]:
        shard_path = manifest_path.parent / shard["path"]

        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                expected = json.loads(line)
                position = expected["position"]

                if position != global_position:
                    raise ValueError("streaming manifest position mismatch")

                row = rows[position]

                for field in ("_id", "title", "text"):
                    if field not in row:
                        raise ValueError(
                            f"corpus row {position} missing {field}"
                        )
                    if not isinstance(row[field], str):
                        raise TypeError(
                            f"corpus row {position} {field} must be a string"
                        )

                source_id = row["_id"]
                title = row["title"]
                text = row["text"]
                retrieval_content = canonical_retrieval_content(
                    title,
                    text,
                )

                if source_id != expected["source_document_id"]:
                    raise ValueError(
                        f"source document ID mismatch at {position}"
                    )

                checks = {
                    "title_sha256": sha256_text(title),
                    "text_sha256": sha256_text(text),
                    "retrieval_content_sha256": sha256_text(
                        retrieval_content
                    ),
                }

                for name, actual in checks.items():
                    if actual != expected[name]:
                        raise ValueError(
                            f"{name} mismatch at corpus position {position}"
                        )

                runtime_documents.append(
                    {
                        "doc_id": source_id,
                        "retrieval_content": retrieval_content,
                        "corpus_position": position,
                    }
                )

                global_position += 1

                if global_position % 250_000 == 0:
                    print(
                        f"verified/materialized corpus: "
                        f"{global_position:,}/"
                        f"{HOTPOTQA_EXPECTED_DOCUMENT_COUNT:,}",
                        flush=True,
                    )

    if global_position != HOTPOTQA_EXPECTED_DOCUMENT_COUNT:
        raise ValueError("verified corpus document count mismatch")

    return runtime_documents


def configure_retriever(retriever_name: str):
    if retriever_name == "dpr":
        from retrievers.dpr_config import DPR_CONFIG
        from retrievers.dpr_original_retriever import (
            OriginalDPRRetriever,
        )

        return OriginalDPRRetriever(DPR_CONFIG), DPR_CONFIG

    if retriever_name == "contriever":
        from retrievers.contriever_config import CONTRIEVER_CONFIG
        from retrievers.contriever_retriever import (
            ContrieverRetriever,
        )

        return ContrieverRetriever(CONTRIEVER_CONFIG), CONTRIEVER_CONFIG

    raise ValueError(retriever_name)


def dense_cache_fingerprint(
    *,
    retriever_name: str,
    config,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA,
        "dataset": "hotpotqa",
        "corpus_scientific_sha256": EXPECTED_CORPUS_SHA256,
        "document_count": HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
        "retriever": retriever_name,
        "retriever_config": config.scientific_payload(),
    }

    fingerprint = sha256_text(canonical_json(payload))
    return fingerprint, payload


def build_or_load_dense_index(
    *,
    retriever_name: str,
    retriever,
    config,
    runtime_documents: list[dict[str, Any]],
    cache_root: Path,
) -> tuple[Any, dict[str, Any]]:
    import faiss

    fingerprint, identity_payload = dense_cache_fingerprint(
        retriever_name=retriever_name,
        config=config,
    )

    root = cache_root / retriever_name / "full" / fingerprint
    root.mkdir(parents=True, exist_ok=True)

    embedding_path = root / "embeddings.npy"
    index_path = root / "index.faiss"
    metadata_path = root / "metadata.json"

    expected_dimension = config.embedding_dimension
    expected_count = HOTPOTQA_EXPECTED_DOCUMENT_COUNT

    if (
        metadata_path.is_file()
        and embedding_path.is_file()
        and index_path.is_file()
    ):
        print("validated cache files found; verifying...", flush=True)

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if metadata.get("identity") != identity_payload:
            raise ValueError("dense cache identity mismatch")

        if metadata.get("fingerprint_sha256") != fingerprint:
            raise ValueError("dense cache fingerprint mismatch")

        if metadata.get("embedding_sha256") != sha256_file(
            embedding_path
        ):
            raise ValueError("dense embedding cache SHA mismatch")

        if metadata.get("index_sha256") != sha256_file(index_path):
            raise ValueError("dense FAISS cache SHA mismatch")

        embeddings = np.load(
            embedding_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if embeddings.shape != (
            expected_count,
            expected_dimension,
        ):
            raise ValueError("cached embedding shape mismatch")

        index = faiss.read_index(str(index_path))

        if index.ntotal != expected_count:
            raise ValueError("cached FAISS ntotal mismatch")
        if index.d != expected_dimension:
            raise ValueError("cached FAISS dimension mismatch")

        print("FULL DENSE CACHE: REUSED", flush=True)

        return index, {
            "cache_reused": True,
            "fingerprint_sha256": fingerprint,
            "cache_root": str(root),
            "embedding_sha256": metadata["embedding_sha256"],
            "index_sha256": metadata["index_sha256"],
            "encoding_seconds": None,
            "index_build_seconds": None,
        }

    print(
        f"encoding {expected_count:,} documents with "
        f"{retriever_name}...",
        flush=True,
    )

    encode_started = time.perf_counter()

    if retriever_name == "dpr":
        embeddings = retriever._encode_contexts(runtime_documents)
    else:
        embeddings = retriever._encode_documents(runtime_documents)

    encoding_seconds = time.perf_counter() - encode_started

    if not isinstance(embeddings, np.ndarray):
        raise TypeError("dense embeddings must be a numpy.ndarray")

    if embeddings.shape != (
        expected_count,
        expected_dimension,
    ):
        raise ValueError(
            f"dense embedding shape mismatch: {embeddings.shape}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"dense embedding dtype mismatch: {embeddings.dtype}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("dense embeddings contain non-finite values")

    print(
        f"encoding complete in {encoding_seconds:.1f}s; "
        "building IndexFlatIP...",
        flush=True,
    )

    index_started = time.perf_counter()

    index = faiss.IndexFlatIP(expected_dimension)
    index.add(embeddings)

    index_build_seconds = time.perf_counter() - index_started

    if index.ntotal != expected_count:
        raise ValueError("FAISS ntotal mismatch after build")

    print("writing dense cache...", flush=True)

    temporary_embedding = embedding_path.with_suffix(".npy.tmp")
    with temporary_embedding.open("wb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_embedding, embedding_path)

    temporary_index = index_path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temporary_index))
    os.replace(temporary_index, index_path)

    embedding_sha256 = sha256_file(embedding_path)
    index_sha256 = sha256_file(index_path)

    metadata = {
        "artifact_format": CACHE_SCHEMA,
        "fingerprint_sha256": fingerprint,
        "identity": identity_payload,
        "embedding_sha256": embedding_sha256,
        "index_sha256": index_sha256,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "index_type": "faiss.IndexFlatIP",
        "index_ntotal": int(index.ntotal),
        "index_dimension": int(index.d),
    }

    metadata_path.write_text(
        canonical_json(metadata) + "\n",
        encoding="utf-8",
    )

    return index, {
        "cache_reused": False,
        "fingerprint_sha256": fingerprint,
        "cache_root": str(root),
        "embedding_sha256": embedding_sha256,
        "index_sha256": index_sha256,
        "encoding_seconds": encoding_seconds,
        "index_build_seconds": index_build_seconds,
    }


def completed_query_count(
    path: Path,
    candidate_pool: int = CANDIDATE_POOL,
) -> int:
    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be positive")

    if not path.exists():
        return 0

    completed = 0

    with path.open("r", encoding="utf-8") as handle:
        for expected_position, line in enumerate(handle):
            payload = json.loads(line)

            if payload.get("position") != expected_position:
                raise ValueError(
                    "existing candidate JSONL is not contiguous"
                )

            if payload.get("evidence_role") != "OFFICIAL_TEST_FULL":
                raise ValueError(
                    "existing candidate JSONL has wrong evidence role"
                )

            if len(payload.get("candidates", [])) != candidate_pool:
                raise ValueError(
                    "existing candidate row does not contain "
                    f"top-{candidate_pool}"
                )

            completed += 1

    if completed > EXPECTED_QUERY_COUNT:
        raise ValueError("candidate JSONL has too many rows")

    return completed


def run_queries(
    *,
    retriever_name: str,
    retriever,
    queries: list[dict[str, Any]],
    candidate_path: Path,
    candidate_pool: int,
) -> dict[str, Any]:
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be positive")

    start_position = completed_query_count(
        candidate_path,
        candidate_pool,
    )

    if start_position:
        print(
            f"resuming retrieval at query "
            f"{start_position:,}/{EXPECTED_QUERY_COUNT:,}",
            flush=True,
        )

    mode = "a" if start_position else "w"
    query_started = time.perf_counter()

    with candidate_path.open(mode, encoding="utf-8") as handle:
        for query in queries[start_position:]:
            retrieved = retriever.retrieve(
                query["question"],
                top_k=candidate_pool,
            )

            if len(retrieved) != candidate_pool:
                raise ValueError(
                    f"{retriever_name} returned "
                    f"{len(retrieved)} candidates instead of "
                    f"{candidate_pool}"
                )

            candidates = []

            for rank, (document, score) in enumerate(
                retrieved,
                start=1,
            ):
                candidates.append(
                    {
                        "rank": rank,
                        "document_id": document["doc_id"],
                        "corpus_position": document[
                            "corpus_position"
                        ],
                        "native_score_hex": float(score).hex(),
                        "document_content_sha256": sha256_text(
                            document["retrieval_content"]
                        ),
                    }
                )

            payload = {
                "position": query["position"],
                "query_id": query["query_id"],
                "query_text_sha256": query[
                    "query_text_sha256"
                ],
                "evidence_role": "OFFICIAL_TEST_FULL",
                "candidate_pool": candidate_pool,
                "candidates": candidates,
            }

            handle.write(canonical_json(payload) + "\n")
            handle.flush()

            completed = query["position"] + 1

            if completed % 100 == 0:
                print(
                    f"retrieved: {completed:,}/"
                    f"{EXPECTED_QUERY_COUNT:,}",
                    flush=True,
                )

    total_rows = completed_query_count(
        candidate_path,
        candidate_pool,
    )

    if total_rows != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"candidate row count mismatch: {total_rows}"
        )

    return {
        "completed_queries": total_rows,
        "query_seconds_this_invocation": (
            time.perf_counter() - query_started
        ),
        "candidate_file_bytes": candidate_path.stat().st_size,
        "candidate_sha256": sha256_file(candidate_path),
    }


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    total_started = time.perf_counter()

    print("=== HOTPOTQA FULL RETRIEVAL PREFLIGHT ===", flush=True)

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
        "runtime corpus: PASS "
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

    retriever, config = configure_retriever(args.retriever)

    index, cache_stats = build_or_load_dense_index(
        retriever_name=args.retriever,
        retriever=retriever,
        config=config,
        runtime_documents=runtime_documents,
        cache_root=args.cache_root,
    )

    retriever.corpus = runtime_documents
    retriever.faiss_index = index
    retriever.is_indexed = True

    output_dir = args.output_root / args.retriever / "full"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate_pool <= 0:
        raise ValueError("--candidate-pool must be positive")

    candidate_path = (
        output_dir
        / f"candidates_top{args.candidate_pool}.jsonl"
    )

    print(
        f"running {EXPECTED_QUERY_COUNT:,} official test queries...",
        flush=True,
    )

    query_stats = run_queries(
        retriever_name=args.retriever,
        retriever=retriever,
        queries=queries,
        candidate_path=candidate_path,
        candidate_pool=args.candidate_pool,
    )

    summary = {
        "artifact_format": RESULT_SCHEMA,
        "status": "PASS",
        "dataset": "hotpotqa",
        "evidence_role": "OFFICIAL_TEST_FULL",
        "git_commit": git_commit(),
        "retriever": args.retriever,
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
        "retriever_config": config.scientific_payload(),
        "cache": cache_stats,
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
        "HOTPOTQA FULL DENSE RETRIEVAL: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
