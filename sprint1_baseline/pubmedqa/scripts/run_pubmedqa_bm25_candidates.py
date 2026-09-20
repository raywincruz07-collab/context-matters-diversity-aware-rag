#!/usr/bin/env python3
"""Run validated PubMedQA BM25 candidate production.

Importing this module performs no dataset access, indexing, retrieval, or writes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrievers.bm25_config import BM25_CONFIG
from retrievers.bm25_retriever import BM25Retriever
from retrieval_artifacts import (
    CandidateArtifact,
    CandidateArtifactConflictError,
    RawCandidateResult,
    RetrieverProvenance,
    bm25_runtime_documents_from_corpus_records,
    build_bm25_retriever_provenance,
    compute_index_fingerprint_sha256,
    produce_bm25_candidate_artifact,
    read_candidate_artifact,
    validate_bm25_index_binding,
    write_candidate_artifact,
)
from scripts.build_corpus_manifests import (
    PUBMEDQA_CONFIG,
    PUBMEDQA_REVISION,
    PUBMEDQA_SOURCE,
    PUBMEDQA_SPLIT,
    ValidatedPubMedQARuntimeCorpus,
    load_validated_pubmedqa_runtime_corpus,
)


CANDIDATE_POOL_SIZE = 20
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/candidates/pubmedqa/bm25"


@dataclass(frozen=True)
class RetrievalFailure:
    sample_id: str | int
    error_type: str
    message: str


@dataclass(frozen=True)
class CandidateRunSummary:
    manifest_sample_count: int
    selected_sample_count: int
    completed_samples: int
    skipped_existing_samples: int
    failed_samples: int
    candidate_pool_size: int
    sample_manifest_id: str
    corpus_manifest_id: str
    retriever_index_fingerprint_sha256: str
    completed_sample_ids: tuple[str | int, ...]
    skipped_sample_ids: tuple[str | int, ...]
    failures: tuple[RetrievalFailure, ...]


class CandidatePoolSizeError(ValueError):
    """BM25 returned an incomplete canonical candidate pool."""


def build_runtime_bm25_provenance(
    runtime: ValidatedPubMedQARuntimeCorpus,
    *,
    library_version: str,
) -> RetrieverProvenance:
    fingerprint = compute_index_fingerprint_sha256(
        corpus_manifest_sha256=runtime.corpus_provenance.manifest_sha256,
        document_id_map_sha256=runtime.corpus_provenance.document_id_map_sha256,
        library_version=library_version,
        bm25_config=BM25_CONFIG,
    )
    provenance = build_bm25_retriever_provenance(
        library_version=library_version,
        index_fingerprint_sha256=fingerprint,
        bm25_config=BM25_CONFIG,
    )
    validate_bm25_index_binding(
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        bm25_config=BM25_CONFIG,
    )
    return provenance


def _candidate_path(output_dir: Path, query) -> Path:
    return output_dir / f"sample_{query.position:04d}.json"


def _require_existing_matches_run(
    artifact: CandidateArtifact,
    *,
    query,
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever_provenance: RetrieverProvenance,
    producing_git_commit: str,
    worktree_clean: bool,
    environment_fingerprint_sha256: str,
) -> None:
    expected = (
        artifact.sample_id == query.sample_id,
        artifact.query_text == query.query_text,
        artifact.dataset == runtime.dataset_provenance,
        artifact.corpus == runtime.corpus_provenance,
        artifact.retriever == retriever_provenance,
        artifact.requested_top_n == CANDIDATE_POOL_SIZE,
        artifact.producing_git_commit == producing_git_commit,
        artifact.worktree_clean is worktree_clean,
        artifact.environment_fingerprint_sha256 == environment_fingerprint_sha256,
    )
    if not all(expected):
        raise CandidateArtifactConflictError(
            f"existing candidate artifact conflicts with run: sample {query.sample_id!r}"
        )


def run_pubmedqa_bm25_candidates(
    *,
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever: BM25Retriever,
    output_dir: Path,
    producing_git_commit: str,
    worktree_clean: bool,
    environment_fingerprint_sha256: str,
    library_version: str,
    max_samples: int | None = None,
    dry_run: bool = False,
    producer: Callable[..., CandidateArtifact] = produce_bm25_candidate_artifact,
) -> CandidateRunSummary:
    """Index once and produce or safely resume ordered per-query artifacts."""
    queries = runtime.ordered_queries
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError("max_samples must be a non-boolean integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        selected_queries = queries[:max_samples]
    else:
        selected_queries = queries
    if retriever.runtime_config != BM25_CONFIG:
        raise ValueError("BM25Retriever runtime config does not match frozen config")
    runtime_documents = bm25_runtime_documents_from_corpus_records(
        corpus_manifest=runtime.corpus_manifest,
        corpus_records=runtime.corpus_records,
    )
    retriever.index(runtime_documents)
    retriever_provenance = build_runtime_bm25_provenance(
        runtime, library_version=library_version
    )

    completed: list[str | int] = []
    skipped: list[str | int] = []
    failures: list[RetrievalFailure] = []
    for query in selected_queries:
        path = _candidate_path(output_dir, query)
        if not dry_run and path.exists():
            existing = read_candidate_artifact(path)
            _require_existing_matches_run(
                existing,
                query=query,
                runtime=runtime,
                retriever_provenance=retriever_provenance,
                producing_git_commit=producing_git_commit,
                worktree_clean=worktree_clean,
                environment_fingerprint_sha256=environment_fingerprint_sha256,
            )
            skipped.append(query.sample_id)
            continue
        try:
            retrieved = retriever.retrieve(
                query.query_text, top_k=CANDIDATE_POOL_SIZE
            )
            if len(retrieved) != CANDIDATE_POOL_SIZE:
                raise CandidatePoolSizeError(
                    f"expected candidate count = {CANDIDATE_POOL_SIZE}; "
                    f"actual candidate count = {len(retrieved)}"
                )
            raw_results = tuple(
                RawCandidateResult(document_id=document["doc_id"], native_score=score)
                for document, score in retrieved
            )
            artifact = producer(
                sample_manifest=runtime.sample_manifest,
                dataset_provenance=runtime.dataset_provenance,
                corpus_manifest=runtime.corpus_manifest,
                corpus_provenance=runtime.corpus_provenance,
                sample_id=query.sample_id,
                query_text=query.query_text,
                retriever_provenance=retriever_provenance,
                requested_top_n=CANDIDATE_POOL_SIZE,
                raw_results=raw_results,
                corpus_records=runtime.corpus_records,
                producing_git_commit=producing_git_commit,
                worktree_clean=worktree_clean,
                environment_fingerprint_sha256=environment_fingerprint_sha256,
                bm25_config=BM25_CONFIG,
            )
            if not dry_run:
                write_candidate_artifact(artifact, path)
            completed.append(query.sample_id)
        except CandidateArtifactConflictError:
            raise
        except Exception as error:
            failures.append(
                RetrievalFailure(
                    sample_id=query.sample_id,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
    return CandidateRunSummary(
        manifest_sample_count=len(queries),
        selected_sample_count=len(selected_queries),
        completed_samples=len(completed),
        skipped_existing_samples=len(skipped),
        failed_samples=len(failures),
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        sample_manifest_id=runtime.sample_manifest.manifest_id,
        corpus_manifest_id=runtime.corpus_manifest.corpus_manifest_id,
        retriever_index_fingerprint_sha256=(
            retriever_provenance.index_fingerprint_sha256
        ),
        completed_sample_ids=tuple(completed),
        skipped_sample_ids=tuple(skipped),
        failures=tuple(failures),
    )


def _environment_fingerprint(library_version: str) -> str:
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "rank_bm25": library_version,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _git_provenance() -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=_positive_integer,
        help="Process only the first N samples in frozen manifest order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the full validated pipeline, including dataset loading, BM25 "
            "indexing and retrieval, but do not write CandidateArtifact files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for deterministic per-query CandidateArtifact files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face dataset cache directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import load_dataset

    rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
    )
    ordered_rows = tuple(rows)
    runtime = load_validated_pubmedqa_runtime_corpus(ordered_rows)
    library_version = importlib_metadata.version("rank-bm25")
    commit, clean = _git_provenance()
    summary = run_pubmedqa_bm25_candidates(
        runtime=runtime,
        retriever=BM25Retriever(BM25_CONFIG),
        output_dir=args.output_dir,
        producing_git_commit=commit,
        worktree_clean=clean,
        environment_fingerprint_sha256=_environment_fingerprint(library_version),
        library_version=library_version,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, default=lambda value: value.__dict__, sort_keys=True))
    return 1 if summary.failed_samples > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
