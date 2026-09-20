#!/usr/bin/env python3
"""Run validated PubMedQA DPR candidate production.

Importing this module performs no dataset access, model loading, indexing,
retrieval, cache writes, or CandidateArtifact writes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrievers.dpr_config import DPR_CONFIG
from retrieval_artifacts import (
    CandidateArtifact,
    CandidateArtifactConflictError,
    RawCandidateResult,
    RetrieverProvenance,
    build_dpr_retriever_provenance,
    produce_dpr_candidate_artifact,
    read_candidate_artifact,
    validate_dpr_index_binding,
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

if TYPE_CHECKING:
    from retrievers.dpr_original_retriever import OriginalDPRRetriever


CANDIDATE_POOL_SIZE = 20
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/candidates/pubmedqa/dpr"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DPRRetrievalFailure:
    sample_id: str | int
    error_type: str
    message: str


@dataclass(frozen=True)
class DPRCandidateRunSummary:
    manifest_sample_count: int
    selected_sample_count: int
    completed_samples: int
    skipped_existing_samples: int
    failed_samples: int
    candidate_pool_size: int
    sample_manifest_id: str
    corpus_manifest_id: str
    retriever_index_fingerprint_sha256: str
    index_artifact_sha256: str
    embedding_artifact_sha256: str
    completed_sample_ids: tuple[str | int, ...]
    skipped_sample_ids: tuple[str | int, ...]
    failures: tuple[DPRRetrievalFailure, ...]


class DPRCandidatePoolSizeError(ValueError):
    """DPR returned a non-canonical PubMedQA candidate pool size."""


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def build_runtime_dpr_provenance(
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever: OriginalDPRRetriever,
    *,
    transformers_version: str,
) -> RetrieverProvenance:
    """Build provenance from a completely initialized, validated DPR index."""
    if retriever.is_indexed is not True:
        raise RuntimeError("DPR retriever must be indexed before provenance")
    if retriever.cache_identity is None:
        raise RuntimeError("indexed DPR retriever is missing cache_identity")
    index_sha = _require_sha256(
        retriever.index_artifact_sha256, "index_artifact_sha256"
    )
    _require_sha256(
        retriever.embedding_artifact_sha256, "embedding_artifact_sha256"
    )
    provenance = build_dpr_retriever_provenance(
        cache_identity=retriever.cache_identity,
        index_artifact_sha256=index_sha,
        transformers_version=transformers_version,
        dpr_config=DPR_CONFIG,
    )
    validate_dpr_index_binding(
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        cache_identity=retriever.cache_identity,
        dpr_config=DPR_CONFIG,
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
        len(artifact.candidates) == CANDIDATE_POOL_SIZE,
        artifact.producing_git_commit == producing_git_commit,
        artifact.worktree_clean is worktree_clean,
        artifact.environment_fingerprint_sha256 == environment_fingerprint_sha256,
    )
    if not all(expected):
        raise CandidateArtifactConflictError(
            f"existing candidate artifact conflicts with run: sample {query.sample_id!r}"
        )


def run_pubmedqa_dpr_candidates(
    *,
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever: OriginalDPRRetriever,
    output_dir: Path,
    producing_git_commit: str,
    worktree_clean: bool,
    environment_fingerprint_sha256: str,
    transformers_version: str,
    max_samples: int | None = None,
    dry_run: bool = False,
    producer: Callable[..., CandidateArtifact] = produce_dpr_candidate_artifact,
) -> DPRCandidateRunSummary:
    """Index the full corpus once and process frozen queries in manifest order."""
    queries = runtime.ordered_queries
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError("max_samples must be a non-boolean integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        selected_queries = queries[:max_samples]
    else:
        selected_queries = queries

    if retriever.config != DPR_CONFIG:
        raise ValueError("DPR retriever runtime config does not match frozen DPR_CONFIG")
    retriever.index_from_corpus_records(
        corpus_manifest=runtime.corpus_manifest,
        corpus_records=runtime.corpus_records,
    )
    retriever_provenance = build_runtime_dpr_provenance(
        runtime,
        retriever,
        transformers_version=transformers_version,
    )

    completed: list[str | int] = []
    skipped: list[str | int] = []
    failures: list[DPRRetrievalFailure] = []
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
                query.query_text,
                top_k=CANDIDATE_POOL_SIZE,
            )
            if len(retrieved) != CANDIDATE_POOL_SIZE:
                raise DPRCandidatePoolSizeError(
                    f"expected candidate count = {CANDIDATE_POOL_SIZE}; "
                    f"actual candidate count = {len(retrieved)}"
                )
            raw_results = tuple(
                RawCandidateResult(
                    document_id=document["doc_id"],
                    native_score=score,
                )
                for document, score in retrieved
            )
            artifact = producer(
                sample_manifest=runtime.sample_manifest,
                dataset_provenance=runtime.dataset_provenance,
                corpus_manifest=runtime.corpus_manifest,
                corpus_provenance=runtime.corpus_provenance,
                cache_identity=retriever.cache_identity,
                sample_id=query.sample_id,
                query_text=query.query_text,
                retriever_provenance=retriever_provenance,
                requested_top_n=CANDIDATE_POOL_SIZE,
                raw_results=raw_results,
                corpus_records=runtime.corpus_records,
                producing_git_commit=producing_git_commit,
                worktree_clean=worktree_clean,
                environment_fingerprint_sha256=environment_fingerprint_sha256,
                dpr_config=DPR_CONFIG,
            )
            if len(artifact.candidates) != CANDIDATE_POOL_SIZE:
                raise DPRCandidatePoolSizeError(
                    "DPR producer returned a non-canonical candidate count"
                )
            if not dry_run:
                write_candidate_artifact(artifact, path)
            completed.append(query.sample_id)
        except CandidateArtifactConflictError:
            raise
        except Exception as error:
            failures.append(
                DPRRetrievalFailure(
                    sample_id=query.sample_id,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    return DPRCandidateRunSummary(
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
        index_artifact_sha256=retriever.index_artifact_sha256,
        embedding_artifact_sha256=retriever.embedding_artifact_sha256,
        completed_sample_ids=tuple(completed),
        skipped_sample_ids=tuple(skipped),
        failures=tuple(failures),
    )


def _environment_payload(
    *,
    transformers_version: str,
    torch_version: str | None,
    numpy_version: str | None,
    faiss_version: str | None,
    device: str | None,
) -> dict[str, str | None]:
    return {
        "device": device,
        "faiss": faiss_version,
        "implementation": platform.python_implementation(),
        "numpy": numpy_version,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_version,
        "transformers": transformers_version,
    }


def _environment_fingerprint(**versions) -> str:
    serialized = json.dumps(
        _environment_payload(**versions),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


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
        help="Process only the first N queries in frozen manifest order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run dataset loading, DPR cache/index setup, retrieval, and producer "
            "validation, but do not write CandidateArtifact files. Validated DPR "
            "cache artifacts may still be created or updated."
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
    import faiss
    import numpy as np
    import torch
    from retrievers.dpr_original_retriever import OriginalDPRRetriever

    rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
    )
    runtime = load_validated_pubmedqa_runtime_corpus(tuple(rows))
    transformers_version = importlib_metadata.version("transformers")
    retriever = OriginalDPRRetriever(DPR_CONFIG)
    environment_fingerprint = _environment_fingerprint(
        transformers_version=transformers_version,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        faiss_version=getattr(faiss, "__version__", None)
        or _package_version("faiss-cpu"),
        device=retriever.device,
    )
    commit, clean = _git_provenance()
    summary = run_pubmedqa_dpr_candidates(
        runtime=runtime,
        retriever=retriever,
        output_dir=args.output_dir,
        producing_git_commit=commit,
        worktree_clean=clean,
        environment_fingerprint_sha256=environment_fingerprint,
        transformers_version=transformers_version,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, default=lambda value: value.__dict__, sort_keys=True))
    return 1 if summary.failed_samples > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
