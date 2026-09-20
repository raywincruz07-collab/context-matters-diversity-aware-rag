#!/usr/bin/env python3
"""Run canonical direct-Stanford PubMedQA ColBERTv2 candidate production."""

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
for path in (REPOSITORY_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from retrievers.colbert_checkpoint import resolve_colbert_checkpoint
from retrievers.colbert_checkpoint_provenance import (
    build_colbert_checkpoint_physical_provenance,
)
from retrievers.colbert_config import COLBERT_CONFIG
from retrieval_artifacts import (
    CandidateArtifact,
    CandidateArtifactConflictError,
    build_colbert_retriever_provenance,
    produce_colbert_candidate_artifact,
    read_candidate_artifact,
    validate_colbert_index_binding,
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


CANDIDATE_POOL_SIZE = COLBERT_CONFIG.candidate_pool_size
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/candidates/pubmedqa/colbertv2"
DEFAULT_INDEX_ROOT = REPOSITORY_ROOT / "artifacts/indexes/colbertv2"


@dataclass(frozen=True)
class ColBERTRetrievalFailure:
    sample_id: str | int
    error_type: str
    message: str


@dataclass(frozen=True)
class ColBERTCandidateRunSummary:
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
    completed_sample_ids: tuple[str | int, ...]
    skipped_sample_ids: tuple[str | int, ...]
    failures: tuple[ColBERTRetrievalFailure, ...]


def _candidate_path(output_dir: Path, query) -> Path:
    return output_dir / f"sample_{query.position:04d}.json"


def _require_existing_matches_run(
    artifact: CandidateArtifact,
    *, query, runtime, retriever_provenance, producing_git_commit,
    worktree_clean, environment_fingerprint_sha256,
) -> None:
    if not all((
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
    )):
        raise CandidateArtifactConflictError(
            f"existing candidate artifact conflicts with run: sample {query.sample_id!r}"
        )


def run_pubmedqa_colbert_candidates(
    *,
    runtime: ValidatedPubMedQARuntimeCorpus,
    retriever,
    output_dir: Path,
    producing_git_commit: str,
    worktree_clean: bool,
    environment_fingerprint_sha256: str,
    max_samples: int | None = None,
    producer: Callable[..., CandidateArtifact] = produce_colbert_candidate_artifact,
) -> ColBERTCandidateRunSummary:
    queries = runtime.ordered_queries
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError("max_samples must be a non-boolean integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        selected = queries[:max_samples]
    else:
        selected = queries
    if retriever.config != COLBERT_CONFIG:
        raise ValueError("ColBERT runtime config does not match frozen COLBERT_CONFIG")
    retriever.index_from_corpus_records(
        corpus_manifest=runtime.corpus_manifest,
        corpus_records=runtime.corpus_records,
    )
    if retriever.cache_identity is None or retriever.index_artifact_sha256 is None:
        raise RuntimeError("initialized ColBERT retriever lacks physical provenance")
    provenance = build_colbert_retriever_provenance(
        cache_identity=retriever.cache_identity,
        index_artifact_sha256=retriever.index_artifact_sha256,
    )
    validate_colbert_index_binding(
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        cache_identity=retriever.cache_identity,
    )
    completed, skipped, failures = [], [], []
    for query in selected:
        path = _candidate_path(output_dir, query)
        if path.exists():
            _require_existing_matches_run(
                read_candidate_artifact(path), query=query, runtime=runtime,
                retriever_provenance=provenance,
                producing_git_commit=producing_git_commit,
                worktree_clean=worktree_clean,
                environment_fingerprint_sha256=environment_fingerprint_sha256,
            )
            skipped.append(query.sample_id)
            continue
        try:
            raw_results = retriever.retrieve(
                query.query_text, top_k=CANDIDATE_POOL_SIZE
            )
            artifact = producer(
                sample_manifest=runtime.sample_manifest,
                dataset_provenance=runtime.dataset_provenance,
                corpus_manifest=runtime.corpus_manifest,
                corpus_provenance=runtime.corpus_provenance,
                cache_identity=retriever.cache_identity,
                sample_id=query.sample_id,
                query_text=query.query_text,
                retriever_provenance=provenance,
                requested_top_n=CANDIDATE_POOL_SIZE,
                raw_results=raw_results,
                corpus_records=runtime.corpus_records,
                producing_git_commit=producing_git_commit,
                worktree_clean=worktree_clean,
                environment_fingerprint_sha256=environment_fingerprint_sha256,
            )
            write_candidate_artifact(artifact, path)
            completed.append(query.sample_id)
        except CandidateArtifactConflictError:
            raise
        except Exception as error:
            failures.append(ColBERTRetrievalFailure(
                sample_id=query.sample_id,
                error_type=type(error).__name__, message=str(error),
            ))
    return ColBERTCandidateRunSummary(
        manifest_sample_count=len(queries), selected_sample_count=len(selected),
        completed_samples=len(completed), skipped_existing_samples=len(skipped),
        failed_samples=len(failures), candidate_pool_size=CANDIDATE_POOL_SIZE,
        sample_manifest_id=runtime.sample_manifest.manifest_id,
        corpus_manifest_id=runtime.corpus_manifest.corpus_manifest_id,
        retriever_index_fingerprint_sha256=provenance.index_fingerprint_sha256,
        index_artifact_sha256=retriever.index_artifact_sha256,
        completed_sample_ids=tuple(completed), skipped_sample_ids=tuple(skipped),
        failures=tuple(failures),
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=_positive_integer)
    parser.add_argument("--checkpoint-cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _git_provenance() -> tuple[str, bool]:
    commit = subprocess.run(("git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(("git", "status", "--porcelain"), cwd=REPOSITORY_ROOT,
                            check=True, capture_output=True, text=True).stdout
    return commit, not bool(status)


def main() -> int:
    args = parse_args()
    from datasets import load_dataset
    import torch
    from retrievers.colbert_runtime import CanonicalColBERTRetriever

    rows = load_dataset(
        PUBMEDQA_SOURCE, PUBMEDQA_CONFIG, split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if args.dataset_cache_dir is None else str(args.dataset_cache_dir),
    )
    runtime = load_validated_pubmedqa_runtime_corpus(tuple(rows))
    resolved = resolve_colbert_checkpoint(
        COLBERT_CONFIG, cache_dir=args.checkpoint_cache_dir, local_files_only=True
    )
    checkpoint_provenance = build_colbert_checkpoint_physical_provenance(resolved)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=resolved,
        checkpoint_provenance=checkpoint_provenance,
        index_root=args.index_root,
    )
    environment = {
        "colbert_ai": importlib_metadata.version("colbert-ai"),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "implementation": platform.python_implementation(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib_metadata.version("transformers"),
    }
    environment_sha = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    commit, clean = _git_provenance()
    summary = run_pubmedqa_colbert_candidates(
        runtime=runtime, retriever=retriever, output_dir=args.output_dir,
        producing_git_commit=commit, worktree_clean=clean,
        environment_fingerprint_sha256=environment_sha,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, default=lambda value: value.__dict__, sort_keys=True))
    return 1 if summary.failed_samples else 0


if __name__ == "__main__":
    raise SystemExit(main())
