#!/usr/bin/env python3
"""Materialize missing-only relevance Top-5 contexts from existing candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
for value in (REPOSITORY_ROOT, SRC_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from generation.cli_support import load_pubmedqa_runtime_local_only
from generation.selected_context import materialize_relevance_selected_contexts
from retrieval_artifacts import read_candidate_artifact
from retrieval_artifacts.candidate_production import materialize_candidate_set_inventory


RETRIEVERS = ("bm25", "dpr", "contriever", "colbertv2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=(*RETRIEVERS, "all"), default="all")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/candidates/pubmedqa",
    )
    parser.add_argument(
        "--candidate-set-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/candidates/sets/pubmedqa",
    )
    parser.add_argument(
        "--context-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/selected_contexts/pubmedqa",
    )
    parser.add_argument(
        "--context-set-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/selected_contexts/sets/pubmedqa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_pubmedqa_runtime_local_only(cache_dir=args.cache_dir)
    retrievers = RETRIEVERS if args.retriever == "all" else (args.retriever,)
    for retriever in retrievers:
        candidate_dir = args.candidate_root / retriever
        first = read_candidate_artifact(candidate_dir / "sample_0000.json")
        candidate_set_path = args.candidate_set_root / f"{retriever}_candidate_set_v1.json"
        materialize_candidate_set_inventory(
            sample_manifest=runtime.sample_manifest,
            ordered_queries=runtime.ordered_queries,
            candidate_directory=candidate_dir,
            dataset_provenance=runtime.dataset_provenance,
            corpus_provenance=runtime.corpus_provenance,
            retriever_provenance=first.retriever,
            corpus_records=runtime.corpus_records,
            candidate_pool=20,
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            retriever=retriever,
            output_path=candidate_set_path,
            provenance={"materializer": "missing-only selected-context prerequisite"},
        )
        wrapper = materialize_relevance_selected_contexts(
            runtime=runtime,
            candidate_directory=candidate_dir,
            candidate_set_path=candidate_set_path,
            output_directory=args.context_root / retriever,
            output_inventory_path=(
                args.context_set_root / f"{retriever}_selected_context_set_v1.json"
            ),
            repository_root=REPOSITORY_ROOT,
        )
        print(f"{retriever}: {wrapper['selected_context_set_id']}")


if __name__ == "__main__":
    main()
