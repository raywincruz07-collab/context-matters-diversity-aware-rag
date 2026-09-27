#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from diversification.dispatch import rerank
from retrieval_artifacts import read_candidate_artifact


ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_ROOT = ROOT / "artifacts/candidates/pubmedqa"
OUTPUT_ROOT = ROOT / "artifacts/diversified/pubmedqa"

EMB_NPY = (
    ROOT
    / "data/embeddings/"
    "contriever_embeddings_"
    "e83069f3c75fe3a146f1059d766e58a55b0382266e9ded7862d4ac2e98de38ea.npy"
)

EXPECTED_QUERIES = 1000
CANDIDATE_POOL = 20
TOP_K = 5

CONDITIONS = [
    "none",
    "mmr_0",
    "mmr_0.25",
    "mmr_0.5",
    "mmr_0.75",
    "kmeans_k2",
    "kmeans_k3",
    "kmeans_k5",
    "agglo_k3",
    "agglo_k5",
    "dpp_map",
]

EMBEDDINGS = None
RETRIEVER = None
INPUT_DIR = None


class EmbeddingLookup:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __getitem__(self, corpus_position):
        return self.embeddings[int(corpus_position)]


def unused_embed_fn(_):
    raise RuntimeError("precomputed embeddings are required")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_worker(retriever: str, input_dir: str):
    global EMBEDDINGS, RETRIEVER, INPUT_DIR

    RETRIEVER = retriever
    INPUT_DIR = Path(input_dir)

    EMBEDDINGS = np.load(
        EMB_NPY,
        mmap_mode="r",
    )

    if EMBEDDINGS.shape != (3358, 768):
        raise RuntimeError(
            f"unexpected PubMedQA Contriever embedding shape: "
            f"{EMBEDDINGS.shape}"
        )


def process_position(position: int):
    path = INPUT_DIR / f"sample_{position:04d}.json"

    artifact = read_candidate_artifact(path)

    if artifact.requested_top_n != 50:
        raise RuntimeError(
            f"{path}: expected Top50 artifact"
        )

    if len(artifact.candidates) != 50:
        raise RuntimeError(
            f"{path}: expected 50 candidates"
        )

    if artifact.sample_id != position:
        raise RuntimeError(
            f"{path}: sample_id {artifact.sample_id!r} "
            f"does not match manifest position {position}"
        )

    query = artifact.query_text

    raw = artifact.candidates[:CANDIDATE_POOL]

    if len(raw) != CANDIDATE_POOL:
        raise RuntimeError(
            f"{path}: incomplete Top20 candidate pool"
        )

    corpus_positions = [
        int(c.corpus_position)
        for c in raw
    ]

    if len(set(corpus_positions)) != CANDIDATE_POOL:
        raise RuntimeError(
            f"{path}: duplicate Top20 corpus positions"
        )

    if any(
        cp < 0 or cp >= EMBEDDINGS.shape[0]
        for cp in corpus_positions
    ):
        raise RuntimeError(
            f"{path}: corpus position outside embedding matrix"
        )

    candidates = []
    by_position = {}

    for c in raw:
        cp = int(c.corpus_position)

        doc = {
            "doc_id": cp,
            "document_id": c.document_id,
        }

        candidates.append(
            (doc, float(c.native_score))
        )

        by_position[cp] = c

    expected_top5_ids = [
        c.document_id
        for c in raw[:TOP_K]
    ]

    lookup = EmbeddingLookup(EMBEDDINGS)

    outputs = []

    for condition in CONDITIONS:
        selected = rerank(
            condition=condition,
            query=query,
            candidates=candidates,
            top_k=TOP_K,
            embed_fn=unused_embed_fn,
            precomputed_embs=lookup,
        )

        if len(selected) != TOP_K:
            raise RuntimeError(
                f"{path}: {condition} returned "
                f"{len(selected)} selections"
            )

        selected_positions = [
            int(doc["doc_id"])
            for doc, _ in selected
        ]

        if len(set(selected_positions)) != TOP_K:
            raise RuntimeError(
                f"{path}: {condition} returned duplicates"
            )

        if not set(selected_positions).issubset(
            set(corpus_positions)
        ):
            raise RuntimeError(
                f"{path}: {condition} selected outside Top20"
            )

        selected_rows = []

        for selected_rank, (doc, score) in enumerate(
            selected,
            start=1,
        ):
            cp = int(doc["doc_id"])
            original = by_position[cp]

            selected_rows.append(
                {
                    "selected_rank": selected_rank,
                    "document_id": original.document_id,
                    "source_document_id": original.source_document_id,
                    "corpus_position": cp,
                    "original_rank": int(original.rank),
                    "retrieval_score": float(score),
                }
            )

        if condition == "none":
            actual_ids = [
                x["document_id"]
                for x in selected_rows
            ]

            if actual_ids != expected_top5_ids:
                raise RuntimeError(
                    f"{path}: relevance baseline changed native Top5"
                )

        payload = {
            "dataset": "pubmedqa",
            "position": position,
            "sample_id": artifact.sample_id,
            "query_text_sha256": hashlib.sha256(
                query.encode("utf-8")
            ).hexdigest(),
            "retriever": RETRIEVER,
            "candidate_pool": CANDIDATE_POOL,
            "top_k": TOP_K,
            "condition": condition,
            "selected": selected_rows,
        }

        outputs.append(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    return position, outputs


def run_one(retriever: str, workers: int):
    input_dir = (
        CANDIDATE_ROOT
        / f"{retriever}_top50"
    )

    if not input_dir.is_dir():
        raise RuntimeError(
            f"missing input directory: {input_dir}"
        )

    files = sorted(
        input_dir.glob("sample_*.json")
    )

    if len(files) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"{retriever}: expected "
            f"{EXPECTED_QUERIES} Top50 artifacts, "
            f"found {len(files)}"
        )

    output_dir = (
        OUTPUT_ROOT
        / retriever
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{retriever}_pubmedqa_"
            "canonical_top20_to5_v1.jsonl"
        )
    )

    tmp = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    print(
        "\n=== PUBMEDQA PARALLEL DIVERSIFICATION ===",
        flush=True,
    )
    print("RETRIEVER:", retriever, flush=True)
    print("WORKERS:", workers, flush=True)
    print("QUERIES:", EXPECTED_QUERIES, flush=True)
    print("CANDIDATE_POOL:", CANDIDATE_POOL, flush=True)
    print("TOP_K:", TOP_K, flush=True)

    ctx = get_context("fork")

    counts = Counter()
    completed = 0
    start = time.time()

    with ctx.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(
            retriever,
            str(input_dir),
        ),
    ) as pool, tmp.open(
        "w",
        encoding="utf-8",
    ) as out:

        iterator = pool.imap(
            process_position,
            range(EXPECTED_QUERIES),
            chunksize=8,
        )

        for position, outputs in iterator:
            if position != completed:
                raise RuntimeError(
                    f"non-deterministic output order: "
                    f"{position} != {completed}"
                )

            for condition, text in zip(
                CONDITIONS,
                outputs,
            ):
                out.write(text + "\n")
                counts[condition] += 1

            completed += 1

            if (
                completed % 100 == 0
                or completed == EXPECTED_QUERIES
            ):
                elapsed = time.time() - start

                rate = (
                    completed / elapsed
                    if elapsed > 0
                    else 0
                )

                eta = (
                    (EXPECTED_QUERIES - completed)
                    / rate
                    if rate > 0
                    else 0
                )

                print(
                    f"{completed}/{EXPECTED_QUERIES} "
                    f"({100*completed/EXPECTED_QUERIES:.1f}%) "
                    f"rate={rate:.2f} q/s "
                    f"ETA={eta/60:.2f} min",
                    flush=True,
                )

        out.flush()
        os.fsync(out.fileno())

    if completed != EXPECTED_QUERIES:
        raise RuntimeError(
            f"{retriever}: incomplete query count"
        )

    for condition in CONDITIONS:
        if counts[condition] != EXPECTED_QUERIES:
            raise RuntimeError(
                f"{retriever}: bad count for {condition}: "
                f"{counts[condition]}"
            )

    os.replace(
        tmp,
        output_path,
    )

    total_rows = sum(
        1
        for _ in output_path.open(
            "r",
            encoding="utf-8",
        )
    )

    expected_rows = (
        EXPECTED_QUERIES
        * len(CONDITIONS)
    )

    if total_rows != expected_rows:
        raise RuntimeError(
            f"{retriever}: expected "
            f"{expected_rows} rows, "
            f"found {total_rows}"
        )

    print("OUTPUT:", output_path, flush=True)
    print("TOTAL_ROWS:", total_rows, flush=True)
    print(
        "CONDITION_COUNTS:",
        dict(sorted(counts.items())),
        flush=True,
    )
    print(
        "SHA256:",
        sha256_file(output_path),
        flush=True,
    )
    print(
        f"PUBMEDQA_{retriever.upper()}_DIVERSIFICATION: PASS",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retriever",
        required=True,
        choices=[
            "bm25",
            "dpr",
            "contriever",
            "colbertv2",
        ],
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=56,
    )

    args = parser.parse_args()

    run_one(
        args.retriever,
        args.workers,
    )


if __name__ == "__main__":
    main()
