#!/usr/bin/env python3

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


ROOT = Path("/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3")
S2 = ROOT / "data/hotpotqa/sprint2"
RET = S2 / "retrieval"

EMB_NPY = (
    ROOT
    / "data/hotpotqa/full_retrieval_cache/contriever/full/"
    "4fe73181ae0d934a35c039919476ea85e95bf24101a1818b26fa412925c943bd/"
    "embeddings.npy"
)

QUERY_SOURCE = (
    ROOT
    / "data/hotpotqa/generation_package_2026-09-06/"
    "bm25_with_context.jsonl"
)

CANDIDATE_POOL = 20
TOP_K = 5
EXPECTED_ROWS = 7405

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
QUERY_MAP = None
RETRIEVER = None


class EmbeddingLookup:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __getitem__(self, corpus_position):
        return self.embeddings[int(corpus_position)]


def unused_embed_fn(_):
    raise RuntimeError("precomputed embeddings are required")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_worker(retriever):
    global EMBEDDINGS, QUERY_MAP, RETRIEVER

    RETRIEVER = retriever

    EMBEDDINGS = np.load(
        EMB_NPY,
        mmap_mode="r",
    )

    QUERY_MAP = {}

    with QUERY_SOURCE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            QUERY_MAP[row["query_id"]] = row["query_text"]


def process_row(item):
    position, line = item

    row = json.loads(line)

    assert row["position"] == position
    assert row["candidate_pool"] == 50

    qid = row["query_id"]
    query = QUERY_MAP[qid]

    query_sha = hashlib.sha256(
        query.encode("utf-8")
    ).hexdigest()

    assert query_sha == row["query_text_sha256"]

    raw = row["candidates"][:CANDIDATE_POOL]
    assert len(raw) == CANDIDATE_POOL

    corpus_positions = [
        int(c["corpus_position"])
        for c in raw
    ]

    assert len(set(corpus_positions)) == CANDIDATE_POOL

    candidates = []
    by_position = {}

    for c in raw:
        cp = int(c["corpus_position"])
        score = float.fromhex(c["native_score_hex"])

        doc = {
            "doc_id": cp,
            "document_id": str(c["document_id"]),
        }

        candidates.append((doc, score))
        by_position[cp] = c

    expected_top5_ids = [
        str(c["document_id"])
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

        assert len(selected) == TOP_K

        selected_positions = [
            int(doc["doc_id"])
            for doc, _ in selected
        ]

        assert len(set(selected_positions)) == TOP_K
        assert set(selected_positions).issubset(
            set(corpus_positions)
        )

        selected_rows = []

        for selected_rank, (doc, score) in enumerate(
            selected, start=1
        ):
            cp = int(doc["doc_id"])
            original = by_position[cp]

            selected_rows.append(
                {
                    "selected_rank": selected_rank,
                    "document_id": str(
                        original["document_id"]
                    ),
                    "corpus_position": cp,
                    "original_rank": int(
                        original["rank"]
                    ),
                    "retrieval_score": float(score),
                }
            )

        if condition == "none":
            assert [
                x["document_id"]
                for x in selected_rows
            ] == expected_top5_ids

        payload = {
            "dataset": "hotpotqa",
            "role": "OFFICIAL_TEST_FULL",
            "position": position,
            "query_id": qid,
            "query_text_sha256": query_sha,
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
            )
        )

    return position, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        required=True,
        choices=["bm25", "dpr", "contriever", "colbertv2"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=48,
    )
    args = parser.parse_args()

    retriever = args.retriever
    workers = args.workers

    retrieval_path = (
        RET
        / f"{retriever}_hotpotqa_official_test_full_top50_v1.jsonl"
    )

    output_path = (
        S2
        / retriever
        / "diversified"
        / (
            f"{retriever}_hotpotqa_official_test_full_"
            "canonical_top20_to5_v1.jsonl"
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")

    with retrieval_path.open("r", encoding="utf-8") as f:
        lines = [
            line
            for line in f
            if line.strip()
        ]

    assert len(lines) == EXPECTED_ROWS

    print("=== HOTPOTQA PARALLEL DIVERSIFICATION ===", flush=True)
    print("RETRIEVER:", retriever, flush=True)
    print("WORKERS:", workers, flush=True)
    print("CANDIDATE_POOL:", CANDIDATE_POOL, flush=True)
    print("TOP_K:", TOP_K, flush=True)

    ctx = get_context("fork")

    start = time.time()
    counts = Counter()

    with ctx.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(retriever,),
    ) as pool, tmp.open("w", encoding="utf-8") as out:

        iterator = pool.imap(
            process_row,
            enumerate(lines),
            chunksize=8,
        )

        completed = 0

        for position, outputs in iterator:
            assert position == completed

            for condition, text in zip(
                CONDITIONS,
                outputs,
            ):
                out.write(text + "\n")
                counts[condition] += 1

            completed += 1

            if completed % 250 == 0 or completed == EXPECTED_ROWS:
                elapsed = time.time() - start
                rate = completed / elapsed
                eta = (
                    EXPECTED_ROWS - completed
                ) / rate if rate else 0

                print(
                    f"{completed}/{EXPECTED_ROWS} "
                    f"({100*completed/EXPECTED_ROWS:.1f}%) "
                    f"rate={rate:.2f} q/s "
                    f"ETA={eta/60:.1f} min",
                    flush=True,
                )

        out.flush()
        os.fsync(out.fileno())

    assert completed == EXPECTED_ROWS

    for condition in CONDITIONS:
        assert counts[condition] == EXPECTED_ROWS

    os.replace(tmp, output_path)

    total_rows = sum(1 for _ in output_path.open("r", encoding="utf-8"))

    assert total_rows == EXPECTED_ROWS * len(CONDITIONS)

    print("OUTPUT:", output_path, flush=True)
    print("TOTAL_ROWS:", total_rows, flush=True)
    print("CONDITION_COUNTS:", dict(sorted(counts.items())), flush=True)
    print("SHA256:", sha256_file(output_path), flush=True)
    print("HOTPOTQA_DIVERSIFICATION: PASS", flush=True)


if __name__ == "__main__":
    main()
