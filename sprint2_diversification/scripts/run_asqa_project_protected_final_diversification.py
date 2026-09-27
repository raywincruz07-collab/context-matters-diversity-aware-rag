#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from diversification.dispatch import rerank


ROOT = Path(
    "/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3"
)
S2 = ROOT / "data/asqa/sprint2"

FINAL_QUERIES = (
    ROOT
    / "data/asqa/generation_package_2026-09-09/"
    / "without_context_queries.jsonl"
)

BM25_RETRIEVAL = (
    ROOT
    / "data/asqa/retrieval/"
    / "bm25_pyserini_wikipedia_dpr_top100_v1.jsonl"
)

CONTRIEVER_RETRIEVAL = (
    ROOT
    / "data/asqa/retrieval/"
    / "contriever_meta_published_top100_v1.jsonl"
)

DPR_RETRIEVAL = (
    ROOT
    / "data/asqa/alce_published/ALCE-data/"
    / "asqa_eval_dpr_top100.json"
)

EMB_IDS = (
    S2 / "project_protected_final_top20_passage_ids_v1.npy"
)

EMB_NPY = (
    S2 / "project_protected_final_top20_contriever_fp16_v1.npy"
)

ROLE = "PROJECT_PROTECTED_FINAL"
EXPECTED_ROWS = 948
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


class EmbeddingLookup:
    def __init__(self, ids, embeddings):
        self.ids = ids
        self.embeddings = embeddings
        self.index = {
            int(pid): i
            for i, pid in enumerate(ids)
        }

    def __getitem__(self, doc_id):
        return self.embeddings[self.index[int(doc_id)]]


def unused_embed_fn(_):
    raise RuntimeError(
        "embed_fn must never be called: "
        "governed precomputed embeddings are required"
    )


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def query_id(row):
    for key in (
        "query_id",
        "sample_id",
        "source_sample_id",
        "_id",
    ):
        if row.get(key) is not None:
            return str(row[key])

    raise ValueError(
        f"query identifier missing; keys={sorted(row)}"
    )


def query_text(row):
    for key in (
        "query_text",
        "question",
        "query",
    ):
        value = row.get(key)

        if isinstance(value, str) and value:
            return value

    raise ValueError(
        f"query text missing; keys={sorted(row)}"
    )


def normalize_passage_id(value):
    value = str(value)

    if value.startswith("wiki:"):
        value = value[5:]

    return int(value)


def load_retrieval(retriever):
    if retriever == "bm25":
        rows = load_jsonl(BM25_RETRIEVAL)
        source = BM25_RETRIEVAL
    elif retriever == "contriever":
        rows = load_jsonl(CONTRIEVER_RETRIEVAL)
        source = CONTRIEVER_RETRIEVAL
    elif retriever == "dpr":
        rows = json.loads(
            DPR_RETRIEVAL.read_text(
                encoding="utf-8"
            )
        )
        source = DPR_RETRIEVAL
    else:
        raise ValueError(retriever)

    assert len(rows) == EXPECTED_ROWS

    return rows, source


def raw_candidates(retriever, row):
    if retriever in ("bm25", "contriever"):
        values = row["hits"][:CANDIDATE_POOL]

        assert len(values) == CANDIDATE_POOL

        result = []

        for expected_rank, hit in enumerate(
            values,
            start=1,
        ):
            pid = normalize_passage_id(
                hit["passage_id"]
            )

            rank = int(hit["rank"])
            assert rank == expected_rank

            result.append(
                {
                    "passage_id": pid,
                    "rank": rank,
                    "score": float(hit["score"]),
                }
            )

        return result

    values = row["docs"][:CANDIDATE_POOL]

    assert len(values) == CANDIDATE_POOL

    result = []

    for rank, doc in enumerate(
        values,
        start=1,
    ):
        pid = normalize_passage_id(
            doc["id"]
        )

        result.append(
            {
                "passage_id": pid,
                "rank": rank,
                "score": float(doc["score"]),
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retriever",
        required=True,
        choices=[
            "bm25",
            "dpr",
            "contriever",
        ],
    )

    args = parser.parse_args()
    retriever = args.retriever

    print(
        "=== ASQA PROJECT_PROTECTED_FINAL "
        "CANONICAL DIVERSIFICATION ==="
    )
    print("RETRIEVER:", retriever)
    print("ROLE:", ROLE)
    print("EXPECTED_ROWS:", EXPECTED_ROWS)
    print("CANDIDATE_POOL:", CANDIDATE_POOL)
    print("TOP_K:", TOP_K)
    print("CONDITIONS:", CONDITIONS)

    final_rows = load_jsonl(FINAL_QUERIES)

    assert len(final_rows) == EXPECTED_ROWS

    final_ids = [
        query_id(row)
        for row in final_rows
    ]

    assert len(set(final_ids)) == EXPECTED_ROWS

    final_text_by_id = {
        query_id(row): query_text(row)
        for row in final_rows
    }

    ids = np.load(
        EMB_IDS,
        mmap_mode="r",
        allow_pickle=False,
    )

    embeddings = np.load(
        EMB_NPY,
        mmap_mode="r",
        allow_pickle=False,
    )

    assert ids.shape == (47651,)
    assert embeddings.shape == (47651, 768)
    assert embeddings.dtype == np.float16
    assert np.all(ids[1:] > ids[:-1])

    lookup = EmbeddingLookup(
        ids,
        embeddings,
    )

    retrieval_rows, retrieval_source = (
        load_retrieval(retriever)
    )

    retrieval_ids = [
        query_id(row)
        for row in retrieval_rows
    ]

    assert retrieval_ids == final_ids

    output_path = (
        S2
        / retriever
        / "diversified"
        / (
            f"{retriever}_asqa_"
            "project_protected_final_"
            "canonical_top20_to5_v1.jsonl"
        )
    )

    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if tmp_path.exists():
        raise FileExistsError(
            f"refusing to overwrite temp file: "
            f"{tmp_path}"
        )

    condition_counts = Counter()
    start = time.time()
    written = 0

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as out:
        for position, row in enumerate(
            retrieval_rows
        ):
            qid = query_id(row)
            qtext = query_text(row)

            assert qid == final_ids[position]
            assert qtext == final_text_by_id[qid]

            query_sha = sha256_text(qtext)

            candidate_rows = raw_candidates(
                retriever,
                row,
            )

            candidate_ids = [
                item["passage_id"]
                for item in candidate_rows
            ]

            assert (
                len(candidate_ids)
                == CANDIDATE_POOL
            )

            assert (
                len(set(candidate_ids))
                == CANDIDATE_POOL
            )

            for pid in candidate_ids:
                assert pid in lookup.index

            candidates = [
                (
                    {
                        "doc_id": item[
                            "passage_id"
                        ],
                        "passage_id": str(
                            item["passage_id"]
                        ),
                    },
                    item["score"],
                )
                for item in candidate_rows
            ]

            original_rank = {
                item["passage_id"]:
                    item["rank"]
                for item in candidate_rows
            }

            expected_top5 = [
                str(item["passage_id"])
                for item
                in candidate_rows[:TOP_K]
            ]

            for condition in CONDITIONS:
                selected = rerank(
                    condition=condition,
                    query=qtext,
                    candidates=candidates,
                    top_k=TOP_K,
                    embed_fn=unused_embed_fn,
                    precomputed_embs=lookup,
                )

                assert len(selected) == TOP_K

                selected_ids = [
                    int(doc["doc_id"])
                    for doc, _ in selected
                ]

                assert (
                    len(set(selected_ids))
                    == TOP_K
                )

                assert set(
                    selected_ids
                ).issubset(
                    set(candidate_ids)
                )

                payload = {
                    "dataset": "asqa",
                    "role": ROLE,
                    "position": position,
                    "sample_id": qid,
                    "query": qtext,
                    "query_text_sha256":
                        query_sha,
                    "retriever": retriever,
                    "candidate_pool":
                        CANDIDATE_POOL,
                    "top_k": TOP_K,
                    "condition": condition,
                    "selected": [
                        {
                            "selected_rank":
                                selected_rank,
                            "passage_id":
                                str(
                                    doc[
                                        "doc_id"
                                    ]
                                ),
                            "original_rank":
                                original_rank[
                                    int(
                                        doc[
                                            "doc_id"
                                        ]
                                    )
                                ],
                            "retrieval_score":
                                float(score),
                        }
                        for selected_rank,
                        (doc, score)
                        in enumerate(
                            selected,
                            start=1,
                        )
                    ],
                }

                if condition == "none":
                    none_ids = [
                        item["passage_id"]
                        for item
                        in payload["selected"]
                    ]

                    assert (
                        none_ids
                        == expected_top5
                    )

                out.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                condition_counts[
                    condition
                ] += 1

                written += 1

            if (
                (position + 1) % 100 == 0
                or position + 1
                == EXPECTED_ROWS
            ):
                elapsed = (
                    time.time() - start
                )

                rate = (
                    (position + 1)
                    / elapsed
                )

                eta = (
                    (
                        EXPECTED_ROWS
                        - position
                        - 1
                    )
                    / rate
                    if rate
                    else 0
                )

                print(
                    f"{position+1}/"
                    f"{EXPECTED_ROWS} "
                    f"rate="
                    f"{rate:.2f} q/s "
                    f"ETA="
                    f"{eta/60:.1f} min",
                    flush=True,
                )

        out.flush()
        os.fsync(out.fileno())

    expected_total = (
        EXPECTED_ROWS
        * len(CONDITIONS)
    )

    assert written == expected_total

    assert condition_counts == Counter(
        {
            condition: EXPECTED_ROWS
            for condition in CONDITIONS
        }
    )

    os.replace(
        tmp_path,
        output_path,
    )

    rows = load_jsonl(output_path)

    assert len(rows) == expected_total

    per_condition = Counter(
        row["condition"]
        for row in rows
    )

    assert per_condition == Counter(
        {
            condition: EXPECTED_ROWS
            for condition in CONDITIONS
        }
    )

    print()
    print("RETRIEVAL_SOURCE:", retrieval_source)
    print(
        "RETRIEVAL_SOURCE_SHA256:",
        sha256_file(retrieval_source),
    )
    print(
        "EMBEDDING_IDS_SHA256:",
        sha256_file(EMB_IDS),
    )
    print(
        "EMBEDDINGS_SHA256:",
        sha256_file(EMB_NPY),
    )
    print("OUTPUT:", output_path)
    print("TOTAL_ROWS:", len(rows))
    print(
        "CONDITION_COUNTS:",
        dict(
            sorted(
                per_condition.items()
            )
        ),
    )
    print(
        "OUTPUT_SHA256:",
        sha256_file(output_path),
    )
    print(
        "ASQA_PROJECT_PROTECTED_FINAL_"
        f"{retriever.upper()}_"
        "DIVERSIFICATION: PASS"
    )


if __name__ == "__main__":
    main()
