#!/usr/bin/env python3

import hashlib
import json
import os
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

from datasets import load_dataset


ROOT = Path("/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3")
S2 = ROOT / "data/hotpotqa/sprint2"

CACHE = ROOT / "data/hotpotqa/hf_cache"
QRELS_REV = "b15429e9244c8ec966985d7778427c3b1543b314"

RETRIEVERS = [
    "bm25",
    "dpr",
    "contriever",
    "colbertv2",
]

EXPECTED_QUERIES = 7405
TOP_K = 5
WORKERS = 48

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

QRELS = None


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score_line(item):
    idx, line = item
    row = json.loads(line)

    qid = str(row["query_id"])
    gold = QRELS[qid]

    selected_ids = [
        str(x["document_id"])
        for x in row["selected"]
    ]

    assert len(selected_ids) == TOP_K
    assert len(set(selected_ids)) == TOP_K

    matched = set(selected_ids) & gold
    recall = len(matched) / len(gold)

    result = dict(row)
    result["positive_qrel_count"] = len(gold)
    result["relevant_retrieved_count"] = len(matched)
    result["recall_at_5"] = recall

    return idx, result


def main():
    global QRELS

    print("=== HOTPOTQA GOVERNED RETRIEVAL METRICS ===", flush=True)

    ds = load_dataset(
        "BeIR/hotpotqa-qrels",
        "default",
        split="test",
        revision=QRELS_REV,
        cache_dir=str(CACHE),
    )

    qrels = defaultdict(set)

    for row in ds:
        if float(row["score"]) >= 1:
            qrels[str(row["query-id"])].add(
                str(row["corpus-id"])
            )

    assert len(qrels) == EXPECTED_QUERIES
    assert sum(len(v) for v in qrels.values()) == 14810

    # BEIR HotpotQA has exactly two positive documents/query.
    assert set(len(v) for v in qrels.values()) == {2}

    QRELS = dict(qrels)

    print("QRELS_QUERIES:", len(QRELS), flush=True)
    print("POSITIVE_QRELS:", sum(len(v) for v in QRELS.values()), flush=True)

    ctx = get_context("fork")

    for retriever in RETRIEVERS:
        print(f"\n=== {retriever.upper()} ===", flush=True)

        inp = (
            S2 / retriever / "diversified"
            / f"{retriever}_hotpotqa_official_test_full_canonical_top20_to5_v1.jsonl"
        )

        out_dir = S2 / retriever / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        out = (
            out_dir
            / f"{retriever}_hotpotqa_official_test_full_canonical_top20_to5_metrics_v1.jsonl"
        )

        summary_path = (
            out_dir
            / f"{retriever}_hotpotqa_official_test_full_canonical_top20_to5_summary_v1.json"
        )

        with inp.open("r", encoding="utf-8") as f:
            lines = [
                line
                for line in f
                if line.strip()
            ]

        expected_rows = EXPECTED_QUERIES * len(CONDITIONS)
        assert len(lines) == expected_rows

        tmp = out.with_suffix(out.suffix + ".tmp")

        condition_count = Counter()
        condition_recall_sum = defaultdict(float)
        condition_full = Counter()
        condition_partial = Counter()
        condition_zero = Counter()

        with ctx.Pool(processes=WORKERS) as pool, \
             tmp.open("w", encoding="utf-8") as fout:

            iterator = pool.imap(
                score_line,
                enumerate(lines),
                chunksize=128,
            )

            completed = 0

            for idx, row in iterator:
                assert idx == completed

                condition = row["condition"]
                recall = float(row["recall_at_5"])

                assert condition in CONDITIONS
                assert row["query_id"] in QRELS
                assert row["positive_qrel_count"] == 2
                assert recall in (0.0, 0.5, 1.0)

                condition_count[condition] += 1
                condition_recall_sum[condition] += recall

                if recall == 1.0:
                    condition_full[condition] += 1
                elif recall == 0.5:
                    condition_partial[condition] += 1
                else:
                    condition_zero[condition] += 1

                fout.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                completed += 1

            fout.flush()
            os.fsync(fout.fileno())

        assert completed == expected_rows

        for condition in CONDITIONS:
            assert condition_count[condition] == EXPECTED_QUERIES

        os.replace(tmp, out)

        summary = {
            "artifact_format": "hotpotqa.sprint2-retrieval-metrics-summary.v1",
            "dataset": "hotpotqa",
            "role": "OFFICIAL_TEST_FULL",
            "retriever": retriever,
            "metric": "positive_qrel_document_recall_at_5",
            "query_count": EXPECTED_QUERIES,
            "candidate_pool": 20,
            "top_k": 5,
            "qrels_revision": QRELS_REV,
            "conditions": {},
        }

        for condition in CONDITIONS:
            summary["conditions"][condition] = {
                "mean_recall_at_5": (
                    condition_recall_sum[condition]
                    / EXPECTED_QUERIES
                ),
                "full_recall_queries": condition_full[condition],
                "partial_recall_queries": condition_partial[condition],
                "zero_recall_queries": condition_zero[condition],
            }

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            f.write("\n")

        print("METRICS_OUTPUT:", out, flush=True)
        print("METRICS_ROWS:", completed, flush=True)
        print("METRICS_SHA256:", sha256_file(out), flush=True)

        print("SUMMARY_OUTPUT:", summary_path, flush=True)

        for condition in CONDITIONS:
            s = summary["conditions"][condition]
            print(
                condition,
                "Recall@5=",
                f'{s["mean_recall_at_5"]:.6f}',
                "full=",
                s["full_recall_queries"],
                "partial=",
                s["partial_recall_queries"],
                "zero=",
                s["zero_recall_queries"],
                flush=True,
            )

        print(f"{retriever.upper()}_METRICS: PASS", flush=True)

    print("\nHOTPOTQA_ALL_RETRIEVAL_METRICS: PASS", flush=True)


if __name__ == "__main__":
    main()
