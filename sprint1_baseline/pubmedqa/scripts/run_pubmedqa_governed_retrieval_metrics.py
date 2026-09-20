#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from retrieval_artifacts import read_candidate_artifact
from scripts.build_corpus_manifests import (
    PUBMEDQA_CONFIG,
    PUBMEDQA_REVISION,
    PUBMEDQA_SAMPLE_MANIFEST_PATH,
    PUBMEDQA_SOURCE,
    PUBMEDQA_SPLIT,
    build_pubmedqa_corpus_manifest,
    load_frozen_pubmedqa_sample_manifest,
)


ROOT = Path(__file__).resolve().parents[1]

DIVERSIFIED_ROOT = ROOT / "artifacts/diversified/pubmedqa"
CANDIDATE_ROOT = ROOT / "artifacts/candidates/pubmedqa"

OUTPUT_ROOT = ROOT / "results/sprint3/pubmedqa/retrieval"

HISTORICAL_SUMMARY = (
    ROOT
    / "results/sprint1/pubmedqa/retrieval_summary.csv"
)

RETRIEVERS = [
    "bm25",
    "dpr",
    "contriever",
    "colbertv2",
]

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

EXPECTED_QUERIES = 1000
CANDIDATE_POOL = 20
TOP_K = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def recall_at_k(
    ranked_document_ids,
    gold_document_ids,
    k,
):
    gold = set(gold_document_ids)

    if not gold:
        raise ValueError(
            "Gold evidence set must not be empty."
        )

    retrieved = set(
        ranked_document_ids[:k]
    )

    return len(gold & retrieved) / len(gold)


def reciprocal_rank_at_k(
    ranked_document_ids,
    gold_document_ids,
    k,
):
    gold = set(gold_document_ids)

    for rank, document_id in enumerate(
        ranked_document_ids[:k],
        start=1,
    ):
        if document_id in gold:
            return 1.0 / rank

    return 0.0


def load_historical_none_means():
    values = {}

    with HISTORICAL_SUMMARY.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        for row in csv.DictReader(f):
            retriever = row["retriever"]

            if retriever in RETRIEVERS:
                values[retriever] = {
                    "recall_at_5":
                        float(row["recall_at_5"]),
                    "mrr_at_5":
                        float(row["mrr_at_5"]),
                    "candidate_recall_at_20":
                        float(
                            row[
                                "candidate_recall_at_20"
                            ]
                        ),
                }

    if set(values) != set(RETRIEVERS):
        raise RuntimeError(
            "Historical PubMedQA summary is incomplete."
        )

    return values


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== PUBMEDQA GOVERNED RETRIEVAL METRICS ===",
        flush=True,
    )

    dataset_rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=str(
            Path.home()
            / ".cache/huggingface/datasets"
        ),
    )

    sample_manifest = (
        load_frozen_pubmedqa_sample_manifest(
            PUBMEDQA_SAMPLE_MANIFEST_PATH
        )
    )

    corpus_build = build_pubmedqa_corpus_manifest(
        tuple(dataset_rows),
        sample_manifest,
    )

    gold_by_sample = (
        corpus_build.gold_document_ids_by_sample
    )

    if len(gold_by_sample) != EXPECTED_QUERIES:
        raise RuntimeError(
            "Unexpected PubMedQA gold mapping size."
        )

    historical = load_historical_none_means()

    for retriever in RETRIEVERS:
        print(
            f"\n=== {retriever.upper()} ===",
            flush=True,
        )

        diversified_path = (
            DIVERSIFIED_ROOT
            / retriever
            / (
                f"{retriever}_pubmedqa_"
                "canonical_top20_to5_v1.jsonl"
            )
        )

        candidate_dir = (
            CANDIDATE_ROOT
            / f"{retriever}_top50"
        )

        if not diversified_path.exists():
            raise FileNotFoundError(
                diversified_path
            )

        # Candidate Recall@20 is fixed for all
        # diversification conditions because every
        # condition reranks the same canonical Top20 pool.
        candidate_recall20 = {}

        for position in range(EXPECTED_QUERIES):
            artifact = read_candidate_artifact(
                candidate_dir
                / f"sample_{position:04d}.json"
            )

            if (
                artifact.sample_id != position
                or artifact.requested_top_n != 50
                or len(artifact.candidates) != 50
            ):
                raise RuntimeError(
                    f"{retriever}: invalid Top50 "
                    f"artifact at position {position}"
                )

            top20_ids = [
                str(c.document_id)
                for c in artifact.candidates[
                    :CANDIDATE_POOL
                ]
            ]

            gold = {
                str(document_id)
                for document_id
                in gold_by_sample[position]
            }

            candidate_recall20[position] = (
                recall_at_k(
                    top20_ids,
                    gold,
                    CANDIDATE_POOL,
                )
            )

        rows = []
        counts = defaultdict(int)
        sums = defaultdict(
            lambda: {
                "recall_at_5": 0.0,
                "mrr_at_5": 0.0,
                "candidate_recall_at_20": 0.0,
            }
        )

        with diversified_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)

                position = int(row["position"])
                sample_id = int(row["sample_id"])
                condition = row["condition"]

                if sample_id != position:
                    raise RuntimeError(
                        f"{retriever}: sample/position "
                        f"mismatch at {position}"
                    )

                if condition not in CONDITIONS:
                    raise RuntimeError(
                        f"{retriever}: unexpected "
                        f"condition {condition}"
                    )

                if (
                    int(row["candidate_pool"])
                    != CANDIDATE_POOL
                    or int(row["top_k"]) != TOP_K
                ):
                    raise RuntimeError(
                        f"{retriever}: invalid "
                        "Top20->Top5 contract"
                    )

                selected = row["selected"]

                if len(selected) != TOP_K:
                    raise RuntimeError(
                        f"{retriever}: incomplete "
                        f"selection at {position}"
                    )

                ranked_ids = [
                    str(x["document_id"])
                    for x in selected
                ]

                if len(set(ranked_ids)) != TOP_K:
                    raise RuntimeError(
                        f"{retriever}: duplicate "
                        f"Top5 at {position}"
                    )

                gold = {
                    str(document_id)
                    for document_id
                    in gold_by_sample[position]
                }

                r5 = recall_at_k(
                    ranked_ids,
                    gold,
                    TOP_K,
                )

                mrr5 = reciprocal_rank_at_k(
                    ranked_ids,
                    gold,
                    TOP_K,
                )

                r20 = candidate_recall20[position]

                result = {
                    "dataset": "pubmedqa",
                    "sample_id": sample_id,
                    "position": position,
                    "retriever": retriever,
                    "condition": condition,
                    "gold_document_count": len(gold),
                    "recall_at_5": r5,
                    "mrr_at_5": mrr5,
                    "candidate_recall_at_20": r20,
                }

                rows.append(result)

                counts[condition] += 1
                sums[condition]["recall_at_5"] += r5
                sums[condition]["mrr_at_5"] += mrr5
                sums[condition][
                    "candidate_recall_at_20"
                ] += r20

        expected_rows = (
            EXPECTED_QUERIES
            * len(CONDITIONS)
        )

        if len(rows) != expected_rows:
            raise RuntimeError(
                f"{retriever}: expected "
                f"{expected_rows} metric rows, "
                f"found {len(rows)}"
            )

        for condition in CONDITIONS:
            if counts[condition] != EXPECTED_QUERIES:
                raise RuntimeError(
                    f"{retriever}: {condition} has "
                    f"{counts[condition]} rows"
                )

        per_query_path = (
            OUTPUT_ROOT
            / (
                f"{retriever}_pubmedqa_"
                "retrieval_metrics_v1.jsonl"
            )
        )

        with per_query_path.open(
            "w",
            encoding="utf-8",
        ) as out:
            for row in rows:
                out.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        summary = {}

        for condition in CONDITIONS:
            n = counts[condition]

            summary[condition] = {
                "questions": n,
                "mean_recall_at_5":
                    sums[condition][
                        "recall_at_5"
                    ] / n,
                "mean_mrr_at_5":
                    sums[condition][
                        "mrr_at_5"
                    ] / n,
                "mean_candidate_recall_at_20":
                    sums[condition][
                        "candidate_recall_at_20"
                    ] / n,
            }

        # Strong regression gate:
        # the relevance-only Top5 and canonical Top20
        # must reproduce the governed Sprint-1 evidence.
        baseline = summary["none"]
        old = historical[retriever]

        for new_key, old_key in [
            (
                "mean_recall_at_5",
                "recall_at_5",
            ),
            (
                "mean_mrr_at_5",
                "mrr_at_5",
            ),
            (
                "mean_candidate_recall_at_20",
                "candidate_recall_at_20",
            ),
        ]:
            if abs(
                baseline[new_key]
                - old[old_key]
            ) > 1e-12:
                raise RuntimeError(
                    f"{retriever}: Sprint-3 "
                    f"{new_key}={baseline[new_key]} "
                    f"does not reproduce historical "
                    f"{old_key}={old[old_key]}"
                )

        summary_path = (
            OUTPUT_ROOT
            / (
                f"{retriever}_pubmedqa_"
                "retrieval_metrics_summary_v1.json"
            )
        )

        summary_payload = {
            "dataset": "pubmedqa",
            "retriever": retriever,
            "metric_contract": {
                "primary":
                    "positive_gold_section_recall_at_5",
                "secondary":
                    "positive_gold_section_mrr_at_5",
                "diagnostic":
                    "positive_gold_section_candidate_recall_at_20",
                "candidate_pool": CANDIDATE_POOL,
                "top_k": TOP_K,
            },
            "conditions": summary,
        }

        summary_path.write_text(
            json.dumps(
                summary_payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "PER_QUERY:",
            per_query_path,
            flush=True,
        )
        print(
            "PER_QUERY_SHA256:",
            sha256_file(per_query_path),
            flush=True,
        )
        print(
            "SUMMARY:",
            summary_path,
            flush=True,
        )
        print(
            "SUMMARY_SHA256:",
            sha256_file(summary_path),
            flush=True,
        )

        for condition in CONDITIONS:
            s = summary[condition]

            print(
                f"{condition:12s} "
                f"Recall@5="
                f"{s['mean_recall_at_5']:.6f} "
                f"MRR@5="
                f"{s['mean_mrr_at_5']:.6f} "
                f"Recall@20="
                f"{s['mean_candidate_recall_at_20']:.6f}",
                flush=True,
            )

        print(
            f"PUBMEDQA_{retriever.upper()}_"
            "RETRIEVAL_METRICS: PASS",
            flush=True,
        )

    print(
        "\nPUBMEDQA_THREE_RETRIEVER_METRICS: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
