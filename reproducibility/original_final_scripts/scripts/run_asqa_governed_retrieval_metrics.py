#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from evaluation.asqa_retrieval_matcher import aspect_matches


ROOT = Path("/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3")
S2 = ROOT / "data/asqa/sprint2"

BODY_FILE = S2 / "governed_dpr_contriever_selected_passage_bodies_v1.jsonl"

ALPHAS_DEV = (0.3, 0.5, 0.7)
ALPHAS_SELECTION = (0.5,)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def alpha_dcg(aspect_lists, alpha):
    counts = defaultdict(int)
    dcg = 0.0

    for rank, aspects in enumerate(aspect_lists, start=1):
        gain = 0.0
        for i in aspects:
            gain += (1.0 - alpha) ** counts[i]

        dcg += gain / math.log2(rank + 1)

        for i in aspects:
            counts[i] += 1

    return dcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True, choices=["dpr", "contriever"])
    args = parser.parse_args()

    retriever = args.retriever

    print("=== ASQA GOVERNED RETRIEVAL METRICS ===")
    print("RETRIEVER:", retriever)

    print("Loading governed passage bodies...", flush=True)
    bodies = {}
    with BODY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            bodies[int(obj["passage_id"])] = obj["body"]

    print("PASSAGE_BODIES:", len(bodies), flush=True)
    assert len(bodies) == 98359

    specs = [
        (
            "DEVELOPMENT",
            S2 / retriever / "diversified" /
            f"{retriever}_asqa_development_canonical_top20_to5_v1.jsonl",
            S2 / "matcher/asqa_development_full_corpus_matcher_v1.jsonl",
            S2 / "matcher/asqa_development_official_aspects_aliases_v1.jsonl",
            S2 / retriever / "metrics" /
            f"{retriever}_asqa_development_canonical_top20_to5_metrics_v1.jsonl",
            S2 / retriever / "metrics" /
            f"{retriever}_asqa_development_canonical_top20_to5_summary_v1.json",
            3482,
            15,
            ALPHAS_DEV,
        ),
        (
            "SELECTION",
            S2 / retriever / "diversified" /
            f"{retriever}_asqa_selection_canonical_top20_to5_v1.jsonl",
            S2 / "matcher/asqa_selection_full_corpus_matcher_v1.jsonl",
            S2 / "matcher/asqa_selection_official_aspects_aliases_v1.jsonl",
            S2 / retriever / "metrics" /
            f"{retriever}_asqa_selection_canonical_top20_to5_metrics_v1.jsonl",
            S2 / retriever / "metrics" /
            f"{retriever}_asqa_selection_canonical_top20_to5_summary_v1.json",
            871,
            11,
            ALPHAS_SELECTION,
        ),
    ]

    for (
        role,
        diversified_path,
        matcher_path,
        aliases_path,
        output_path,
        summary_path,
        expected_questions,
        expected_conditions,
        alphas,
    ) in specs:

        print(f"\n=== {retriever.upper()} {role} METRICS ===", flush=True)
        start = time.time()

        diversified = load_jsonl(diversified_path)
        matcher_rows = load_jsonl(matcher_path)
        alias_rows = load_jsonl(aliases_path)

        assert len(matcher_rows) == expected_questions
        assert len(alias_rows) == expected_questions
        assert len(diversified) == expected_questions * expected_conditions

        matcher_by_pos = {r["position"]: r for r in matcher_rows}
        alias_by_pos = {r["position"]: r for r in alias_rows}

        assert len(matcher_by_pos) == expected_questions
        assert len(alias_by_pos) == expected_questions

        # Cache J_q(d,i) work across conditions:
        # same question/passage may occur in many diversification conditions.
        match_cache = {}

        aggregates = defaultdict(
            lambda: {
                "total": 0,
                "bzero": 0,
                "srecall": [],
                "ndcg": {a: [] for a in alphas},
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")

        with tmp.open("w", encoding="utf-8") as out:
            for idx, row in enumerate(diversified):
                pos = row["position"]
                condition = row["condition"]

                m = matcher_by_pos[pos]
                arow = alias_by_pos[pos]

                assert row["sample_id"] == m["sample_id"] == arow["sample_id"]
                assert row["query_text_sha256"] == m["query_text_sha256"]
                assert row["query_text_sha256"] == arow["query_text_sha256"]

                B_q = int(m["B_q"])
                G_q = int(m["G_q"])
                coverable = set(int(x) for x in m["coverable_aspect_indices"])

                assert G_q == int(arow["aspect_count"])

                selected_passages = []
                aspect_lists = []

                for selected in row["selected"]:
                    pid = int(selected["passage_id"])
                    key = (pos, pid)

                    if key not in match_cache:
                        body = bodies[pid]
                        matched = []

                        for aspect in arow["aspects"]:
                            aspect_index = int(aspect["aspect_index"])
                            aliases = aspect["raw_official_aliases"]

                            if aspect_matches(body, aliases):
                                matched.append(aspect_index)

                        match_cache[key] = tuple(matched)

                    matched = list(match_cache[key])
                    aspect_lists.append(matched)

                    selected_passages.append(
                        {
                            "selected_rank": int(selected["selected_rank"]),
                            "passage_id": str(pid),
                            "original_rank": int(selected["original_rank"]),
                            "aspect_indices": matched,
                        }
                    )

                retrieved_coverable = sorted(
                    set().union(*(set(x) for x in aspect_lists)) & coverable
                )

                if B_q == 0:
                    srecall = None
                else:
                    srecall = len(retrieved_coverable) / B_q

                alpha_results = {}

                for alpha in alphas:
                    dcg = alpha_dcg(aspect_lists, alpha)
                    idcg = float(m["alpha_idcg_at5"][str(alpha)])

                    ndcg = None if idcg == 0.0 else dcg / idcg

                    alpha_results[str(alpha)] = {
                        "dcg": dcg,
                        "idcg": idcg,
                        "ndcg": ndcg,
                    }

                if role == "DEVELOPMENT":
                    payload = {
                        "dataset": "asqa",
                        "role": role,
                        "retriever": retriever,
                        "candidate_pool": 20,
                        "top_k": 5,
                        "position": pos,
                        "sample_id": row["sample_id"],
                        "condition": condition,
                        "G_q": G_q,
                        "B_q": B_q,
                        "retrieved_coverable_aspect_indices": retrieved_coverable,
                        "SRecall_at5": srecall,
                        "alpha_nDCG_at5": alpha_results,
                        "selected_passages": selected_passages,
                    }
                else:
                    payload = {
                        "dataset": "asqa",
                        "role": role,
                        "retriever": retriever,
                        "sample_id": row["sample_id"],
                        "position": pos,
                        "query_text_sha256": row["query_text_sha256"],
                        "candidate_pool": 20,
                        "top_k": 5,
                        "condition": condition,
                        "B_q": B_q,
                        "SRecall_at5": srecall,
                        "alpha_nDCG_at5_alpha_0.5": alpha_results["0.5"]["ndcg"],
                        "selected_passage_ids": [
                            x["passage_id"] for x in selected_passages
                        ],
                    }

                out.write(json.dumps(payload, ensure_ascii=False) + "\n")

                agg = aggregates[condition]
                agg["total"] += 1

                if B_q == 0:
                    agg["bzero"] += 1
                else:
                    agg["srecall"].append(srecall)
                    for alpha in alphas:
                        ndcg = alpha_results[str(alpha)]["ndcg"]
                        if ndcg is not None:
                            agg["ndcg"][alpha].append(ndcg)

                if (idx + 1) % 10000 == 0:
                    print(
                        f"{role}: {idx+1}/{len(diversified)} records "
                        f"cache={len(match_cache)}",
                        flush=True,
                    )

            out.flush()
            os.fsync(out.fileno())

        os.replace(tmp, output_path)

        # Validate output row count.
        output_rows = load_jsonl(output_path)
        assert len(output_rows) == len(diversified)

        summary_conditions = {}

        for condition in sorted(aggregates):
            agg = aggregates[condition]

            assert agg["total"] == expected_questions
            assert agg["bzero"] == sum(
                1 for x in matcher_rows if int(x["B_q"]) == 0
            )

            scored = len(agg["srecall"])

            result = {
                "questions_total": agg["total"],
                "questions_B_q_zero": agg["bzero"],
                "questions_scored": scored,
                "mean_SRecall_at5": (
                    sum(agg["srecall"]) / scored if scored else None
                ),
            }

            for alpha in alphas:
                vals = agg["ndcg"][alpha]
                result[f"mean_alpha_nDCG_at5_alpha_{alpha}"] = (
                    sum(vals) / len(vals) if vals else None
                )

            summary_conditions[condition] = result

        summary = {
            "dataset": "asqa",
            "role": role,
            "retriever": retriever,
            "candidate_pool": 20,
            "top_k": 5,
            "questions": expected_questions,
            "conditions": expected_conditions,
            "records": len(output_rows),
            "match_cache_pairs": len(match_cache),
            "elapsed_seconds": time.time() - start,
            "output_sha256": sha256_file(output_path),
            "condition_metrics": summary_conditions,
        }

        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

        print("OUTPUT:", output_path, flush=True)
        print("SUMMARY:", summary_path, flush=True)
        print("ROWS:", len(output_rows), flush=True)
        print("MATCH_CACHE_PAIRS:", len(match_cache), flush=True)
        print("SHA256:", summary["output_sha256"], flush=True)

        print("\nCONDITION RESULTS:")
        for condition in sorted(summary_conditions):
            r = summary_conditions[condition]
            print(
                condition,
                "SRecall@5=",
                None if r["mean_SRecall_at5"] is None
                else round(r["mean_SRecall_at5"], 6),
                "alpha-nDCG@5(0.5)=",
                None if r.get("mean_alpha_nDCG_at5_alpha_0.5") is None
                else round(r["mean_alpha_nDCG_at5_alpha_0.5"], 6),
            )

        # Frozen equivalence check on DEVELOPMENT.
        if role == "DEVELOPMENT":
            assert (
                summary_conditions["none"]["mean_SRecall_at5"]
                ==
                summary_conditions["mmr_1"]["mean_SRecall_at5"]
            )
            assert (
                summary_conditions["none"]["mean_alpha_nDCG_at5_alpha_0.5"]
                ==
                summary_conditions["mmr_1"]["mean_alpha_nDCG_at5_alpha_0.5"]
            )

        print(f"{role}_METRICS: PASS", flush=True)

    print()
    print(f"ASQA_{retriever.upper()}_RETRIEVAL_METRICS: PASS")


if __name__ == "__main__":
    main()
