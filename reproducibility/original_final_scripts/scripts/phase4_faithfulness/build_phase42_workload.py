#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


FINAL8 = {
    "none",
    "mmr_0",
    "mmr_0.25",
    "mmr_0.5",
    "mmr_0.75",
    "kmeans_k2",
    "agglo_k3",
    "dpp_map",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")

NON_BOUNDARY_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "fig.",
    "eq.",
    "no.",
    "nos.",
    "st.",
    "mt.",
}


def _terminal_token(text: str):
    token = text.strip().rsplit(" ", 1)[-1]

    token = token.strip(
        '*_`"“”\'‘’()[]{}'
    )

    return token.lower()


def _is_non_boundary_abbreviation(text: str):
    token = _terminal_token(text)

    if token in NON_BOUNDARY_ABBREVIATIONS:
        return True

    if re.fullmatch(r"[a-z]\.", token):
        return True

    if re.fullmatch(r"(?:[a-z]\.){2,}", token):
        return True

    return False


def normalize_and_split(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text.strip())

    if not text:
        return []

    candidates = [
        x.strip()
        for x in SENTENCE_SPLIT_RE.split(text)
        if x.strip()
    ]

    merged = []

    for segment in candidates:
        if (
            merged
            and _is_non_boundary_abbreviation(
                merged[-1]
            )
        ):
            merged[-1] = (
                merged[-1]
                + " "
                + segment
            )
        else:
            merged.append(segment)

    return merged

def split_document_block(block: str):
    pattern = re.compile(r"\[Document [1-5]\]\s*")
    parts = pattern.split(block)

    docs = [
        x.strip()
        for x in parts
        if x.strip()
    ]

    if len(docs) != 5:
        raise RuntimeError(
            f"Expected 5 PubMedQA documents, got {len(docs)}"
        )

    if any(not x for x in docs):
        raise RuntimeError(
            "PubMedQA contains an empty document"
        )

    return docs


def extract_pubmed_explanation(raw_content: str):
    marker = "Explanation:"

    if marker not in raw_content:
        raise RuntimeError(
            "PubMedQA OK artifact missing Explanation marker"
        )

    explanation = raw_content.split(marker, 1)[1].strip()

    claims = normalize_and_split(explanation)

    if not claims:
        raise RuntimeError(
            "PubMedQA explanation yielded zero claims"
        )

    return claims


def get_passage_id(passage):
    value = (
        passage.get("passage_id")
        or passage.get("document_id")
    )

    if value is None:
        raise RuntimeError(
            "Passage has no passage_id/document_id"
        )

    return str(value)


def get_passage_body(passage):
    value = (
        passage.get("passage_body")
        or passage.get("text")
        or passage.get("body")
    )

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "Passage has no non-empty body"
        )

    return value.strip()


def find_source_row(jsonl_path, position, sample_id=None):
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("position") != position:
                continue

            if (
                sample_id is not None
                and row.get("sample_id") is not None
                and str(row.get("sample_id")) != str(sample_id)
            ):
                continue

            return line_no, row

    raise RuntimeError(
        f"No matching source row in {jsonl_path} "
        f"for position={position}"
    )


def verify_passage_parity(generation, source_row):
    generation_ids = [
        str(x)
        for x in generation.get("passage_ids", [])
    ]

    source_passages = source_row.get("passages", [])

    source_ids = [
        get_passage_id(x)
        for x in source_passages
    ]

    if len(source_ids) != 5:
        raise RuntimeError(
            f"Expected 5 source passages, got {len(source_ids)}"
        )

    if generation_ids != source_ids:
        raise RuntimeError(
            "Generation/source passage-ID mismatch:\n"
            f"generation={generation_ids}\n"
            f"source={source_ids}"
        )

    return [
        get_passage_body(x)
        for x in source_passages
    ]


def smoke_pubmed(root):
    parquet_path = (
        root
        / "results/sprint2/pubmedqa/"
          "correctness_20260925/"
          "pubmedqa_correctness_per_generation.parquet"
    )

    table = pq.read_table(
        parquet_path,
        columns=[
            "sample_id",
            "retriever",
            "condition",
            "physical_model_id",
            "status",
            "generation_relative_path",
        ],
    )

    chosen = None

    for row in table.to_pylist():
        if row["status"] == "OK":
            chosen = row
            break

    if chosen is None:
        raise RuntimeError(
            "No PubMedQA OK row found"
        )

    if chosen["condition"] not in FINAL8:
        raise RuntimeError(
            "Unexpected PubMedQA condition"
        )

    gen_path = (
        root
        / "data/pubmedqa/sprint2/"
          "generation_outputs_2026-09-25"
        / chosen["generation_relative_path"]
    )

    generation = json.loads(
        gen_path.read_text(encoding="utf-8")
    )

    raw_content = generation["observation"]["raw_content"]
    context_block = generation["request"]["prompt"]["context_block"]

    claims = extract_pubmed_explanation(raw_content)
    passages = split_document_block(context_block)

    return {
        "dataset": "pubmedqa",
        "retriever": chosen["retriever"],
        "condition": chosen["condition"],
        "physical_model_id": chosen["physical_model_id"],
        "sample_id": chosen["sample_id"],
        "claims": claims,
        "passages": passages,
        "generation_path": str(gen_path),
    }


def smoke_hotpot(root):
    parquet_path = (
        root
        / "results/sprint2/hotpotqa/"
          "correctness_20260926/"
          "per_generation.parquet"
    )

    table = pq.read_table(
        parquet_path,
        columns=[
            "sample_id",
            "retriever",
            "condition",
            "logical_model_id",
            "status",
            "measurable",
            "prediction",
            "source_root",
            "source_relative_path",
        ],
    )

    chosen = None

    for row in table.to_pylist():
        if row["measurable"]:
            chosen = row
            break

    if chosen is None:
        raise RuntimeError(
            "No measurable HotpotQA row found"
        )

    if chosen["condition"] not in FINAL8:
        raise RuntimeError(
            "Unexpected HotpotQA condition"
        )

    root_map = {
        "refresh": (
            root
            / "data/hotpotqa/sprint2/"
              "baseline_refresh_outputs_2026-09-24"
        ),
        "fresh": (
            root
            / "data/hotpotqa/sprint2/"
              "generation_outputs_2026-09-23"
        ),
    }

    if chosen["source_root"] not in root_map:
        raise RuntimeError(
            f"Unknown HotpotQA source_root={chosen['source_root']}"
        )

    gen_path = (
        root_map[chosen["source_root"]]
        / chosen["source_relative_path"]
    )

    generation = json.loads(
        gen_path.read_text(encoding="utf-8")
    )

    source_input = (
        root
        / "data/hotpotqa/sprint2/"
          "generation_package_2026-09-22"
        / generation["source_input_relative_path"]
    )

    _, source_row = find_source_row(
        source_input,
        generation["position"],
    )

    passages = verify_passage_parity(
        generation,
        source_row,
    )

    question = generation["query_text"].strip()
    answer = chosen["prediction"].strip()

    if not question or not answer:
        raise RuntimeError(
            "HotpotQA question/answer empty"
        )

    claim = (
        f"The answer to {question} is {answer}."
    )

    return {
        "dataset": "hotpotqa",
        "retriever": chosen["retriever"],
        "condition": chosen["condition"],
        "logical_model_id": chosen["logical_model_id"],
        "sample_id": chosen["sample_id"],
        "claims": [claim],
        "passages": passages,
        "passage_ids": generation["passage_ids"],
        "generation_path": str(gen_path),
        "source_input": str(source_input),
    }


def smoke_asqa(root):
    freeze = (
        root
        / "exports/"
          "asqa_sprint2_correctness_input_freeze_20260926/"
          "asqa_final_72cell_outputs.jsonl"
    )

    chosen = None

    with freeze.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("measurable"):
                chosen = row
                break

    if chosen is None:
        raise RuntimeError(
            "No measurable ASQA row found"
        )

    if chosen["condition"] not in FINAL8:
        raise RuntimeError(
            "Unexpected ASQA condition"
        )

    gen_path = (
        root
        / chosen["source_relative_path"]
    )

    generation = json.loads(
        gen_path.read_text(encoding="utf-8")
    )

    if chosen["condition"] == "none":
        package = (
            root
            / "data/asqa/"
              "generation_package_2026-09-09"
        )

        source_input = (
            package
            / f"{chosen['retriever']}_with_context.jsonl"
        )
    else:
        package = (
            root
            / "data/asqa/sprint2/"
              "generation_package_2026-09-22"
        )

        source_input = (
            package
            / chosen["retriever"]
            / f"{chosen['condition']}.jsonl"
        )

    _, source_row = find_source_row(
        source_input,
        generation["position"],
        generation.get("sample_id"),
    )

    passages = verify_passage_parity(
        generation,
        source_row,
    )

    claims = normalize_and_split(
        chosen["raw_answer"]
    )

    if not claims:
        raise RuntimeError(
            "ASQA long-form answer yielded zero sentences"
        )

    return {
        "dataset": "asqa",
        "retriever": chosen["retriever"],
        "condition": chosen["condition"],
        "logical_model_id": chosen["logical_model_id"],
        "sample_id": chosen["sample_id"],
        "claims": claims,
        "passages": passages,
        "passage_ids": generation["passage_ids"],
        "generation_path": str(gen_path),
        "source_input": str(source_input),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.smoke_test:
        raise RuntimeError(
            "Full workload materialization is intentionally "
            "not enabled in this first validation version"
        )

    results = [
        smoke_pubmed(args.root),
        smoke_hotpot(args.root),
        smoke_asqa(args.root),
    ]

    print("=== PHASE 4.2 WORKLOAD BUILDER SMOKE TEST ===")

    for result in results:
        print(f"\n[{result['dataset']}]")
        print(
            "identity =",
            {
                k: result.get(k)
                for k in [
                    "retriever",
                    "condition",
                    "logical_model_id",
                    "physical_model_id",
                    "sample_id",
                ]
                if k in result
            },
        )
        print(
            "claims =",
            len(result["claims"]),
        )
        print(
            "passages =",
            len(result["passages"]),
        )
        print(
            "claim_1 =",
            repr(result["claims"][0][:500]),
        )
        print(
            "passage_1 =",
            repr(result["passages"][0][:500]),
        )

        if len(result["passages"]) != 5:
            raise RuntimeError(
                f"{result['dataset']}: expected 5 passages"
            )

    print("\nPHASE_4_2_WORKLOAD_BUILDER_SMOKE_TEST=PASS")
    print("MODEL_INFERENCE_RUN=NO")
    print("GENERATION_CALLS=0")
    print("GPU_WORK_STARTED=NO")


if __name__ == "__main__":
    main()
