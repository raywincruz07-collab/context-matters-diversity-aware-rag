#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = (
    "MoritzLaurer/"
    "DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)

DEFAULT_REVISION = (
    "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
)

EXPECTED_LABELS = {
    0: "ENTAILMENT",
    1: "NEUTRAL",
    2: "CONTRADICTION",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Phase 4.2 sentence/claim × passage NLI "
            "faithfulness inference."
        )
    )

    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Frozen Phase 4.2 workload JSONL.",
    )

    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Answer-level NLI result JSONL.",
    )

    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Output provenance/summary manifest JSON.",
    )

    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional answer-record limit for smoke tests only.",
    )

    return p.parse_args()


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def normalize_labels(model):
    observed = {
        int(k): str(v).upper()
        for k, v in model.config.id2label.items()
    }

    if observed != EXPECTED_LABELS:
        raise RuntimeError(
            "Unexpected NLI label mapping. "
            f"Expected {EXPECTED_LABELS}, got {observed}"
        )


def load_model(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but CUDA is unavailable"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )

    dtype = (
        torch.float16
        if args.device == "cuda"
        else torch.float32
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
    )

    normalize_labels(model)

    model.eval()
    model.to(args.device)

    return tokenizer, model


def validate_record(obj, line_no):
    claims = obj.get("claims")
    passages = obj.get("passages")

    if not isinstance(claims, list) or not claims:
        raise RuntimeError(
            f"line {line_no}: missing/non-empty claims list required"
        )

    if not isinstance(passages, list) or len(passages) != 5:
        raise RuntimeError(
            f"line {line_no}: expected exactly 5 passages"
        )

    for i, claim in enumerate(claims):
        if not isinstance(claim, str) or not claim.strip():
            raise RuntimeError(
                f"line {line_no}: invalid claim {i}"
            )

    for i, passage in enumerate(passages):
        if isinstance(passage, str):
            body = passage
        elif isinstance(passage, dict):
            body = (
                passage.get("passage_body")
                or passage.get("text")
                or passage.get("body")
            )
        else:
            body = None

        if not isinstance(body, str) or not body.strip():
            raise RuntimeError(
                f"line {line_no}: invalid passage {i}"
            )


def passage_text(passage):
    if isinstance(passage, str):
        return passage

    return (
        passage.get("passage_body")
        or passage.get("text")
        or passage.get("body")
    )


def infer_pairs(
    premises,
    hypotheses,
    tokenizer,
    model,
    args,
):
    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation="only_first",
        max_length=args.max_length,
        return_tensors="pt",
    )

    encoded = {
        k: v.to(args.device)
        for k, v in encoded.items()
    }

    with torch.inference_mode():
        logits = model(**encoded).logits

    labels = (
        torch.argmax(logits, dim=-1)
        .detach()
        .cpu()
        .tolist()
    )

    return labels


def process_group(records, tokenizer, model, args):
    premises = []
    hypotheses = []
    layout = []

    for record_idx, obj in enumerate(records):
        claims = obj["claims"]
        passages = obj["passages"]

        for claim_idx, claim in enumerate(claims):
            for passage_idx, passage in enumerate(passages):
                premises.append(
                    passage_text(passage).strip()
                )
                hypotheses.append(
                    claim.strip()
                )
                layout.append(
                    (
                        record_idx,
                        claim_idx,
                        passage_idx,
                    )
                )

    # Length-aware batching reduces padding waste without
    # changing any premise, hypothesis, tokenizer setting,
    # model, truncation rule, or output ordering.
    #
    # Sort only the execution order of NLI pairs, then scatter
    # predictions back to their original pair positions.
    pair_indices = sorted(
        range(len(premises)),
        key=lambda i: (
            len(premises[i]) + len(hypotheses[i]),
            i,
        ),
    )

    labels = [None] * len(premises)

    for start in range(
        0,
        len(pair_indices),
        args.batch_size,
    ):
        batch_indices = pair_indices[
            start:start + args.batch_size
        ]

        batch_labels = infer_pairs(
            [
                premises[i]
                for i in batch_indices
            ],
            [
                hypotheses[i]
                for i in batch_indices
            ],
            tokenizer,
            model,
            args,
        )

        for original_idx, label in zip(
            batch_indices,
            batch_labels,
        ):
            labels[original_idx] = label

    if any(
        label is None
        for label in labels
    ):
        raise RuntimeError(
            "Length-aware batching left an NLI pair without a label"
        )

    by_record = []

    for obj in records:
        claim_labels = [
            [None] * 5
            for _ in obj["claims"]
        ]
        by_record.append(claim_labels)

    for (
        record_idx,
        claim_idx,
        passage_idx,
    ), label in zip(layout, labels):
        by_record[record_idx][claim_idx][passage_idx] = label

    results = []

    for obj, claim_labels in zip(records, by_record):
        claim_results = []
        all_supporting_sources = set()

        for claim, passage_labels in zip(
            obj["claims"],
            claim_labels,
        ):
            supporting = [
                i
                for i, label in enumerate(passage_labels)
                if label == 0
            ]

            all_supporting_sources.update(supporting)

            claim_results.append(
                {
                    "claim": claim,
                    "supported": bool(supporting),
                    "supporting_passage_indices": supporting,
                    "argmax_labels": [
                        EXPECTED_LABELS[label]
                        for label in passage_labels
                    ],
                }
            )

        supported_count = sum(
            int(x["supported"])
            for x in claim_results
        )

        total_claims = len(claim_results)

        faithfulness = (
            supported_count / total_claims
        )

        hallucination = (
            supported_count < total_claims
        )

        result = {
            "dataset": obj.get("dataset"),
            "sample_id": obj.get("sample_id"),
            "retriever": obj.get("retriever"),
            "condition": obj.get("condition"),
            "logical_model_id": obj.get(
                "logical_model_id"
            ),
            "physical_model_id": obj.get(
                "physical_model_id"
            ),
            "status": obj.get("status"),
            "claim_count": total_claims,
            "supported_claim_count": supported_count,
            "faithfulness": faithfulness,
            "hallucination": hallucination,
            "supporting_source_count": len(
                all_supporting_sources
            ),
            "claims": claim_results,
        }

        if "hotpot_answer_source" in obj:
            result["hotpot_answer_source"] = obj[
                "hotpot_answer_source"
            ]

        results.append(result)

    return results


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")

    if args.max_length != 512:
        raise RuntimeError(
            "Frozen protocol requires max_length=512"
        )

    if args.output.exists():
        raise RuntimeError(
            f"Refusing existing output: {args.output}"
        )

    if args.manifest.exists():
        raise RuntimeError(
            f"Refusing existing manifest: {args.manifest}"
        )

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer, model = load_model(args)

    counters = Counter()
    status_counts = Counter()

    # Group enough answer records so each inference pass
    # contains many sentence/claim × passage pairs.
    # At the validated A100 batch size 128 this yields
    # approximately 4096 pairs per length-sorted execution group.
    target_pairs_per_group = max(
        args.batch_size * 32,
        args.batch_size,
    )

    pending = []
    pending_pairs = 0

    def flush(out_f):
        nonlocal pending, pending_pairs

        if not pending:
            return

        results = process_group(
            pending,
            tokenizer,
            model,
            args,
        )

        for result in results:
            out_f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            counters["answers"] += 1
            counters["claims"] += result["claim_count"]
            counters["supported_claims"] += result[
                "supported_claim_count"
            ]
            counters["nli_pairs"] += (
                result["claim_count"] * 5
            )
            counters["hallucinated_answers"] += int(
                result["hallucination"]
            )

            status_counts[
                str(result["status"])
            ] += 1

        pending = []
        pending_pairs = 0

    with (
        args.input.open(
            "r",
            encoding="utf-8",
        ) as in_f,
        args.output.open(
            "x",
            encoding="utf-8",
        ) as out_f,
    ):
        for line_no, line in enumerate(
            in_f,
            start=1,
        ):
            if (
                args.limit is not None
                and counters["answers"]
                + len(pending)
                >= args.limit
            ):
                break

            obj = json.loads(line)

            validate_record(
                obj,
                line_no,
            )

            pending.append(obj)

            pending_pairs += (
                len(obj["claims"]) * 5
            )

            if pending_pairs >= target_pairs_per_group:
                flush(out_f)

        flush(out_f)

    if counters["answers"] == 0:
        raise RuntimeError("No answers processed")

    output_sha = sha256_file(
        args.output
    )

    faithfulness = (
        counters["supported_claims"]
        / counters["claims"]
    )

    hallucination_rate = (
        counters["hallucinated_answers"]
        / counters["answers"]
    )

    manifest = {
        "artifact_format": (
            "context-matters."
            "phase42-faithfulness-nli.v1"
        ),
        "input": str(args.input),
        "input_sha256": sha256_file(
            args.input
        ),
        "output": str(args.output),
        "output_sha256": output_sha,
        "model": args.model,
        "revision": args.revision,
        "id2label": EXPECTED_LABELS,
        "max_length": args.max_length,
        "truncation": "only_first",
        "premise": "retrieved_passage",
        "hypothesis": "frozen_claim",
        "support_rule": (
            "claim supported iff >=1 of 5 passages "
            "has argmax ENTAILMENT"
        ),
        "device": args.device,
        "batch_size": args.batch_size,
        "pair_batch_order": (
            "ascending_combined_character_length_within_group"
        ),
        "target_pairs_per_group": target_pairs_per_group,
        "length_aware_batching": True,
        "limit": args.limit,
        "answers": counters["answers"],
        "claims": counters["claims"],
        "supported_claims": counters[
            "supported_claims"
        ],
        "nli_pairs": counters["nli_pairs"],
        "hallucinated_answers": counters[
            "hallucinated_answers"
        ],
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "status_counts": dict(
            sorted(status_counts.items())
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "model_inference_run": True,
    }

    args.manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
