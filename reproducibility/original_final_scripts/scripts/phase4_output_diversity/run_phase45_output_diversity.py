#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "facebook/contriever"
DEFAULT_REVISION = "2bd46a25019aeea091fd42d1f0fd4801675cf699"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Phase 4.5 ASQA intra-answer semantic output diversity "
            "using pinned Contriever sentence embeddings."
        )
    )

    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Frozen ASQA Phase 4.2 workload JSONL.",
    )

    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Per-answer Phase 4.5 JSONL output.",
    )

    p.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Aggregate CSV by retriever × condition × logical model.",
    )

    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Provenance and run manifest JSON.",
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
        help="Optional answer limit for smoke tests only.",
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


def mean_pool(last_hidden_state, attention_mask):
    if last_hidden_state.ndim != 3:
        raise RuntimeError(
            "Contriever last_hidden_state must have rank 3"
        )

    if attention_mask.shape != last_hidden_state.shape[:2]:
        raise RuntimeError(
            "Attention-mask shape mismatch"
        )

    mask = attention_mask.unsqueeze(-1).to(
        last_hidden_state.dtype
    )

    denominator = mask.sum(dim=1)

    if torch.any(denominator == 0):
        raise RuntimeError(
            "Encountered zero-token sentence"
        )

    return (
        (last_hidden_state * mask).sum(dim=1)
        / denominator
    )


def encode_sentences(
    sentences,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
):
    embeddings = []

    for start in range(
        0,
        len(sentences),
        batch_size,
    ):
        batch = sentences[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if "attention_mask" not in inputs:
            raise RuntimeError(
                "Contriever tokenizer output "
                "requires attention_mask"
            )

        inputs = {
            name: value.to(device)
            for name, value in inputs.items()
        }

        with torch.inference_mode():
            output = model(**inputs)

            pooled = mean_pool(
                output.last_hidden_state,
                inputs["attention_mask"],
            )

        if pooled.dtype != torch.float32:
            raise RuntimeError(
                "Frozen Contriever semantics "
                "require float32 pooled embeddings"
            )

        embeddings.append(
            pooled.cpu().numpy()
        )

    return np.vstack(
        embeddings
    ).astype(
        np.float32,
        copy=False,
    )


def output_diversity(embeddings):
    if embeddings.ndim != 2:
        raise RuntimeError(
            "Embeddings must have rank 2"
        )

    n = embeddings.shape[0]

    if n < 2:
        return None

    embs = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        embs,
        axis=1,
        keepdims=True,
    )

    norms = np.where(
        norms > 0,
        norms,
        1.0,
    )

    embs = embs / norms

    sim_matrix = (
        embs @ embs.T
    )

    triu_indices = np.triu_indices(
        n,
        k=1,
    )

    pairwise_sims = (
        sim_matrix[
            triu_indices
        ]
    )

    return float(
        np.mean(
            1.0 - pairwise_sims
        )
    )


def load_records(path, limit=None):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            if (
                limit is not None
                and len(records) >= limit
            ):
                break

            obj = json.loads(line)

            if obj.get("dataset") != "asqa":
                raise RuntimeError(
                    f"line {line_no}: "
                    "Phase 4.5 primary input must be ASQA"
                )

            claims = obj.get("claims")

            if (
                not isinstance(claims, list)
                or not claims
            ):
                raise RuntimeError(
                    f"line {line_no}: "
                    "missing claims"
                )

            for claim in claims:
                if (
                    not isinstance(claim, str)
                    or not claim.strip()
                ):
                    raise RuntimeError(
                        f"line {line_no}: "
                        "invalid sentence"
                    )

            records.append(obj)

    if not records:
        raise RuntimeError(
            "No ASQA records loaded"
        )

    return records


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise RuntimeError(
            "--batch-size must be >= 1"
        )

    if args.max_length != 512:
        raise RuntimeError(
            "Frozen protocol requires max_length=512"
        )

    if not args.input.is_file():
        raise FileNotFoundError(
            args.input
        )

    for path in [
        args.output,
        args.summary,
        args.manifest,
    ]:
        if path.exists():
            raise RuntimeError(
                f"Refusing existing output: {path}"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "--device cuda requested "
            "but CUDA is unavailable"
        )

    records = load_records(
        args.input,
        args.limit,
    )

    sentences = []
    spans = []

    for record in records:
        start = len(sentences)

        row_sentences = [
            claim.strip()
            for claim in record["claims"]
        ]

        sentences.extend(
            row_sentences
        )

        spans.append(
            (
                start,
                len(sentences),
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )

    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
    )

    model = model.to(
        args.device
    )

    model = model.float()
    model.eval()

    embeddings = encode_sentences(
        sentences=sentences,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if (
        embeddings.shape[0]
        != len(sentences)
    ):
        raise RuntimeError(
            "Sentence embedding count mismatch"
        )

    if (
        embeddings.shape[1]
        != 768
    ):
        raise RuntimeError(
            "Expected Contriever embedding dimension 768"
        )

    aggregate = defaultdict(
        lambda: {
            "answers": 0,
            "eligible": 0,
            "diversity_sum": 0.0,
        }
    )

    eligible_count = 0
    ineligible_count = 0

    with args.output.open(
        "x",
        encoding="utf-8",
    ) as out_f:
        for record, (
            start,
            stop,
        ) in zip(
            records,
            spans,
        ):
            sentence_count = (
                stop - start
            )

            eligible = (
                sentence_count >= 2
            )

            if eligible:
                diversity = output_diversity(
                    embeddings[
                        start:stop
                    ]
                )

                eligible_count += 1
            else:
                diversity = None
                ineligible_count += 1

            result = {
                "dataset": "asqa",
                "sample_id": record.get(
                    "sample_id"
                ),
                "retriever": record.get(
                    "retriever"
                ),
                "condition": record.get(
                    "condition"
                ),
                "logical_model_id": record.get(
                    "logical_model_id"
                ),
                "physical_model_id": record.get(
                    "physical_model_id"
                ),
                "status": record.get(
                    "status"
                ),
                "sentence_count": (
                    sentence_count
                ),
                "eligible": eligible,
                "output_diversity": (
                    diversity
                ),
            }

            out_f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            key = (
                result["retriever"],
                result["condition"],
                result[
                    "logical_model_id"
                ],
            )

            agg = aggregate[key]

            agg["answers"] += 1

            if eligible:
                agg["eligible"] += 1
                agg[
                    "diversity_sum"
                ] += diversity

    with args.summary.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as summary_f:
        writer = csv.writer(
            summary_f
        )

        writer.writerow(
            [
                "retriever",
                "condition",
                "logical_model_id",
                "answers",
                "eligible_answers",
                "ineligible_answers",
                "eligibility_fraction",
                "mean_output_diversity",
            ]
        )

        for (
            retriever,
            condition,
            logical_model_id,
        ), agg in sorted(
            aggregate.items()
        ):
            answers = (
                agg["answers"]
            )

            eligible = (
                agg["eligible"]
            )

            ineligible = (
                answers - eligible
            )

            mean_diversity = (
                agg["diversity_sum"]
                / eligible
                if eligible
                else None
            )

            writer.writerow(
                [
                    retriever,
                    condition,
                    logical_model_id,
                    answers,
                    eligible,
                    ineligible,
                    eligible / answers,
                    mean_diversity,
                ]
            )

    manifest = {
        "artifact_format": (
            "context-matters."
            "phase45-output-diversity.v1"
        ),
        "dataset": "asqa",
        "input": str(
            args.input
        ),
        "input_sha256": sha256_file(
            args.input
        ),
        "output": str(
            args.output
        ),
        "output_sha256": sha256_file(
            args.output
        ),
        "summary": str(
            args.summary
        ),
        "summary_sha256": sha256_file(
            args.summary
        ),
        "model": args.model,
        "model_revision": (
            args.revision
        ),
        "tokenizer": args.model,
        "tokenizer_revision": (
            args.revision
        ),
        "model_loader": "AutoModel",
        "tokenizer_loader": (
            "AutoTokenizer"
        ),
        "pooling": (
            "attention-mask-aware "
            "mean pooling of "
            "last_hidden_state"
        ),
        "embedding_dimension": 768,
        "max_length": (
            args.max_length
        ),
        "truncation": True,
        "padding": (
            "dynamic to longest "
            "encoded input in batch"
        ),
        "compute_dtype": (
            "float32"
        ),
        "autocast": False,
        "embedding_dtype": (
            "float32"
        ),
        "metric": (
            "mean_pairwise_cosine_distance_"
            "among_answer_sentences"
        ),
        "metric_normalization": (
            "L2 normalize each sentence "
            "embedding before cosine similarity"
        ),
        "pair_selection": (
            "upper_triangle_k1_no_self_pairs"
        ),
        "answers": len(
            records
        ),
        "sentences": len(
            sentences
        ),
        "eligible_answers": (
            eligible_count
        ),
        "ineligible_answers": (
            ineligible_count
        ),
        "batch_size": (
            args.batch_size
        ),
        "device": (
            args.device
        ),
        "limit": (
            args.limit
        ),
        "python_version": (
            platform.python_version()
        ),
        "torch_version": (
            torch.__version__
        ),
        "transformers_version": (
            transformers.__version__
        ),
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
