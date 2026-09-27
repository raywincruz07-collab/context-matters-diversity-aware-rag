#!/usr/bin/env python3
"""Generate PubMedQA Sprint-2 answers from already-frozen diversified Top5 contexts.

This runner performs NO retrieval and NO diversification.
It consumes the existing:
  Top20 -> diversify -> Top5
or
  Top50 -> diversify -> Top5
artifacts and sends exactly five passage bodies to maKI.

Outputs are missing-only/resumable and preserve the existing frozen PubMedQA
prompt, model bindings, decoding, response parsing, and infrastructure retry policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generation._io import (
    file_sha256,
    sha256_text,
    stable_json_sha256,
)
from generation.artifacts import (
    build_generation_artifact,
    generation_request_payload,
    generation_request_sha256,
    read_generation_artifact,
    write_generation_artifact,
)
from generation.cli_support import (
    adapter_from_bindings,
    load_model_bindings,
    load_pubmedqa_runtime_local_only,
    provenance_hashes,
)
from generation.prompts import (
    PROMPT_BUNDLE_SHA256,
    render_pubmedqa_prompt,
)
from generation.pubmedqa import classify_pubmedqa_response
from generation.repeatability import require_passing_repeatability_gate
from generation.runner import decoding_payload
from retrieval_artifacts.contracts import canonical_stable_id


RETRIEVERS = ("bm25", "dpr", "contriever", "colbertv2")

# All actual diversified conditions currently present in the frozen artifacts.
# "none" is intentionally excluded because Sprint 1 already contains the
# relevance baseline.
DIVERSIFIED_CONDITIONS = (
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
)

LLMS = (
    "llama-3.3-70b",
    "gemma4-26b",
    "ministral-3-14b",
)

EXPECTED = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", choices=("top20", "top50"), required=True)
    p.add_argument("--retriever", choices=RETRIEVERS, required=True)
    p.add_argument("--condition", choices=DIVERSIFIED_CONDITIONS, required=True)
    p.add_argument("--llm", choices=LLMS, required=True)

    p.add_argument(
        "--model-bindings",
        type=Path,
        default=ROOT / "configs/sprint3/maki_model_bindings_v7.json",
    )
    p.add_argument(
        "--repeatability-gate",
        type=Path,
        default=ROOT / "artifacts/generation_repeatability/repeatability_gate_v6.json",
    )
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--confirm-api-calls",
        default=None,
    )
    return p.parse_args()


def canonical_key(value):
    return json.dumps(
        canonical_stable_id(value, "sample_id"),
        sort_keys=True,
        separators=(",", ":"),
    )


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value


def diversified_path(pool: str, retriever: str) -> Path:
    base = ROOT / "artifacts/diversified/pubmedqa" / retriever

    if pool == "top20":
        return (
            base
            / f"{retriever}_pubmedqa_canonical_top20_to5_v1.jsonl"
        )

    return (
        base
        / f"{retriever}_pubmedqa_sensitivity_top50_to5_v1.jsonl"
    )


def load_condition_rows(path: Path, condition: str, pool_size: int):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if obj["condition"] != condition:
                continue

            if obj["candidate_pool"] != pool_size:
                raise RuntimeError(
                    f"candidate_pool mismatch in {path}"
                )

            if obj["top_k"] != 5:
                raise RuntimeError(
                    f"top_k mismatch in {path}"
                )

            if len(obj["selected"]) != 5:
                raise RuntimeError(
                    f"selected context is not Top5 in {path}"
                )

            selected = sorted(
                obj["selected"],
                key=lambda x: int(x["selected_rank"]),
            )

            if [int(x["selected_rank"]) for x in selected] != [1, 2, 3, 4, 5]:
                raise RuntimeError(
                    "selected ranks must be exactly 1..5"
                )

            if len(
                {str(x["document_id"]) for x in selected}
            ) != 5:
                raise RuntimeError(
                    "selected Top5 contains duplicate documents"
                )

            obj["selected"] = selected
            rows.append(obj)

    if len(rows) != EXPECTED:
        raise RuntimeError(
            f"{condition}: expected {EXPECTED} rows, found {len(rows)}"
        )

    rows.sort(key=lambda x: int(x["position"]))

    if [int(x["position"]) for x in rows] != list(range(EXPECTED)):
        raise RuntimeError(
            "diversified artifact does not cover positions 0..999 exactly"
        )

    return rows


def make_run_id(
    *,
    pool,
    retriever,
    condition,
    llm,
    source_sha,
    physical_model,
):
    identity = {
        "dataset": "pubmedqa",
        "sprint": 2,
        "pool": pool,
        "retriever": retriever,
        "condition": condition,
        "llm": llm,
        "physical_model": physical_model,
        "source_sha256": source_sha,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
    }

    digest = stable_json_sha256(identity)[:24]

    name = "-".join(
        [
            "sprint2",
            "pubmedqa",
            pool,
            slug(retriever),
            slug(condition),
            slug(llm),
        ]
    )

    return f"run-{name}-{digest}"


def main():
    args = parse_args()

    pool_size = 20 if args.pool == "top20" else 50

    source = diversified_path(
        args.pool,
        args.retriever,
    )

    if not source.is_file():
        raise FileNotFoundError(source)

    source_sha = file_sha256(source)

    rows = load_condition_rows(
        source,
        args.condition,
        pool_size,
    )

    runtime = load_pubmedqa_runtime_local_only(
        cache_dir=args.cache_dir,
    )

    if len(runtime.ordered_queries) != EXPECTED:
        raise RuntimeError(
            "PubMedQA runtime does not contain exactly 1000 queries"
        )

    bindings = load_model_bindings(
        args.model_bindings,
    )

    adapter = adapter_from_bindings(
        bindings,
        args.llm,
        max_tokens=256,
    )

    identities = {
        logical_id: config.runtime_identity()
        for logical_id, config in bindings.items()
    }

    require_passing_repeatability_gate(
        args.repeatability_gate,
        model_runtime_identities=identities,
    )

    environment, runtime_provenance, _, _ = provenance_hashes(
        adapter
    )

    run_id = make_run_id(
        pool=args.pool,
        retriever=args.retriever,
        condition=args.condition,
        llm=args.llm,
        source_sha=source_sha,
        physical_model=adapter.config.physical_model_id,
    )

    candidate_identity = {
        "dataset": "pubmedqa",
        "candidate_pool": pool_size,
        "retriever": args.retriever,
        "source_artifact_sha256": source_sha,
    }

    candidate_set_id = (
        "candidate-set:sha256:"
        + stable_json_sha256(candidate_identity)
    )

    output_dir = (
        ROOT
        / "results/sprint2/pubmedqa/generation"
        / args.pool
        / args.retriever
        / args.condition
        / args.llm
    )

    print("=== PUBMEDQA SPRINT 2 GENERATION ===")
    print("RUN_ID:", run_id)
    print("POOL:", args.pool, f"({pool_size})")
    print("RETRIEVER:", args.retriever)
    print("CONDITION:", args.condition)
    print("LOGICAL_LLM:", args.llm)
    print("PHYSICAL_LLM:", adapter.config.physical_model_id)
    print("SOURCE:", source)
    print("SOURCE_SHA256:", source_sha)
    print("SOURCE_ROWS:", len(rows))
    print("FINAL_CONTEXT_SIZE: 5")
    print("PROMPT_BUNDLE_SHA256:", PROMPT_BUNDLE_SHA256)
    print("OUTPUT:", output_dir)

    # Validate every selected passage against the canonical runtime corpus
    # before any API call.
    prepared = []

    for row, query in zip(
        rows,
        runtime.ordered_queries,
        strict=True,
    ):
        if int(row["position"]) != query.position:
            raise RuntimeError(
                "query/diversification position mismatch"
            )

        if canonical_key(row["sample_id"]) != canonical_key(query.sample_id):
            raise RuntimeError(
                f"sample identity mismatch at position {query.position}"
            )

        bodies = []
        ordered_ids = []

        for selected in row["selected"]:
            cp = int(selected["corpus_position"])

            if cp < 0 or cp >= len(runtime.corpus_records):
                raise RuntimeError(
                    f"invalid corpus position {cp}"
                )

            record = runtime.corpus_records[cp]

            if int(record.corpus_position) != cp:
                raise RuntimeError(
                    "runtime corpus position mismatch"
                )

            if str(record.document_id) != str(selected["document_id"]):
                raise RuntimeError(
                    f"document mismatch at sample {query.position}"
                )

            body = record.text

            if not isinstance(body, str) or not body or body != body.strip():
                raise RuntimeError(
                    "invalid canonical passage body"
                )

            bodies.append(body)
            ordered_ids.append(selected["document_id"])

        if len(bodies) != 5:
            raise RuntimeError(
                "prepared context does not contain exactly 5 passages"
            )

        selected_identity = {
            "dataset": "pubmedqa",
            "sample_id": canonical_stable_id(
                query.sample_id,
                "sample_id",
            ),
            "candidate_pool": pool_size,
            "retriever": args.retriever,
            "condition": args.condition,
            "ordered_document_ids": ordered_ids,
            "source_artifact_sha256": source_sha,
        }

        selected_context_id = (
            "selected-context:sha256:"
            + stable_json_sha256(selected_identity)
        )

        prompt = render_pubmedqa_prompt(
            question=query.query_text,
            passage_bodies=tuple(bodies),
        )

        request = generation_request_payload(
            run_id=run_id,
            dataset="pubmedqa",
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            sample_id=query.sample_id,
            question_text_sha256=sha256_text(
                query.query_text
            ),
            llm_logical_id=adapter.config.logical_model_id,
            provider="Mannheim Maki",
            physical_model_id=adapter.config.physical_model_id,
            model_revision=adapter.config.model_revision,
            model_revision_kind=adapter.config.model_revision_kind,
            condition="WITH_CONTEXT",
            retriever=args.retriever,
            candidate_set_id=candidate_set_id,
            selected_context_id=selected_context_id,
            prompt=prompt.provenance_payload(),
            decoding=decoding_payload(adapter),
        )

        prepared.append(
            (
                query,
                prompt,
                request,
                selected_context_id,
                ordered_ids,
            )
        )

    print("PREPARED_ROWS:", len(prepared))

    if args.dry_run:
        q, prompt, request, context_id, ids = prepared[0]

        print("FIRST_SAMPLE_ID:", q.sample_id)
        print("FIRST_SELECTED_DOCUMENT_IDS:", ids)
        print("FIRST_SELECTED_CONTEXT_ID:", context_id)
        print(
            "FIRST_REQUEST_SHA256:",
            generation_request_sha256(request),
        )
        print(
            "FIRST_RENDERED_PROMPT_SHA256:",
            prompt.rendered_prompt_sha256,
        )
        print("API_CALLS_MADE: 0")
        print("DRY_RUN: PASS")
        return

    if (
        args.confirm_api_calls
        != "I_UNDERSTAND_THIS_MAKES_MODEL_REQUESTS"
    ):
        raise RuntimeError(
            "API execution requires "
            "--confirm-api-calls "
            "I_UNDERSTAND_THIS_MAKES_MODEL_REQUESTS"
        )

    adapter.require_api_key()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_counts = Counter()
    start = time.time()

    for index, (
        query,
        prompt,
        request,
        selected_context_id,
        ordered_ids,
    ) in enumerate(prepared, start=1):

        path = output_dir / f"sample_{query.position:04d}.json"

        if path.exists():
            artifact = read_generation_artifact(path)

            if artifact["request_sha256"] != generation_request_sha256(
                request
            ):
                raise RuntimeError(
                    f"resume conflict: {path}"
                )
        else:
            created_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            )

            completion = adapter.complete(prompt)

            status, parsed = classify_pubmedqa_response(
                raw_content=completion.raw_content,
                finish_reason=completion.finish_reason,
                provider_refusal=completion.provider_refusal,
                transport_exhausted=completion.transport_exhausted,
            )

            completed_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            )

            artifact = build_generation_artifact(
                request=request,
                status=status,
                raw_content=completion.raw_content,
                finish_reason=completion.finish_reason,
                provider_metadata=completion.provider_metadata,
                parsed_output=parsed,
                attempts=completion.attempts,
                environment=environment,
                runtime=runtime_provenance,
                hardware_summary=(
                    f"platform={platform.platform()}"
                ),
                created_at=created_at,
                completed_at=completed_at,
            )

            write_generation_artifact(
                artifact,
                path,
            )

        status_counts[
            artifact["observation"]["status"]
        ] += 1

        if index % 50 == 0 or index == EXPECTED:
            elapsed = time.time() - start
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (
                (EXPECTED - index) / rate
                if rate > 0
                else 0.0
            )

            print(
                f"{index}/{EXPECTED} "
                f"({100*index/EXPECTED:.1f}%) "
                f"rate={rate:.2f} samples/s "
                f"ETA={eta/60:.1f} min",
                flush=True,
            )

    if sum(status_counts.values()) != EXPECTED:
        raise RuntimeError(
            "final generation count is not 1000"
        )

    manifest = {
        "schema_version":
            "context-matters.sprint2.pubmedqa-generation-run.v1",
        "run_id": run_id,
        "dataset": "pubmedqa",
        "sprint": 2,
        "candidate_pool_label": args.pool,
        "candidate_pool": pool_size,
        "final_top_k": 5,
        "retriever": args.retriever,
        "diversification_condition": args.condition,
        "logical_llm": args.llm,
        "physical_llm": adapter.config.physical_model_id,
        "source_diversification_artifact": str(
            source.relative_to(ROOT)
        ),
        "source_diversification_sha256": source_sha,
        "candidate_set_id": candidate_set_id,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        "temperature": 0,
        "max_tokens": 256,
        "seed": adapter.config.seed,
        "expected_rows": EXPECTED,
        "completed_rows": EXPECTED,
        "status_counts": dict(status_counts),
        "model_bindings_path": str(
            args.model_bindings.relative_to(ROOT)
        ),
        "model_bindings_sha256": file_sha256(
            args.model_bindings
        ),
        "repeatability_gate_path": str(
            args.repeatability_gate.relative_to(ROOT)
        ),
        "repeatability_gate_sha256": file_sha256(
            args.repeatability_gate
        ),
        "output_directory": str(
            output_dir.relative_to(ROOT)
        ),
    }

    manifest["scientific_sha256"] = stable_json_sha256(
        manifest
    )

    manifest_path = output_dir / "run_manifest.json"

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, manifest_path)

    print("STATUS_COUNTS:", dict(status_counts))
    print("RUN_MANIFEST:", manifest_path)
    print("RUN_MANIFEST_SHA256:", file_sha256(manifest_path))
    print("SPRINT2_BLOCK: COMPLETE")


if __name__ == "__main__":
    main()
