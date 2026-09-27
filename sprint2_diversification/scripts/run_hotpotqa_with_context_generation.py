#!/usr/bin/env python3
"""Resumable canonical HotpotQA WITH_CONTEXT maKI generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
import time

from generation.cli_support import (
    adapter_from_bindings,
    load_model_bindings,
)
from generation.hotpotqa import classify_hotpotqa_response
from generation.hotpotqa_prompts import render_hotpotqa_prompt
from generation.maki import PRIMARY_LLM_LOGICAL_IDS


EXPECTED_ROWS = 7_405

INPUT_FILES = {
    "bm25": "bm25_with_context.jsonl",
    "dpr": "dpr_with_context.jsonl",
    "contriever": "contriever_with_context.jsonl",
    "colbertv2": "colbertv2_with_context.jsonl",
}

EXPECTED_READY = {
    "bm25": 7_405,
    "dpr": 7_405,
    "contriever": 7_399,
    "colbertv2": 7_405,
}

EXPECTED_MISSING = {
    "bm25": 0,
    "dpr": 0,
    "contriever": 6,
    "colbertv2": 0,
}


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)

        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


def read_rows(path: Path, retriever: str) -> list[dict]:
    rows: list[dict] = []
    ready = 0
    missing = 0

    with path.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            row = json.loads(line)

            if row["position"] != position:
                raise ValueError(
                    f"position mismatch at input row {position}"
                )

            if row["condition"] != "WITH_CONTEXT":
                raise ValueError(
                    f"condition mismatch at position {position}"
                )

            if row["retriever"] != retriever:
                raise ValueError(
                    f"retriever mismatch at position {position}"
                )

            status = row["status"]

            if status == "READY":
                if row["selected_context_id"] is None:
                    raise ValueError(
                        f"READY row lacks selected context: {position}"
                    )

                selected = row["selected_context"]

                if selected is None:
                    raise ValueError(
                        f"READY row lacks selected wrapper: {position}"
                    )

                scientific = selected["scientific_payload"]

                if len(scientific["passages"]) != 5:
                    raise ValueError(
                        f"READY row must have five passages: {position}"
                    )

                bodies = tuple(
                    passage["passage_body"]
                    for passage in scientific["passages"]
                )

                if any(not body for body in bodies):
                    raise ValueError(
                        f"READY row contains empty body: {position}"
                    )

                prompt = render_hotpotqa_prompt(
                    question=row["query_text"],
                    passage_bodies=bodies,
                )

                if prompt.context_block != row["context_block"]:
                    raise ValueError(
                        f"context block mismatch: {position}"
                    )

                if (
                    scientific["context_block"]
                    != row["context_block"]
                ):
                    raise ValueError(
                        f"selected context block mismatch: {position}"
                    )

                ready += 1

            elif status == "UPSTREAM_CONTEXT_MISSING":
                if retriever != "contriever":
                    raise ValueError(
                        "upstream context missing is only expected "
                        "for Contriever"
                    )

                if row["selected_context_id"] is not None:
                    raise ValueError(
                        f"missing row has selected context: {position}"
                    )

                if row["context_block"] is not None:
                    raise ValueError(
                        f"missing row has context block: {position}"
                    )

                if not row["missing_passages"]:
                    raise ValueError(
                        f"missing row lacks missing passage evidence: "
                        f"{position}"
                    )

                missing += 1

            else:
                raise ValueError(
                    f"unexpected input status {status!r} "
                    f"at position {position}"
                )

            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} rows, found {len(rows)}"
        )

    if ready != EXPECTED_READY[retriever]:
        raise ValueError(
            f"{retriever} READY count mismatch: {ready}"
        )

    if missing != EXPECTED_MISSING[retriever]:
        raise ValueError(
            f"{retriever} missing count mismatch: {missing}"
        )

    return rows


def output_path(
    root: Path,
    retriever: str,
    logical_id: str,
    position: int,
) -> Path:
    safe_model = logical_id.replace("/", "_")

    return (
        root
        / retriever
        / safe_model
        / f"sample_{position:04d}.json"
    )


def validate_existing(
    path: Path,
    *,
    row: dict,
    retriever: str,
    logical_id: str,
    physical_model_id: str,
) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))

    expected = {
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "query_text_sha256": row["query_text_sha256"],
        "retriever": retriever,
        "candidate_set_id": row["candidate_set_id"],
        "selected_context_id": row["selected_context_id"],
        "logical_model_id": logical_id,
        "physical_model_id": physical_model_id,
    }

    for key, value in expected.items():
        if obj.get(key) != value:
            raise ValueError(
                f"resume identity mismatch in {path}: {key}"
            )

    if row["status"] == "UPSTREAM_CONTEXT_MISSING":
        if obj.get("status") != "UPSTREAM_CONTEXT_MISSING":
            raise ValueError(
                f"missingness resume mismatch in {path}"
            )
        if obj.get("maki_request") is not False:
            raise ValueError(
                f"missingness artifact claims maKI request in {path}"
            )

    return "RESUMED"


def write_missing(
    *,
    row: dict,
    retriever: str,
    logical_id: str,
    physical_model_id: str,
    model_revision: str | None,
    model_revision_kind: str,
    output_root: Path,
) -> str:
    path = output_path(
        output_root,
        retriever,
        logical_id,
        row["position"],
    )

    if path.exists():
        return validate_existing(
            path,
            row=row,
            retriever=retriever,
            logical_id=logical_id,
            physical_model_id=physical_model_id,
        )

    artifact = {
        "artifact_format": (
            "sprint3.hotpotqa-with-context-generation.v1"
        ),
        "dataset": "hotpotqa",
        "evidence_role": "OFFICIAL_TEST_FULL",
        "condition": "WITH_CONTEXT",
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "query_text": row["query_text"],
        "query_text_sha256": row["query_text_sha256"],
        "retriever": retriever,
        "candidate_set_id": row["candidate_set_id"],
        "candidate_artifact_id": row["candidate_artifact_id"],
        "selected_context_id": None,
        "logical_model_id": logical_id,
        "physical_model_id": physical_model_id,
        "model_revision": model_revision,
        "model_revision_kind": model_revision_kind,
        "status": "UPSTREAM_CONTEXT_MISSING",
        "missing_passages": row["missing_passages"],
        "maki_request": False,
        "parsed_output": None,
        "raw_content": None,
        "finish_reason": None,
        "provider_metadata": {},
        "attempts": [],
    }

    atomic_write_json(path, artifact)

    return "UPSTREAM_CONTEXT_MISSING"


def generate_one(
    *,
    row: dict,
    retriever: str,
    logical_id: str,
    bindings: dict,
    output_root: Path,
) -> str:
    adapter = adapter_from_bindings(bindings, logical_id)

    if row["status"] == "UPSTREAM_CONTEXT_MISSING":
        return write_missing(
            row=row,
            retriever=retriever,
            logical_id=logical_id,
            physical_model_id=adapter.config.physical_model_id,
            model_revision=adapter.config.model_revision,
            model_revision_kind=adapter.config.model_revision_kind,
            output_root=output_root,
        )

    scientific = row["selected_context"]["scientific_payload"]

    bodies = tuple(
        passage["passage_body"]
        for passage in scientific["passages"]
    )

    prompt = render_hotpotqa_prompt(
        question=row["query_text"],
        passage_bodies=bodies,
    )

    if prompt.context_block != row["context_block"]:
        raise ValueError(
            f"context block mismatch at position {row['position']}"
        )

    path = output_path(
        output_root,
        retriever,
        logical_id,
        row["position"],
    )

    if path.exists():
        validate_existing(
            path,
            row=row,
            retriever=retriever,
            logical_id=logical_id,
            physical_model_id=adapter.config.physical_model_id,
        )

        obj = json.loads(path.read_text(encoding="utf-8"))

        if (
            obj.get("rendered_prompt_sha256")
            != prompt.rendered_prompt_sha256
        ):
            raise ValueError(
                f"prompt identity mismatch in {path}"
            )

        return "RESUMED"

    started = time.time()

    completion = adapter.complete(prompt)

    status, parsed = classify_hotpotqa_response(
        raw_content=completion.raw_content,
        finish_reason=completion.finish_reason,
    )

    artifact = {
        "artifact_format": (
            "sprint3.hotpotqa-with-context-generation.v1"
        ),
        "dataset": "hotpotqa",
        "evidence_role": "OFFICIAL_TEST_FULL",
        "condition": "WITH_CONTEXT",
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "query_text": row["query_text"],
        "query_text_sha256": row["query_text_sha256"],
        "retriever": retriever,
        "candidate_set_id": row["candidate_set_id"],
        "candidate_artifact_id": row["candidate_artifact_id"],
        "selected_context_id": row["selected_context_id"],
        "context_block_sha256": (
            scientific["context_block_sha256"]
        ),
        "logical_model_id": logical_id,
        "physical_model_id": adapter.config.physical_model_id,
        "model_revision": adapter.config.model_revision,
        "model_revision_kind": adapter.config.model_revision_kind,
        "rendered_prompt_sha256": (
            prompt.rendered_prompt_sha256
        ),
        "prompt_provenance": prompt.provenance_payload(),
        "decoding": {
            "temperature": 0,
            "max_tokens": 256,
            "n": 1,
            "direct_mode_status": (
                adapter.config.direct_mode_status
            ),
            "direct_mode_control": dict(
                adapter.config.direct_mode_control
            ),
        },
        "status": status.value,
        "maki_request": True,
        "parsed_output": parsed,
        "raw_content": completion.raw_content,
        "finish_reason": completion.finish_reason,
        "provider_metadata": completion.provider_metadata,
        "attempts": list(completion.attempts),
        "elapsed_seconds": time.time() - started,
    }

    atomic_write_json(path, artifact)

    return status.value


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retriever",
        required=True,
        choices=tuple(INPUT_FILES),
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(
            "data/hotpotqa/"
            "generation_package_2026-09-06"
        ),
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=Path(
            "configs/sprint3/maki_model_bindings_v7.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/hotpotqa/"
            "generation_outputs_2026-09-06/"
            "with_context"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not os.environ.get("MAKI_API_KEY"):
        raise RuntimeError("MAKI_API_KEY is not set")

    if args.concurrency != 16:
        raise ValueError(
            "canonical production concurrency is frozen at 16"
        )

    input_path = (
        args.package_root
        / INPUT_FILES[args.retriever]
    )

    rows = read_rows(input_path, args.retriever)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    bindings = load_model_bindings(args.bindings)

    logical_ids = tuple(PRIMARY_LLM_LOGICAL_IDS)

    if len(logical_ids) != 3:
        raise ValueError("expected exactly three primary LLMs")

    jobs = [
        (row, logical_id)
        for row in rows
        for logical_id in logical_ids
    ]

    total = len(jobs)

    ready_calls = sum(
        1
        for row, _ in jobs
        if row["status"] == "READY"
    )
    missing_cells = total - ready_calls

    print(
        f"START retriever={args.retriever} "
        f"rows={len(rows)} "
        f"models={len(logical_ids)} "
        f"cells={total} "
        f"maki_requests={ready_calls} "
        f"missing_cells={missing_cells} "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    counts: dict[str, int] = {}

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {
            executor.submit(
                generate_one,
                row=row,
                retriever=args.retriever,
                logical_id=logical_id,
                bindings=bindings,
                output_root=args.output_root,
            ): (row["position"], logical_id)
            for row, logical_id in jobs
        }

        completed = 0

        for future in as_completed(futures):
            position, logical_id = futures[future]

            try:
                outcome = future.result()
            except Exception as exc:
                print(
                    f"FAILED retriever={args.retriever} "
                    f"position={position} "
                    f"model={logical_id} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise

            counts[outcome] = counts.get(outcome, 0) + 1
            completed += 1

            if completed % 100 == 0 or completed == total:
                print(
                    f"PROGRESS retriever={args.retriever} "
                    f"{completed}/{total} "
                    f"{json.dumps(counts, sort_keys=True)}",
                    flush=True,
                )

    print(
        f"PASS: HOTPOTQA WITH_CONTEXT {args.retriever} COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
