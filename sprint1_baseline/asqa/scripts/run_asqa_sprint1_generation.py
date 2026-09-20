#!/usr/bin/env python3
"""Resumable canonical ASQA Sprint-1 maKI generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
import time

from generation.asqa import classify_asqa_response
from generation.asqa_prompts import render_asqa_prompt
from generation.cli_support import adapter_from_bindings, load_model_bindings
from generation.maki import PRIMARY_LLM_LOGICAL_IDS


EXPECTED_ROWS = 948
EXPECTED_MODELS = 3
EXPECTED_CONDITIONS = 4
EXPECTED_TOTAL = (
    EXPECTED_ROWS * EXPECTED_MODELS * EXPECTED_CONDITIONS
)

INPUT_FILES = {
    "without_context": "without_context_queries.jsonl",
    "dpr": "dpr_with_context.jsonl",
    "bm25": "bm25_with_context.jsonl",
    "contriever": "contriever_with_context.jsonl",
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


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_inputs(package_root: Path) -> dict[str, list[dict]]:
    result = {
        key: load_jsonl(package_root / filename)
        for key, filename in INPUT_FILES.items()
    }

    for key, rows in result.items():
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(
                f"{key}: expected {EXPECTED_ROWS} rows, found {len(rows)}"
            )

        for position, row in enumerate(rows):
            if row["position"] != position:
                raise ValueError(
                    f"{key}: position mismatch at {position}"
                )

            if row["status"] != "READY":
                raise ValueError(
                    f"{key}: non-READY row at {position}"
                )

            if row["dataset"] != "asqa":
                raise ValueError(
                    f"{key}: dataset mismatch at {position}"
                )

            if row["evidence_role"] != "PROJECT_PROTECTED_FINAL":
                raise ValueError(
                    f"{key}: evidence role mismatch at {position}"
                )

    baseline = result["without_context"]

    for position, row in enumerate(baseline):
        if row["condition"] != "WITHOUT_CONTEXT":
            raise ValueError(
                f"without_context condition mismatch at {position}"
            )

    for retriever in ("dpr", "bm25", "contriever"):
        rows = result[retriever]

        for position, row in enumerate(rows):
            base = baseline[position]

            if row["condition"] != "WITH_CONTEXT":
                raise ValueError(
                    f"{retriever}: condition mismatch at {position}"
                )

            if row["retriever"] != retriever:
                raise ValueError(
                    f"{retriever}: retriever mismatch at {position}"
                )

            if row["sample_id"] != base["sample_id"]:
                raise ValueError(
                    f"{retriever}: sample mismatch at {position}"
                )

            if row["query_id"] != base["query_id"]:
                raise ValueError(
                    f"{retriever}: query ID mismatch at {position}"
                )

            if row["query_text"] != base["query_text"]:
                raise ValueError(
                    f"{retriever}: question mismatch at {position}"
                )

            if row["query_text_sha256"] != base["query_text_sha256"]:
                raise ValueError(
                    f"{retriever}: question hash mismatch at {position}"
                )

            passages = row["passages"]

            if row["selected_k"] != 5 or len(passages) != 5:
                raise ValueError(
                    f"{retriever}: expected exactly five passages "
                    f"at {position}"
                )

            for rank, passage in enumerate(passages, start=1):
                if passage["rank"] != rank:
                    raise ValueError(
                        f"{retriever}: passage rank mismatch at {position}"
                    )

                body = passage["passage_body"]

                if not isinstance(body, str) or not body:
                    raise ValueError(
                        f"{retriever}: empty passage at {position}"
                    )

                if body != body.strip():
                    raise ValueError(
                        f"{retriever}: noncanonical outer whitespace "
                        f"at {position}"
                    )

            bodies = tuple(
                passage["passage_body"]
                for passage in passages
            )

            prompt = render_asqa_prompt(
                question=row["query_text"],
                passage_bodies=bodies,
            )

            if prompt.context_block != row["context_block"]:
                raise ValueError(
                    f"{retriever}: context block mismatch at {position}"
                )

    return result


def output_path(
    root: Path,
    source_key: str,
    logical_id: str,
    position: int,
) -> Path:
    safe_model = logical_id.replace("/", "_")

    if source_key == "without_context":
        return (
            root
            / "without_context"
            / safe_model
            / f"sample_{position:04d}.json"
        )

    return (
        root
        / "with_context"
        / source_key
        / safe_model
        / f"sample_{position:04d}.json"
    )


def validate_existing(
    path: Path,
    *,
    row: dict,
    source_key: str,
    logical_id: str,
    physical_model_id: str,
    rendered_prompt_sha256: str,
) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))

    expected = {
        "dataset": "asqa",
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "query_text_sha256": row["query_text_sha256"],
        "condition": row["condition"],
        "retriever": (
            None if source_key == "without_context" else source_key
        ),
        "logical_model_id": logical_id,
        "physical_model_id": physical_model_id,
        "rendered_prompt_sha256": rendered_prompt_sha256,
    }

    for key, value in expected.items():
        if obj.get(key) != value:
            raise ValueError(
                f"resume identity mismatch in {path}: {key}"
            )

    decoding = obj.get("decoding")

    if not isinstance(decoding, dict):
        raise ValueError(
            f"resume artifact lacks decoding block: {path}"
        )

    if decoding.get("max_tokens") != 512:
        raise ValueError(
            f"resume artifact has wrong ASQA token limit: {path}"
        )

    return "RESUMED"


def generate_one(
    *,
    row: dict,
    source_key: str,
    logical_id: str,
    bindings: dict,
    output_root: Path,
) -> str:
    adapter = adapter_from_bindings(
        bindings,
        logical_id,
        max_tokens=512,
    )

    if source_key == "without_context":
        passages = None
        retriever = None
    else:
        passages = tuple(
            passage["passage_body"]
            for passage in row["passages"]
        )
        retriever = source_key

    prompt = render_asqa_prompt(
        question=row["query_text"],
        passage_bodies=passages,
    )

    if source_key != "without_context":
        if prompt.context_block != row["context_block"]:
            raise ValueError(
                f"context block mismatch at position {row['position']}"
            )

    path = output_path(
        output_root,
        source_key,
        logical_id,
        row["position"],
    )

    if path.exists():
        return validate_existing(
            path,
            row=row,
            source_key=source_key,
            logical_id=logical_id,
            physical_model_id=adapter.config.physical_model_id,
            rendered_prompt_sha256=prompt.rendered_prompt_sha256,
        )

    started = time.time()

    completion = adapter.complete(prompt)

    status, parsed = classify_asqa_response(
        raw_content=completion.raw_content,
        finish_reason=completion.finish_reason,
        provider_refusal=completion.provider_refusal,
        transport_exhausted=completion.transport_exhausted,
    )

    artifact = {
        "artifact_format": "sprint3.asqa-sprint1-generation.v1",
        "dataset": "asqa",
        "evidence_role": "PROJECT_PROTECTED_FINAL",
        "condition": row["condition"],
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "query_text": row["query_text"],
        "query_text_sha256": row["query_text_sha256"],
        "retriever": retriever,
        "selected_k": (
            None
            if source_key == "without_context"
            else 5
        ),
        "passage_ids": (
            None
            if source_key == "without_context"
            else [
                passage["passage_id"]
                for passage in row["passages"]
            ]
        ),
        "logical_model_id": logical_id,
        "physical_model_id": adapter.config.physical_model_id,
        "model_revision": adapter.config.model_revision,
        "model_revision_kind": adapter.config.model_revision_kind,
        "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
        "prompt_provenance": prompt.provenance_payload(),
        "decoding": {
            "temperature": 0,
            "max_tokens": 512,
            "n": 1,
            "direct_mode_status": (
                adapter.config.direct_mode_status
            ),
            "direct_mode_control": dict(
                adapter.config.direct_mode_control
            ),
        },
        "status": status.value,
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
        "--package-root",
        type=Path,
        required=True,
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
        required=True,
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

    inputs = read_inputs(args.package_root)

    bindings = load_model_bindings(args.bindings)
    logical_ids = tuple(PRIMARY_LLM_LOGICAL_IDS)

    if len(logical_ids) != EXPECTED_MODELS:
        raise ValueError("expected exactly three primary LLMs")

    # Secret-only preflight. No network request is made here.
    for logical_id in logical_ids:
        adapter_from_bindings(
            bindings,
            logical_id,
            max_tokens=512,
        ).require_api_key()

    jobs = []

    for source_key, rows in inputs.items():
        selected_rows = rows

        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            selected_rows = rows[: args.limit]

        for row in selected_rows:
            for logical_id in logical_ids:
                jobs.append(
                    (source_key, row, logical_id)
                )

    total = len(jobs)

    if args.limit is None and total != EXPECTED_TOTAL:
        raise ValueError(
            f"expected {EXPECTED_TOTAL} requests, found {total}"
        )

    print(
        f"START conditions={len(inputs)} "
        f"rows_per_condition="
        f"{EXPECTED_ROWS if args.limit is None else args.limit} "
        f"models={len(logical_ids)} "
        f"requests={total} "
        f"max_tokens=512 "
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
                source_key=source_key,
                logical_id=logical_id,
                bindings=bindings,
                output_root=args.output_root,
            ): (
                source_key,
                row["position"],
                logical_id,
            )
            for source_key, row, logical_id in jobs
        }

        completed = 0

        for future in as_completed(futures):
            source_key, position, logical_id = futures[future]

            try:
                outcome = future.result()
            except Exception as exc:
                print(
                    f"FAILED source={source_key} "
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
                    f"PROGRESS {completed}/{total} "
                    f"{json.dumps(counts, sort_keys=True)}",
                    flush=True,
                )

    print(
        "PASS: ASQA SPRINT1 GENERATION COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
