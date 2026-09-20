#!/usr/bin/env python3
"""Resumable canonical HotpotQA WITHOUT_CONTEXT maKI generation."""

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
EXPECTED_TOTAL = EXPECTED_ROWS * 3


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


def read_rows(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)

            if row["status"] != "READY":
                raise ValueError(
                    f"unexpected input status at position {row['position']}"
                )

            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} rows, found {len(rows)}"
        )

    return rows


def output_path(root: Path, logical_id: str, position: int) -> Path:
    safe_model = logical_id.replace("/", "_")
    return root / safe_model / f"sample_{position:04d}.json"


def validate_existing(
    path: Path,
    *,
    row: dict,
    logical_id: str,
    physical_model_id: str,
    rendered_prompt_sha256: str,
) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))

    expected = {
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "query_text_sha256": row["query_text_sha256"],
        "logical_model_id": logical_id,
        "physical_model_id": physical_model_id,
        "rendered_prompt_sha256": rendered_prompt_sha256,
    }

    for key, value in expected.items():
        if obj.get(key) != value:
            raise ValueError(
                f"resume identity mismatch in {path}: {key}"
            )


def generate_one(
    *,
    row: dict,
    logical_id: str,
    bindings: dict,
    output_root: Path,
) -> str:
    adapter = adapter_from_bindings(bindings, logical_id)

    prompt = render_hotpotqa_prompt(
        question=row["query_text"],
        passage_bodies=None,
    )

    path = output_path(
        output_root,
        logical_id,
        row["position"],
    )

    if path.exists():
        validate_existing(
            path,
            row=row,
            logical_id=logical_id,
            physical_model_id=adapter.config.physical_model_id,
            rendered_prompt_sha256=prompt.rendered_prompt_sha256,
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
            "sprint3.hotpotqa-no-context-generation.v1"
        ),
        "dataset": "hotpotqa",
        "evidence_role": "OFFICIAL_TEST_FULL",
        "condition": "WITHOUT_CONTEXT",
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "query_text": row["query_text"],
        "query_text_sha256": row["query_text_sha256"],
        "logical_model_id": logical_id,
        "physical_model_id": adapter.config.physical_model_id,
        "model_revision": adapter.config.model_revision,
        "model_revision_kind": adapter.config.model_revision_kind,
        "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
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
        "--input",
        type=Path,
        default=Path(
            "data/hotpotqa/generation_package_2026-09-06/"
            "without_context_queries.jsonl"
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
            "without_context"
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

    rows = read_rows(args.input)

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

    print(
        f"START rows={len(rows)} models={len(logical_ids)} "
        f"requests={total} concurrency={args.concurrency}",
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
                    f"FAILED position={position} "
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
        "PASS: HOTPOTQA WITHOUT_CONTEXT GENERATION COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
