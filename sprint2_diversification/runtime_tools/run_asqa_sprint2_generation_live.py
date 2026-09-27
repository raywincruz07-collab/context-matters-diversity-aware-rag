#!/usr/bin/env python3
"""Resumable operational launcher for frozen ASQA Sprint-2 maKI generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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
EXPECTED_RETRIEVERS = ("bm25", "dpr", "contriever")
EXPECTED_CONDITIONS = (
    "mmr_0",
    "mmr_0.25",
    "mmr_0.5",
    "mmr_0.75",
    "kmeans_k2",
    "agglo_k3",
    "dpp_map",
)
EXPECTED_FILES = len(EXPECTED_RETRIEVERS) * len(EXPECTED_CONDITIONS)
EXPECTED_CONTEXT_ROWS = EXPECTED_FILES * EXPECTED_ROWS
EXPECTED_MODELS = 3
EXPECTED_REQUESTS = EXPECTED_CONTEXT_ROWS * EXPECTED_MODELS


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_manifest(package_root: Path) -> tuple[dict, list[dict]]:
    manifest_path = package_root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("artifact_format") != (
        "context-matters.sprint2-asqa-generation-package.v1"
    ):
        raise ValueError("unexpected package artifact format")

    if manifest.get("dataset") != "asqa":
        raise ValueError("unexpected dataset")

    if manifest.get("evidence_role") != "PROJECT_PROTECTED_FINAL":
        raise ValueError("unexpected evidence role")

    if tuple(manifest.get("retrievers", ())) != EXPECTED_RETRIEVERS:
        raise ValueError("retriever set/order mismatch")

    if tuple(manifest.get("new_generation_conditions", ())) != (
        EXPECTED_CONDITIONS
    ):
        raise ValueError("new-generation condition set/order mismatch")

    counts = manifest.get("counts", {})

    if counts.get("new_maki_input_rows") != EXPECTED_CONTEXT_ROWS:
        raise ValueError(
            "unexpected frozen new_maki_input_rows"
        )

    entries = [
        entry
        for entry in manifest["files"]
        if entry.get("generation_action") == "RUN_MAKI"
    ]

    if len(entries) != EXPECTED_FILES:
        raise ValueError(
            f"expected {EXPECTED_FILES} RUN_MAKI files, "
            f"found {len(entries)}"
        )

    seen = set()

    for entry in entries:
        retriever = entry["retriever"]
        condition = entry["condition"]

        identity = (retriever, condition)

        if identity in seen:
            raise ValueError(f"duplicate package entry: {identity}")

        seen.add(identity)

        if retriever not in EXPECTED_RETRIEVERS:
            raise ValueError(f"unexpected retriever: {retriever}")

        if condition not in EXPECTED_CONDITIONS:
            raise ValueError(f"unexpected condition: {condition}")

        if entry["rows"] != EXPECTED_ROWS:
            raise ValueError(
                f"{identity}: expected {EXPECTED_ROWS} rows, "
                f"found {entry['rows']}"
            )

        path = package_root / entry["relative_path"]

        if not path.is_file():
            raise FileNotFoundError(path)

        actual_sha = sha256_file(path)

        if actual_sha != entry["sha256"]:
            raise ValueError(
                f"{identity}: package file SHA mismatch"
            )

    expected_seen = {
        (retriever, condition)
        for retriever in EXPECTED_RETRIEVERS
        for condition in EXPECTED_CONDITIONS
    }

    if seen != expected_seen:
        raise ValueError("RUN_MAKI matrix is incomplete")

    return manifest, entries


def load_rows(
    package_root: Path,
    entry: dict,
) -> list[dict]:
    path = package_root / entry["relative_path"]

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"{entry['relative_path']}: expected "
            f"{EXPECTED_ROWS} rows, found {len(rows)}"
        )

    retriever = entry["retriever"]
    condition = entry["condition"]

    for position, row in enumerate(rows):
        if row.get("dataset") != "asqa":
            raise ValueError(
                f"{retriever}/{condition}/{position}: dataset mismatch"
            )

        if row.get("evidence_role") != "PROJECT_PROTECTED_FINAL":
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "evidence-role mismatch"
            )

        if row.get("position") != position:
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "position mismatch"
            )

        if row.get("retriever") != retriever:
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "retriever mismatch"
            )

        if row.get("condition") != condition:
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "condition mismatch"
            )

        if row.get("status") != "READY":
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "non-READY row"
            )

        if row.get("selected_k") != 5:
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "selected_k mismatch"
            )

        query_text = row.get("query_text")

        if not isinstance(query_text, str) or not query_text:
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "invalid query text"
            )

        if sha256_text(query_text) != row.get("query_text_sha256"):
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "query hash mismatch"
            )

        passages = row.get("passages")

        if not isinstance(passages, list) or len(passages) != 5:
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "expected exactly five passages"
            )

        bodies = []

        for rank, passage in enumerate(passages, start=1):
            if passage.get("rank") != rank:
                raise ValueError(
                    f"{retriever}/{condition}/{position}: "
                    f"passage rank mismatch at {rank}"
                )

            body = passage.get("passage_body")

            if not isinstance(body, str) or not body:
                raise ValueError(
                    f"{retriever}/{condition}/{position}: "
                    f"empty passage at rank {rank}"
                )

            if body != body.strip():
                raise ValueError(
                    f"{retriever}/{condition}/{position}: "
                    f"noncanonical passage whitespace at rank {rank}"
                )

            bodies.append(body)

        prompt = render_asqa_prompt(
            question=query_text,
            passage_bodies=tuple(bodies),
        )

        if prompt.context_block != row.get("context_block"):
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "rendered context block mismatch"
            )

        if sha256_text(row["context_block"]) != (
            row.get("context_block_sha256")
        ):
            raise ValueError(
                f"{retriever}/{condition}/{position}: "
                "context-block SHA mismatch"
            )

    return rows


def output_path(
    output_root: Path,
    retriever: str,
    condition: str,
    logical_id: str,
    position: int,
) -> Path:
    safe_model = logical_id.replace("/", "_")

    return (
        output_root
        / retriever
        / condition
        / safe_model
        / f"sample_{position:04d}.json"
    )


def validate_existing(
    path: Path,
    *,
    row: dict,
    entry: dict,
    logical_id: str,
    physical_model_id: str,
    rendered_prompt_sha256: str,
) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))

    expected = {
        "dataset": "asqa",
        "evidence_role": "PROJECT_PROTECTED_FINAL",
        "retriever": row["retriever"],
        "condition": row["condition"],
        "position": row["position"],
        "sample_id": row["sample_id"],
        "query_text_sha256": row["query_text_sha256"],
        "context_block_sha256": row["context_block_sha256"],
        "source_input_sha256": entry["sha256"],
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

    if decoding.get("temperature") != 0:
        raise ValueError(
            f"resume artifact temperature mismatch: {path}"
        )

    if decoding.get("max_tokens") != 512:
        raise ValueError(
            f"resume artifact max_tokens mismatch: {path}"
        )

    return "RESUMED"


def generate_one(
    *,
    row: dict,
    entry: dict,
    logical_id: str,
    bindings: dict,
    output_root: Path,
) -> str:
    adapter = adapter_from_bindings(
        bindings,
        logical_id,
        max_tokens=512,
    )

    bodies = tuple(
        passage["passage_body"]
        for passage in row["passages"]
    )

    prompt = render_asqa_prompt(
        question=row["query_text"],
        passage_bodies=bodies,
    )

    if prompt.context_block != row["context_block"]:
        raise ValueError(
            f"context mismatch at "
            f"{row['retriever']}/{row['condition']}/"
            f"{row['position']}"
        )

    path = output_path(
        output_root,
        row["retriever"],
        row["condition"],
        logical_id,
        row["position"],
    )

    if path.exists():
        return validate_existing(
            path,
            row=row,
            entry=entry,
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

    passage_ids = []

    for passage in row["passages"]:
        passage_id = passage.get("passage_id")

        if passage_id is None:
            passage_id = passage.get("document_id")

        if passage_id is None:
            passage_id = passage.get("id")

        passage_ids.append(passage_id)

    artifact = {
        "artifact_format":
            "context-matters.sprint2-asqa-generation.v1",

        "dataset":
            "asqa",

        "evidence_role":
            "PROJECT_PROTECTED_FINAL",

        "retriever":
            row["retriever"],

        "condition":
            row["condition"],

        "position":
            row["position"],

        "sample_id":
            row["sample_id"],

        "query_text":
            row["query_text"],

        "query_text_sha256":
            row["query_text_sha256"],

        "context_block_sha256":
            row["context_block_sha256"],

        "selected_k":
            5,

        "passage_ids":
            passage_ids,

        "source_diversified_artifact_sha256":
            row.get("source_diversified_artifact_sha256"),

        "source_input_relative_path":
            entry["relative_path"],

        "source_input_sha256":
            entry["sha256"],

        "logical_model_id":
            logical_id,

        "physical_model_id":
            adapter.config.physical_model_id,

        "model_revision":
            adapter.config.model_revision,

        "model_revision_kind":
            adapter.config.model_revision_kind,

        "rendered_prompt_sha256":
            prompt.rendered_prompt_sha256,

        "prompt_provenance":
            prompt.provenance_payload(),

        "decoding": {
            "temperature": 0,
            "max_tokens": 512,
            "n": 1,
            "direct_mode_status":
                adapter.config.direct_mode_status,
            "direct_mode_control":
                dict(adapter.config.direct_mode_control),
        },

        "status":
            status.value,

        "parsed_output":
            parsed,

        "raw_content":
            completion.raw_content,

        "finish_reason":
            completion.finish_reason,

        "provider_metadata":
            completion.provider_metadata,

        "attempts":
            list(completion.attempts),

        "elapsed_seconds":
            time.time() - started,
    }

    atomic_write_json(path, artifact)

    return status.value


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--package-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bindings",
        type=Path,
        required=True,
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
        "--limit-per-file",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.concurrency != 16:
        raise ValueError(
            "canonical production concurrency is frozen at 16"
        )

    if args.limit_per_file is not None:
        if args.limit_per_file <= 0:
            raise ValueError(
                "--limit-per-file must be positive"
            )

    manifest, entries = load_manifest(args.package_root)

    loaded = []

    print(
        "VALIDATING_FROZEN_ASQA_SPRINT2_PACKAGE",
        flush=True,
    )

    for entry in entries:
        rows = load_rows(
            args.package_root,
            entry,
        )

        loaded.append(
            (entry, rows)
        )

        print(
            f"INPUT_OK retriever={entry['retriever']} "
            f"condition={entry['condition']} "
            f"rows={len(rows)} "
            f"sha256={entry['sha256']}",
            flush=True,
        )

    bindings = load_model_bindings(args.bindings)
    logical_ids = tuple(PRIMARY_LLM_LOGICAL_IDS)

    if len(logical_ids) != EXPECTED_MODELS:
        raise ValueError(
            "expected exactly three primary LLMs"
        )

    if not os.environ.get("MAKI_API_KEY"):
        raise RuntimeError(
            "MAKI_API_KEY is not set in this shell"
        )

    for logical_id in logical_ids:
        adapter = adapter_from_bindings(
            bindings,
            logical_id,
            max_tokens=512,
        )

        adapter.require_api_key()

        print(
            f"MODEL_OK logical={logical_id} "
            f"physical={adapter.config.physical_model_id} "
            f"revision={adapter.config.model_revision!r} "
            f"revision_kind="
            f"{adapter.config.model_revision_kind!r}",
            flush=True,
        )

    print(
        f"FROZEN_CONTEXT_ROWS={EXPECTED_CONTEXT_ROWS}",
        flush=True,
    )
    print(
        f"FROZEN_MODELS={EXPECTED_MODELS}",
        flush=True,
    )
    print(
        f"FROZEN_TOTAL_MAKI_CALLS={EXPECTED_REQUESTS}",
        flush=True,
    )

    if args.preflight_only:
        print(
            "ASQA_SPRINT2_LIVE_RUNNER_PREFLIGHT=PASS",
            flush=True,
        )
        print(
            "NETWORK_REQUESTS_MADE=0",
            flush=True,
        )
        return 0

    jobs = []

    for entry, rows in loaded:
        selected_rows = rows

        if args.limit_per_file is not None:
            selected_rows = rows[:args.limit_per_file]

        for row in selected_rows:
            for logical_id in logical_ids:
                jobs.append(
                    (
                        entry,
                        row,
                        logical_id,
                    )
                )

    total = len(jobs)

    if args.limit_per_file is None:
        if total != EXPECTED_REQUESTS:
            raise ValueError(
                f"expected {EXPECTED_REQUESTS} requests, "
                f"found {total}"
            )

    print(
        f"START_ASQA_SPRINT2 "
        f"input_files={len(entries)} "
        f"models={len(logical_ids)} "
        f"requests={total} "
        f"max_tokens=512 "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    counts: dict[str, int] = {}
    failures = 0

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {}

        for entry, row, logical_id in jobs:
            future = executor.submit(
                generate_one,
                row=row,
                entry=entry,
                logical_id=logical_id,
                bindings=bindings,
                output_root=args.output_root,
            )

            futures[future] = (
                row["retriever"],
                row["condition"],
                row["position"],
                logical_id,
            )

        completed = 0

        for future in as_completed(futures):
            (
                retriever,
                condition,
                position,
                logical_id,
            ) = futures[future]

            try:
                outcome = future.result()
                counts[outcome] = (
                    counts.get(outcome, 0) + 1
                )

            except Exception as exc:
                failures += 1

                print(
                    f"FAILED retriever={retriever} "
                    f"condition={condition} "
                    f"position={position} "
                    f"model={logical_id} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            completed += 1

            if completed % 100 == 0 or completed == total:
                print(
                    f"PROGRESS {completed}/{total} "
                    f"failures={failures} "
                    f"{json.dumps(counts, sort_keys=True)}",
                    flush=True,
                )

    print(
        f"COMPLETE requests={total} "
        f"failures={failures} "
        f"counts={json.dumps(counts, sort_keys=True)}",
        flush=True,
    )

    if failures:
        print(
            "ASQA_SPRINT2_GENERATION=INCOMPLETE_RESUMABLE",
            flush=True,
        )
        return 2

    print(
        "ASQA_SPRINT2_GENERATION=PASS",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
