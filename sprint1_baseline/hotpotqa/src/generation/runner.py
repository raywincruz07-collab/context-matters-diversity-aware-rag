"""Governed missing-only PubMedQA generation block runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from generation._io import file_sha256, sha256_text, stable_json_sha256, write_immutable_json
from generation.artifacts import (
    GenerationArtifactConflictError,
    GenerationStatus,
    build_generation_artifact,
    generation_artifact_path,
    generation_request_payload,
    generation_request_sha256,
    read_generation_artifact,
    write_generation_artifact,
)
from generation.maki import CanonicalMakiAdapter, PRIMARY_LLM_LOGICAL_IDS
from generation.prompts import PROMPT_BUNDLE_SHA256, render_pubmedqa_prompt
from generation.pubmedqa import classify_pubmedqa_response
from generation.repeatability import require_passing_repeatability_gate
from generation.selected_context import read_selected_context, read_selected_context_set
from retrieval_artifacts.contracts import canonical_stable_id
from run_registry import (
    DEFAULT_EVIDENCE_AUTHORITY_PATH,
    DEFAULT_REGISTRY_PATH,
    RUN_SCHEMA_VERSION,
    append_run_record,
    finalize_planned_record,
    output_artifact,
    output_inventory_sha256,
    read_registry,
    stable_json_sha256 as registry_sha256,
    validate_run_record,
)


GENERATION_OUTPUT_INVENTORY_SCHEMA = "sprint3.generation-output-inventory.v1"
GENERATION_OUTPUT_INVENTORY_FORMAT = "sprint3.generation-output-inventory-artifact.v1"
GENERATION_RUNNER_VERSION = "sprint3.pubmedqa-generation-runner.v1"
PUBMEDQA_EXPECTED_ROWS = 1000
CANONICAL_RETRIEVERS = ("bm25", "dpr", "contriever", "colbertv2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class GenerationBlock:
    context_mode: str
    llm_logical_id: str
    retriever: str | None = None

    def __post_init__(self) -> None:
        if self.context_mode not in {"with_context", "without_context"}:
            raise ValueError("context_mode is not canonical")
        if self.llm_logical_id not in PRIMARY_LLM_LOGICAL_IDS:
            raise ValueError("LLM is not a primary canonical generator")
        if self.context_mode == "without_context":
            if self.retriever is not None:
                raise ValueError("WITHOUT_CONTEXT is shared and retriever-independent")
        elif self.retriever not in CANONICAL_RETRIEVERS:
            raise ValueError("WITH_CONTEXT requires one canonical retriever")


def canonical_pubmedqa_generation_matrix() -> tuple[GenerationBlock, ...]:
    blocks = [
        GenerationBlock("without_context", llm)
        for llm in PRIMARY_LLM_LOGICAL_IDS
    ]
    blocks.extend(
        GenerationBlock("with_context", llm, retriever)
        for retriever in CANONICAL_RETRIEVERS
        for llm in PRIMARY_LLM_LOGICAL_IDS
    )
    return tuple(blocks)


def expected_matrix_row_count() -> int:
    return sum(PUBMEDQA_EXPECTED_ROWS for _ in canonical_pubmedqa_generation_matrix())


def decoding_payload(adapter: CanonicalMakiAdapter) -> dict[str, Any]:
    return {
        "decoding_version": "sprint3.pubmedqa.decoding.v1",
        "temperature": 0,
        "max_tokens": 256,
        "n": 1,
        "canonical_generation_replicas": 1,
        "direct_mode_status": adapter.config.direct_mode_status,
        "direct_mode_control": dict(adapter.config.direct_mode_control),
    }


def generation_protocol_bundle(adapter: CanonicalMakiAdapter) -> dict[str, Any]:
    decoding = decoding_payload(adapter)
    return {
        "runner_version": GENERATION_RUNNER_VERSION,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        "decoding": decoding,
        "decoding_sha256": stable_json_sha256(decoding),
        "retry_policy": {
            "maximum_total_infrastructure_attempts_per_request": 3,
            "content_retry": False,
        },
        "runtime_identity": adapter.config.runtime_identity(),
    }


def build_generation_planned_record(
    *,
    created_at: str,
    block: GenerationBlock,
    adapter: CanonicalMakiAdapter,
    git: Mapping[str, Any],
    sample_manifest_ref: Mapping[str, Any],
    corpus_manifest_ref: Mapping[str, Any] | None,
    retrieval: Mapping[str, Any] | None,
    environment_sha256: str,
    runtime_sha256: str,
    hardware_summary: str,
    output_directory: str,
    parent_run_id: str | None = None,
    evidence_authority_path: Path = DEFAULT_EVIDENCE_AUTHORITY_PATH,
) -> dict[str, Any]:
    if adapter.config.logical_model_id != block.llm_logical_id:
        raise ValueError("adapter and generation block LLM identities differ")
    if block.context_mode == "without_context":
        if corpus_manifest_ref is not None or retrieval is not None:
            raise ValueError("WITHOUT_CONTEXT cannot carry corpus/retrieval identity")
        diversification = None
    else:
        if corpus_manifest_ref is None or retrieval is None:
            raise ValueError("WITH_CONTEXT requires corpus/retrieval identity")
        if retrieval.get("retriever") != block.retriever:
            raise ValueError("retrieval identity does not match generation block")
        diversification = {
            "method": "none",
            "parameters": {},
            "config_sha256": registry_sha256({}),
            "seed": None,
        }
    bundle = generation_protocol_bundle(adapter)
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": created_at,
        "sprint": "sprint1",
        "stage": 1,
        "run_type": "GENERATION",
        "evidence_role": "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        "origin": "PROSPECTIVE_BACKFILL",
        "protocol_config_bundle_sha256": registry_sha256(bundle),
        "git": dict(git),
        "data": {
            "dataset": "pubmedqa",
            "split": "train",
            "source": "qiaojin/PubMedQA",
            "revision": "9001f2853fb87cab8d220904e0de81ac6973b318",
            "sample_manifest": dict(sample_manifest_ref),
            "corpus_manifest": (
                None if corpus_manifest_ref is None else dict(corpus_manifest_ref)
            ),
        },
        "retrieval": None if retrieval is None else dict(retrieval),
        "diversification": diversification,
        "context_mode": block.context_mode,
        "generation": {
            "llm_logical_id": block.llm_logical_id,
            "provider": "Mannheim Maki",
            "physical_model_id": adapter.config.physical_model_id,
            "model_revision": adapter.config.model_revision,
            "model_revision_kind": adapter.config.model_revision_kind,
            "prompt_sha256": PROMPT_BUNDLE_SHA256,
            "decoding_sha256": stable_json_sha256(decoding_payload(adapter)),
        },
        "evaluation": None,
        "execution": {
            "environment_sha256": environment_sha256,
            "runtime_sha256": runtime_sha256,
            "hardware_summary": hardware_summary,
            "started_at": None,
            "completed_at": None,
            "status": "PLANNED",
            "attempt_count": 0,
            "failure_reason": None,
            "parent_run_id": parent_run_id,
            "resume_of": None,
        },
        "output": {
            "expected_row_count": PUBMEDQA_EXPECTED_ROWS,
            "completed_row_count": 0,
            "successful_row_count": 0,
            "failed_row_count": 0,
            "output_directory": output_directory,
            "partial_output_retained": False,
            "artifacts": [],
            "output_inventory_sha256": None,
            "raw_artifact_sha256": None,
        },
    }
    return finalize_planned_record(
        record, evidence_authority_path=evidence_authority_path
    )


def _running_record(
    planned: Mapping[str, Any],
    *,
    attempt_count: int,
    started_at: str,
    prior_failure_reason: str | None,
    evidence_authority_path: Path,
) -> dict[str, Any]:
    updated = deepcopy(dict(planned))
    updated["execution"].update(
        {
            "started_at": started_at,
            "completed_at": None,
            "status": "RUNNING",
            "attempt_count": attempt_count,
            "failure_reason": prior_failure_reason,
            "resume_of": updated["run_id"] if attempt_count > 1 else None,
        }
    )
    return validate_run_record(updated, evidence_authority_path=evidence_authority_path)


def _selected_context_by_sample(
    selected_context_set: Mapping[str, Any],
) -> dict[str, str]:
    return {
        json.dumps(entry["sample_id"], sort_keys=True): entry["selected_context_id"]
        for entry in selected_context_set["scientific_payload"]["entries"]
    }


def _request_for_query(
    *,
    planned: Mapping[str, Any],
    adapter: CanonicalMakiAdapter,
    query: Any,
    selected_context: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Any]:
    if selected_context is None:
        prompt = render_pubmedqa_prompt(question=query.query_text)
        retriever = candidate_set_id = context_id = None
        condition = "WITHOUT_CONTEXT"
    else:
        scientific = selected_context["scientific_payload"]
        if scientific["sample_id"] != canonical_stable_id(query.sample_id, "sample_id"):
            raise ValueError("selected context sample identity mismatch")
        if scientific["query_text"] != query.query_text:
            raise ValueError("selected context query text mismatch")
        if scientific["retriever"] != planned["retrieval"]["retriever"]:
            raise ValueError("selected context retriever differs from run identity")
        if scientific["candidate_set_id"] != planned["retrieval"]["candidate_set"]["artifact_id"]:
            raise ValueError("selected context candidate set differs from run identity")
        prompt = render_pubmedqa_prompt(
            question=query.query_text,
            passage_bodies=tuple(
                item["passage_body"] for item in scientific["passages"]
            ),
        )
        if prompt.context_block != scientific["context_block"]:
            raise ValueError("selected context and prompt renderer differ")
        retriever = scientific["retriever"]
        candidate_set_id = scientific["candidate_set_id"]
        context_id = selected_context["selected_context_id"]
        condition = "WITH_CONTEXT"
    request = generation_request_payload(
        run_id=planned["run_id"],
        dataset="pubmedqa",
        evidence_role=planned["evidence_role"],
        sample_id=query.sample_id,
        question_text_sha256=sha256_text(query.query_text),
        llm_logical_id=adapter.config.logical_model_id,
        provider="Mannheim Maki",
        physical_model_id=adapter.config.physical_model_id,
        model_revision=adapter.config.model_revision,
        model_revision_kind=adapter.config.model_revision_kind,
        condition=condition,
        retriever=retriever,
        candidate_set_id=candidate_set_id,
        selected_context_id=context_id,
        prompt=prompt.provenance_payload(),
        decoding=decoding_payload(adapter),
    )
    return request, prompt


def generation_output_inventory(
    *,
    run_id: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [dict(item) for item in entries]
    values.sort(key=lambda item: json.dumps(item["sample_id"], sort_keys=True))
    status_counts = {status.value: 0 for status in GenerationStatus}
    for item in values:
        status_counts[item["status"]] += 1
    scientific = {
        "schema_version": GENERATION_OUTPUT_INVENTORY_SCHEMA,
        "run_id": run_id,
        "expected_row_count": PUBMEDQA_EXPECTED_ROWS,
        "completed_row_count": len(values),
        "status_counts": status_counts,
        "entries": values,
    }
    digest = stable_json_sha256(scientific)
    return {
        "artifact_format": GENERATION_OUTPUT_INVENTORY_FORMAT,
        "generation_output_inventory_id": f"generation-output:sha256:{digest}",
        "scientific_sha256": digest,
        "scientific_payload": scientific,
    }


def write_generation_output_inventory(
    path: Path,
    *,
    run_id: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    wrapper = generation_output_inventory(run_id=run_id, entries=entries)
    write_immutable_json(path, wrapper, conflict_error=GenerationArtifactConflictError)
    return wrapper


def _terminal_record(
    running: Mapping[str, Any],
    *,
    inventory_path: Path,
    inventory: Mapping[str, Any],
    repository_root: Path,
    completed_at: str,
    evidence_authority_path: Path,
) -> dict[str, Any]:
    status_counts = inventory["scientific_payload"]["status_counts"]
    failed = status_counts[GenerationStatus.ERROR.value]
    successful = PUBMEDQA_EXPECTED_ROWS - failed
    artifact = output_artifact(
        inventory_path,
        repository_root=repository_root,
        row_count=PUBMEDQA_EXPECTED_ROWS,
        status_counts=status_counts,
        artifact_id=inventory["generation_output_inventory_id"],
    )
    updated = deepcopy(dict(running))
    updated["execution"].update(
        {
            "completed_at": completed_at,
            "status": "COMPLETE",
            "failure_reason": None,
        }
    )
    updated["output"].update(
        {
            "completed_row_count": PUBMEDQA_EXPECTED_ROWS,
            "successful_row_count": successful,
            "failed_row_count": failed,
            "partial_output_retained": False,
            "artifacts": [artifact],
            "output_inventory_sha256": output_inventory_sha256([artifact]),
            "raw_artifact_sha256": file_sha256(inventory_path),
        }
    )
    return validate_run_record(updated, evidence_authority_path=evidence_authority_path)


def _failed_record(
    running: Mapping[str, Any],
    *,
    reason: str,
    completed_at: str,
    evidence_authority_path: Path,
) -> dict[str, Any]:
    updated = deepcopy(dict(running))
    updated["execution"].update(
        {"completed_at": completed_at, "status": "FAILED", "failure_reason": reason}
    )
    # A failed block never relabels partial per-row artifacts as complete run output.
    updated["output"].update(
        {
            "completed_row_count": 0,
            "successful_row_count": 0,
            "failed_row_count": 0,
            "partial_output_retained": False,
            "artifacts": [],
            "output_inventory_sha256": None,
            "raw_artifact_sha256": None,
        }
    )
    return validate_run_record(updated, evidence_authority_path=evidence_authority_path)


def _latest_for_run(registry_path: Path, run_id: str, evidence_authority_path: Path) -> dict[str, Any] | None:
    if not Path(registry_path).is_file():
        return None
    records = read_registry(registry_path, evidence_authority_path=evidence_authority_path)
    matching = [item for item in records if item["run_id"] == run_id]
    return None if not matching else matching[-1]


def execute_generation_block(
    *,
    planned_record: Mapping[str, Any],
    runtime: Any,
    adapter: CanonicalMakiAdapter,
    repeatability_gate_path: Path,
    all_model_runtime_identities: Mapping[str, Mapping[str, Any]],
    output_directory: Path,
    output_inventory_path: Path,
    environment: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    hardware_summary: str,
    selected_context_directory: Path | None = None,
    selected_context_set_path: Path | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    evidence_authority_path: Path = DEFAULT_EVIDENCE_AUTHORITY_PATH,
    repository_root: Path | None = None,
    clock=utc_now,
) -> dict[str, Any]:
    """Register, run, safely resume, and complete one independent 1,000-row block."""
    repository_root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    planned = validate_run_record(
        planned_record, evidence_authority_path=evidence_authority_path
    )
    if planned["execution"]["status"] != "PLANNED":
        raise ValueError("planned_record must be PLANNED")
    if len(runtime.ordered_queries) != PUBMEDQA_EXPECTED_ROWS:
        raise ValueError("canonical PubMedQA block requires exactly 1,000 queries")
    if (
        stable_json_sha256(environment)
        != planned["execution"]["environment_sha256"]
        or stable_json_sha256(runtime_provenance)
        != planned["execution"]["runtime_sha256"]
    ):
        raise ValueError("current environment/runtime differs from registered run")
    require_passing_repeatability_gate(
        repeatability_gate_path,
        model_runtime_identities=all_model_runtime_identities,
    )
    if adapter.config.runtime_identity() != dict(
        all_model_runtime_identities[adapter.config.logical_model_id]
    ):
        raise ValueError("adapter runtime differs from admitted repeatability identity")
    adapter.require_api_key()

    context_mode = planned["context_mode"]
    context_set = None
    context_ids: dict[str, str] = {}
    if context_mode == "with_context":
        if selected_context_directory is None or selected_context_set_path is None:
            raise ValueError("WITH_CONTEXT requires selected-context directory and set")
        context_set = read_selected_context_set(selected_context_set_path)
        if context_set["selected_context_set_id"] != planned["retrieval"]["selected_context"]["artifact_id"]:
            raise ValueError("registry selected-context set identity mismatch")
        if context_set["scientific_payload"]["candidate_set_id"] != planned["retrieval"]["candidate_set"]["artifact_id"]:
            raise ValueError("selected-context set candidate-set identity mismatch")
        context_ids = _selected_context_by_sample(context_set)
        if len(context_ids) != PUBMEDQA_EXPECTED_ROWS:
            raise ValueError("selected-context set must cover exactly 1,000 samples")
    elif selected_context_directory is not None or selected_context_set_path is not None:
        raise ValueError("WITHOUT_CONTEXT forbids selected-context inputs")

    latest = _latest_for_run(registry_path, planned["run_id"], evidence_authority_path)
    if latest is None:
        append_run_record(
            registry_path, planned, evidence_authority_path=evidence_authority_path
        )
        latest = planned
    elif latest["execution"]["status"] == "COMPLETE":
        return latest
    elif latest["execution"]["status"] == "FAILED":
        raise RuntimeError("governed generation run is terminal FAILED")
    if latest["execution"]["status"] == "PLANNED":
        attempt_count = 1
        prior_failure = None
    else:
        attempt_count = latest["execution"]["attempt_count"] + 1
        prior_failure = "resuming interrupted governed generation block"
        if attempt_count > 3:
            raise RuntimeError("governed generation block has exhausted three run attempts")
    running = _running_record(
        planned,
        attempt_count=attempt_count,
        started_at=clock(),
        prior_failure_reason=prior_failure,
        evidence_authority_path=evidence_authority_path,
    )
    append_run_record(
        registry_path, running, evidence_authority_path=evidence_authority_path
    )

    entries: list[dict[str, Any]] = []
    try:
        for query in runtime.ordered_queries:
            selected_wrapper = None
            if context_mode == "with_context":
                key = json.dumps(canonical_stable_id(query.sample_id, "sample_id"), sort_keys=True)
                expected_context_id = context_ids.get(key)
                if expected_context_id is None:
                    raise ValueError("selected-context set is missing a query")
                context_path = Path(selected_context_directory) / f"sample_{query.position:04d}.json"
                selected = read_selected_context(context_path)
                if selected.artifact_id != expected_context_id:
                    raise ValueError("selected-context file differs from aggregate set")
                selected_wrapper = selected.wrapper()
            request, prompt = _request_for_query(
                planned=planned,
                adapter=adapter,
                query=query,
                selected_context=selected_wrapper,
            )
            path = generation_artifact_path(output_directory, query.position)
            if path.exists():
                artifact = read_generation_artifact(path)
                if artifact["request_sha256"] != generation_request_sha256(request):
                    raise GenerationArtifactConflictError(
                        f"existing generation row has conflicting request identity: {path}"
                    )
                provenance = artifact["execution_provenance"]
                if (
                    provenance["environment_sha256"] != stable_json_sha256(environment)
                    or provenance["runtime_sha256"] != stable_json_sha256(runtime_provenance)
                ):
                    raise GenerationArtifactConflictError(
                        f"existing generation row has incompatible execution provenance: {path}"
                    )
            else:
                created_at = clock()
                completion = adapter.complete(prompt)
                status, parsed = classify_pubmedqa_response(
                    raw_content=completion.raw_content,
                    finish_reason=completion.finish_reason,
                    provider_refusal=completion.provider_refusal,
                    transport_exhausted=completion.transport_exhausted,
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
                    hardware_summary=hardware_summary,
                    created_at=created_at,
                    completed_at=clock(),
                )
                write_generation_artifact(artifact, path)
            entries.append(
                {
                    "sample_id": canonical_stable_id(query.sample_id, "sample_id"),
                    "path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
                    "sha256": file_sha256(path),
                    "generation_artifact_id": artifact["generation_artifact_id"],
                    "status": artifact["observation"]["status"],
                    "request_sha256": artifact["request_sha256"],
                }
            )
        inventory = write_generation_output_inventory(
            output_inventory_path, run_id=planned["run_id"], entries=entries
        )
        terminal = _terminal_record(
            running,
            inventory_path=output_inventory_path,
            inventory=inventory,
            repository_root=repository_root,
            completed_at=clock(),
            evidence_authority_path=evidence_authority_path,
        )
        append_run_record(
            registry_path, terminal, evidence_authority_path=evidence_authority_path
        )
        return terminal
    except Exception as exc:
        failed = _failed_record(
            running,
            reason=f"{type(exc).__name__}: {str(exc)[:500]}",
            completed_at=clock(),
            evidence_authority_path=evidence_authority_path,
        )
        append_run_record(
            registry_path, failed, evidence_authority_path=evidence_authority_path
        )
        raise


def inspect_generation_block(
    *,
    planned_record: Mapping[str, Any],
    output_directory: Path,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    evidence_authority_path: Path = DEFAULT_EVIDENCE_AUTHORITY_PATH,
) -> dict[str, Any]:
    planned = validate_run_record(
        planned_record, evidence_authority_path=evidence_authority_path
    )
    latest = _latest_for_run(registry_path, planned["run_id"], evidence_authority_path)
    counts = {status.value: 0 for status in GenerationStatus}
    invalid: list[str] = []
    present = 0
    for position in range(PUBMEDQA_EXPECTED_ROWS):
        path = generation_artifact_path(output_directory, position)
        if not path.exists():
            continue
        try:
            artifact = read_generation_artifact(path)
            if artifact["request"]["run_id"] != planned["run_id"]:
                raise ValueError("row run ID mismatch")
            counts[artifact["observation"]["status"]] += 1
            present += 1
        except Exception as exc:
            invalid.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "run_id": planned["run_id"],
        "registry_status": None if latest is None else latest["execution"]["status"],
        "expected_rows": PUBMEDQA_EXPECTED_ROWS,
        "valid_terminal_rows": present,
        "missing_rows": PUBMEDQA_EXPECTED_ROWS - present,
        "status_counts": counts,
        "invalid_artifacts": invalid,
    }
