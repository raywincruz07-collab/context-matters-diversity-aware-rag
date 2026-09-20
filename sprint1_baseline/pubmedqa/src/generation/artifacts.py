"""Immutable canonical generation-row artifact contract."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from generation._io import stable_json_sha256, write_immutable_json
from retrieval_artifacts.contracts import canonical_stable_id


GENERATION_SCHEMA_VERSION = "sprint3.generation-row.v1"
GENERATION_ARTIFACT_FORMAT = "sprint3.generation-row-artifact.v1"
_RUN_ID_RE = re.compile(r"^run-[a-z0-9][a-z0-9-]*-[0-9a-f]{24}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_SET_RE = re.compile(r"^candidate-set:sha256:[0-9a-f]{64}$")
_SELECTED_CONTEXT_RE = re.compile(r"^selected-context:sha256:[0-9a-f]{64}$")


class GenerationStatus(str, Enum):
    OK = "OK"
    REFUSAL = "REFUSAL"
    PARSE_FAILURE = "PARSE_FAILURE"
    TRUNCATED = "TRUNCATED"
    ERROR = "ERROR"


class GenerationArtifactConflictError(ValueError):
    """An existing generation artifact has different immutable content."""


def _require_text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
    return value


def _require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def generation_request_payload(
    *,
    run_id: str,
    dataset: str,
    evidence_role: str,
    sample_id: str | int,
    question_text_sha256: str,
    llm_logical_id: str,
    provider: str,
    physical_model_id: str,
    model_revision: str | None,
    model_revision_kind: str,
    condition: str,
    retriever: str | None,
    candidate_set_id: str | None,
    selected_context_id: str | None,
    prompt: Mapping[str, Any],
    decoding: Mapping[str, Any],
) -> dict[str, Any]:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run_id is not canonical")
    for name, value in (
        ("dataset", dataset),
        ("evidence_role", evidence_role),
        ("llm_logical_id", llm_logical_id),
        ("provider", provider),
        ("physical_model_id", physical_model_id),
        ("model_revision_kind", model_revision_kind),
    ):
        _require_text(value, name)
    canonical_sample = canonical_stable_id(sample_id, "sample_id")
    _require_sha(question_text_sha256, "question_text_sha256")
    if model_revision is not None:
        _require_text(model_revision, "model_revision")
    if model_revision_kind == "NOT_PROVIDED_BY_PROVIDER":
        if model_revision is not None:
            raise ValueError("NOT_PROVIDED_BY_PROVIDER requires null revision")
    elif model_revision_kind not in {"IMMUTABLE_REVISION", "PROVIDER_SNAPSHOT"}:
        raise ValueError("model_revision_kind is invalid")
    elif model_revision is None:
        raise ValueError("physical revision is required for this revision kind")
    if condition == "WITHOUT_CONTEXT":
        if any(value is not None for value in (retriever, candidate_set_id, selected_context_id)):
            raise ValueError("WITHOUT_CONTEXT forbids retrieval identity")
    elif condition == "WITH_CONTEXT":
        _require_text(retriever, "retriever")
        if not isinstance(candidate_set_id, str) or _CANDIDATE_SET_RE.fullmatch(candidate_set_id) is None:
            raise ValueError("WITH_CONTEXT requires a candidate-set ID")
        if not isinstance(selected_context_id, str) or _SELECTED_CONTEXT_RE.fullmatch(selected_context_id) is None:
            raise ValueError("WITH_CONTEXT requires a selected-context ID")
    else:
        raise ValueError("condition is not canonical")
    prompt_value = dict(prompt)
    decoding_value = dict(decoding)
    for name in (
        "prompt_protocol_version",
        "prompt_bundle_sha256",
        "system_message_sha256",
        "user_template_sha256",
        "user_message_sha256",
        "rendered_prompt_sha256",
    ):
        if name not in prompt_value:
            raise ValueError(f"prompt provenance is missing {name}")
    for name in (
        "prompt_bundle_sha256",
        "system_message_sha256",
        "user_template_sha256",
        "user_message_sha256",
        "rendered_prompt_sha256",
    ):
        _require_sha(prompt_value[name], f"prompt.{name}")
    if condition == "WITH_CONTEXT":
        _require_sha(prompt_value.get("context_block_sha256"), "context_block_sha256")
    elif prompt_value.get("context_block_sha256") is not None:
        raise ValueError("WITHOUT_CONTEXT prompt cannot carry context hash")
    if decoding_value.get("temperature") != 0:
        raise ValueError("canonical temperature must be zero")
    if decoding_value.get("max_tokens") != 256:
        raise ValueError("canonical PubMedQA max_tokens must be 256")
    if decoding_value.get("n") != 1:
        raise ValueError("canonical completion count must be one")
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": dataset,
        "evidence_role": evidence_role,
        "sample_id": canonical_sample,
        "question_text_sha256": question_text_sha256,
        "llm": {
            "logical_id": llm_logical_id,
            "provider": provider,
            "physical_model_id": physical_model_id,
            "model_revision": model_revision,
            "model_revision_kind": model_revision_kind,
        },
        "condition": condition,
        "retriever": retriever,
        "candidate_set_id": candidate_set_id,
        "selected_context_id": selected_context_id,
        "prompt": prompt_value,
        "decoding": decoding_value,
    }


def generation_request_sha256(request: Mapping[str, Any]) -> str:
    return stable_json_sha256(dict(request))


def _validate_attempts(attempts: Sequence[Mapping[str, Any]], status: GenerationStatus) -> list[dict[str, Any]]:
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        raise TypeError("attempts must be a sequence")
    values = [dict(item) for item in attempts]
    if not 1 <= len(values) <= 3:
        raise ValueError("generation must preserve one to three attempts")
    for index, item in enumerate(values, start=1):
        if set(item) != {
            "attempt",
            "started_at",
            "completed_at",
            "outcome",
            "error_type",
            "error_message",
        }:
            raise ValueError("attempt metadata keys mismatch")
        if item["attempt"] != index:
            raise ValueError("attempt numbers must be contiguous starting at one")
        for timestamp_name in ("started_at", "completed_at"):
            _require_text(item[timestamp_name], f"attempt.{timestamp_name}")
        if item["outcome"] not in {"SUCCESS", "INFRASTRUCTURE_ERROR"}:
            raise ValueError("attempt outcome is invalid")
        if item["outcome"] == "SUCCESS":
            if item["error_type"] is not None or item["error_message"] is not None:
                raise ValueError("successful attempt cannot carry an error")
        else:
            _require_text(item["error_type"], "attempt.error_type")
            _require_text(item["error_message"], "attempt.error_message")
    if status is GenerationStatus.ERROR:
        if len(values) != 3 or any(item["outcome"] != "INFRASTRUCTURE_ERROR" for item in values):
            raise ValueError("ERROR requires exactly three failed infrastructure attempts")
    elif values[-1]["outcome"] != "SUCCESS":
        raise ValueError("non-ERROR status requires a successful final attempt")
    return values


def build_generation_artifact(
    *,
    request: Mapping[str, Any],
    status: GenerationStatus | str,
    raw_content: str | None,
    finish_reason: str | None,
    provider_metadata: Mapping[str, Any],
    parsed_output: Mapping[str, Any] | None,
    attempts: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    runtime: Mapping[str, Any],
    hardware_summary: str,
    created_at: str,
    completed_at: str,
) -> dict[str, Any]:
    status_value = status if isinstance(status, GenerationStatus) else GenerationStatus(status)
    request_value = dict(request)
    if request_value.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ValueError("generation request schema mismatch")
    attempt_values = _validate_attempts(attempts, status_value)
    if status_value is GenerationStatus.ERROR:
        if raw_content is not None or finish_reason is not None or parsed_output is not None:
            raise ValueError("ERROR must not fabricate provider content")
    else:
        _require_text(raw_content, "raw_content", allow_empty=True)
    if parsed_output is not None and not isinstance(parsed_output, Mapping):
        raise TypeError("parsed_output must be an object or null")
    if status_value is GenerationStatus.OK and parsed_output is None:
        raise ValueError("OK requires parsed output")
    if status_value is not GenerationStatus.OK and parsed_output is not None:
        raise ValueError("only OK may carry parsed output")
    _require_text(hardware_summary, "hardware_summary")
    _require_text(created_at, "created_at")
    _require_text(completed_at, "completed_at")
    environment_value = dict(environment)
    runtime_value = dict(runtime)
    environment_sha = stable_json_sha256(environment_value)
    runtime_sha = stable_json_sha256(runtime_value)
    observation = {
        "status": status_value.value,
        "raw_content": raw_content,
        "finish_reason": finish_reason,
        "provider_metadata": dict(provider_metadata),
        "parsed_output": None if parsed_output is None else dict(parsed_output),
    }
    identity_payload = {
        "request": request_value,
        "observation": observation,
        "attempt_outcomes": [
            {
                "attempt": item["attempt"],
                "outcome": item["outcome"],
                "error_type": item["error_type"],
                "error_message": item["error_message"],
            }
            for item in attempt_values
        ],
        "environment_sha256": environment_sha,
        "runtime_sha256": runtime_sha,
    }
    digest = stable_json_sha256(identity_payload)
    return {
        "artifact_format": GENERATION_ARTIFACT_FORMAT,
        "generation_artifact_id": f"generation:sha256:{digest}",
        "scientific_sha256": digest,
        "request_sha256": generation_request_sha256(request_value),
        "request": request_value,
        "observation": observation,
        "execution_provenance": {
            "environment": environment_value,
            "environment_sha256": environment_sha,
            "runtime": runtime_value,
            "runtime_sha256": runtime_sha,
            "hardware_summary": hardware_summary,
            "attempts": attempt_values,
            "created_at": created_at,
            "completed_at": completed_at,
        },
    }


def _rebuild(stored: Mapping[str, Any]) -> dict[str, Any]:
    execution = stored["execution_provenance"]
    observation = stored["observation"]
    return build_generation_artifact(
        request=stored["request"],
        status=observation["status"],
        raw_content=observation["raw_content"],
        finish_reason=observation["finish_reason"],
        provider_metadata=observation["provider_metadata"],
        parsed_output=observation["parsed_output"],
        attempts=execution["attempts"],
        environment=execution["environment"],
        runtime=execution["runtime"],
        hardware_summary=execution["hardware_summary"],
        created_at=execution["created_at"],
        completed_at=execution["completed_at"],
    )


def read_generation_artifact(path: Path) -> dict[str, Any]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping):
        raise ValueError("generation artifact must be an object")
    if set(stored) != {
        "artifact_format",
        "generation_artifact_id",
        "scientific_sha256",
        "request_sha256",
        "request",
        "observation",
        "execution_provenance",
    }:
        raise ValueError("generation artifact wrapper keys mismatch")
    if stored["artifact_format"] != GENERATION_ARTIFACT_FORMAT:
        raise ValueError("generation artifact format mismatch")
    rebuilt = _rebuild(stored)
    if dict(stored) != rebuilt:
        raise ValueError("generation artifact identity mismatch")
    return rebuilt


def write_generation_artifact(artifact: Mapping[str, Any], path: Path) -> None:
    rebuilt = _rebuild(artifact)
    if dict(artifact) != rebuilt:
        raise ValueError("generation artifact is not canonical")
    write_immutable_json(
        path,
        rebuilt,
        conflict_error=GenerationArtifactConflictError,
    )


def generation_artifact_path(output_directory: Path, position: int) -> Path:
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("position must be a nonnegative integer")
    return Path(output_directory) / f"sample_{position:04d}.json"
