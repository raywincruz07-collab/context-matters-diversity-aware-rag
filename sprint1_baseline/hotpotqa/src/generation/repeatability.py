"""Machine-readable 20-prompt by 3-call generator repeatability gate."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from generation._io import stable_json_sha256, write_immutable_json
from generation.maki import CanonicalMakiAdapter, PRIMARY_LLM_LOGICAL_IDS
from generation.prompts import PROMPT_PROTOCOL_VERSION, RenderedPrompt
from retrieval_artifacts.contracts import canonical_stable_id


REPEATABILITY_PROMPT_MANIFEST_SCHEMA = "sprint3.generation-repeatability-prompts.v1"
REPEATABILITY_PROMPT_MANIFEST_FORMAT = "sprint3.generation-repeatability-prompts-artifact.v1"
REPEATABILITY_GATE_SCHEMA = "sprint3.generation-repeatability-gate.v1"
REPEATABILITY_GATE_FORMAT = "sprint3.generation-repeatability-gate-artifact.v1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class RepeatabilityGateError(ValueError):
    """Repeatability input or admission failure."""


def repeatability_prompt_manifest_payload(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(entries) != 20:
        raise RepeatabilityGateError("repeatability manifest requires exactly 20 prompts")
    values: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(entries):
        if set(raw) != {
            "prompt_id",
            "dataset",
            "evidence_role",
            "sample_id",
            "prompt",
        }:
            raise RepeatabilityGateError("repeatability prompt entry keys mismatch")
        prompt_id = raw["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id.strip() or prompt_id in seen_ids:
            raise RepeatabilityGateError("repeatability prompt IDs must be unique")
        seen_ids.add(prompt_id)
        if raw["evidence_role"] != "DEVELOPMENT":
            raise RepeatabilityGateError("repeatability prompts must be DEVELOPMENT only")
        prompt = raw["prompt"]
        if not isinstance(prompt, Mapping):
            raise RepeatabilityGateError("repeatability prompt must be an object")
        if prompt.get("prompt_protocol_version") != PROMPT_PROTOCOL_VERSION:
            raise RepeatabilityGateError("repeatability prompt protocol mismatch")
        for hash_name in (
            "prompt_bundle_sha256",
            "system_message_sha256",
            "user_template_sha256",
            "user_message_sha256",
            "rendered_prompt_sha256",
        ):
            if not isinstance(prompt.get(hash_name), str) or _SHA_RE.fullmatch(prompt[hash_name]) is None:
                raise RepeatabilityGateError(f"repeatability prompt missing {hash_name}")
        values.append(
            {
                "position": index,
                "prompt_id": prompt_id,
                "dataset": raw["dataset"],
                "evidence_role": raw["evidence_role"],
                "sample_id": canonical_stable_id(raw["sample_id"], "sample_id"),
                "prompt": dict(prompt),
            }
        )
    return {
        "schema_version": REPEATABILITY_PROMPT_MANIFEST_SCHEMA,
        "selection_status": (
            "PROSPECTIVELY_FROZEN_EXPLICIT_MANIFEST; protocol does not define "
            "an ID-selection namespace"
        ),
        "prompt_count": 20,
        "entries": values,
    }


def repeatability_prompt_manifest_wrapper(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = repeatability_prompt_manifest_payload(entries)
    digest = stable_json_sha256(payload)
    return {
        "artifact_format": REPEATABILITY_PROMPT_MANIFEST_FORMAT,
        "manifest_id": f"generation-repeatability-prompts:sha256:{digest}",
        "scientific_sha256": digest,
        "scientific_payload": payload,
    }


def write_repeatability_prompt_manifest(
    path: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    write_immutable_json(path, repeatability_prompt_manifest_wrapper(entries))


def read_repeatability_prompt_manifest(path: Path) -> dict[str, Any]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping) or set(stored) != {
        "artifact_format",
        "manifest_id",
        "scientific_sha256",
        "scientific_payload",
    }:
        raise RepeatabilityGateError("repeatability prompt manifest wrapper is invalid")
    entries = stored["scientific_payload"].get("entries", [])
    rebuilt = repeatability_prompt_manifest_wrapper(
        [
            {
                "prompt_id": item["prompt_id"],
                "dataset": item["dataset"],
                "evidence_role": item["evidence_role"],
                "sample_id": item["sample_id"],
                "prompt": item["prompt"],
            }
            for item in entries
        ]
    )
    if dict(stored) != rebuilt:
        raise RepeatabilityGateError("repeatability prompt manifest identity mismatch")
    return rebuilt


def _rendered_from_provenance(value: Mapping[str, Any]) -> RenderedPrompt:
    prompt = RenderedPrompt(
        prompt_protocol_version=value["prompt_protocol_version"],
        context_mode=value["context_mode"],
        system_message=value["system_message"],
        user_template=value["user_template"],
        user_message=value["user_message"],
        context_block=value["context_block"],
    )
    if prompt.provenance_payload() != dict(value):
        raise RepeatabilityGateError("stored repeatability prompt is not canonical")
    return prompt


def run_repeatability_gate(
    *,
    prompt_manifest_path: Path,
    adapters: Mapping[str, CanonicalMakiAdapter],
    output_path: Path,
    created_at: str,
) -> dict[str, Any]:
    """Execute 180 explicitly requested calls and freeze the aggregate result."""
    if tuple(sorted(adapters)) != tuple(sorted(PRIMARY_LLM_LOGICAL_IDS)):
        raise RepeatabilityGateError("repeatability gate requires all three primary LLMs")
    manifest = read_repeatability_prompt_manifest(prompt_manifest_path)
    for adapter in adapters.values():
        adapter.require_api_key()
    models: list[dict[str, Any]] = []
    for logical_id in PRIMARY_LLM_LOGICAL_IDS:
        adapter = adapters[logical_id]
        if adapter.config.logical_model_id != logical_id:
            raise RepeatabilityGateError("adapter logical model binding mismatch")
        prompt_results: list[dict[str, Any]] = []
        identical_count = 0
        for entry in manifest["scientific_payload"]["entries"]:
            prompt = _rendered_from_provenance(entry["prompt"])
            calls: list[dict[str, Any]] = []
            for repetition in range(1, 4):
                completion = adapter.complete(prompt)
                calls.append(
                    {
                        "repetition": repetition,
                        "raw_content": completion.raw_content,
                        "finish_reason": completion.finish_reason,
                        "provider_refusal": completion.provider_refusal,
                        "provider_metadata": dict(completion.provider_metadata),
                        "attempts": [dict(item) for item in completion.attempts],
                        "transport_exhausted": completion.transport_exhausted,
                    }
                )
            stripped = [
                None if item["raw_content"] is None else item["raw_content"].strip()
                for item in calls
            ]
            identical = (
                all(value is not None for value in stripped)
                and len(set(stripped)) == 1
            )
            identical_count += int(identical)
            prompt_results.append(
                {
                    "prompt_id": entry["prompt_id"],
                    "calls": calls,
                    "all_three_stripped_identical": identical,
                }
            )
        models.append(
            {
                "logical_model_id": logical_id,
                "runtime_identity": adapter.config.runtime_identity(),
                "identical_prompt_count": identical_count,
                "prompt_count": 20,
                "passed": identical_count >= 19,
                "prompt_results": prompt_results,
            }
        )
    scientific = {
        "schema_version": REPEATABILITY_GATE_SCHEMA,
        "prompt_manifest_id": manifest["manifest_id"],
        "prompt_manifest_sha256": manifest["scientific_sha256"],
        "calls_per_prompt": 3,
        "pass_threshold": 19,
        "models": models,
        "all_primary_models_passed": all(item["passed"] for item in models),
    }
    digest = stable_json_sha256(scientific)
    wrapper = {
        "artifact_format": REPEATABILITY_GATE_FORMAT,
        "gate_id": f"generation-repeatability-gate:sha256:{digest}",
        "scientific_sha256": digest,
        "scientific_payload": scientific,
        "created_at": created_at,
    }
    write_immutable_json(output_path, wrapper)
    return wrapper


def read_repeatability_gate(path: Path) -> dict[str, Any]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping) or set(stored) != {
        "artifact_format",
        "gate_id",
        "scientific_sha256",
        "scientific_payload",
        "created_at",
    }:
        raise RepeatabilityGateError("repeatability gate wrapper is invalid")
    if stored["artifact_format"] != REPEATABILITY_GATE_FORMAT:
        raise RepeatabilityGateError("repeatability gate format mismatch")
    scientific = stored["scientific_payload"]
    if not isinstance(scientific, Mapping) or set(scientific) != {
        "schema_version",
        "prompt_manifest_id",
        "prompt_manifest_sha256",
        "calls_per_prompt",
        "pass_threshold",
        "models",
        "all_primary_models_passed",
    }:
        raise RepeatabilityGateError("repeatability gate scientific payload is invalid")
    if (
        scientific["schema_version"] != REPEATABILITY_GATE_SCHEMA
        or scientific["calls_per_prompt"] != 3
        or scientific["pass_threshold"] != 19
        or not isinstance(scientific["models"], list)
        or len(scientific["models"]) != 3
    ):
        raise RepeatabilityGateError("repeatability gate frozen dimensions mismatch")
    seen_models: set[str] = set()
    model_passes: list[bool] = []
    for model in scientific["models"]:
        if not isinstance(model, Mapping) or set(model) != {
            "logical_model_id",
            "runtime_identity",
            "identical_prompt_count",
            "prompt_count",
            "passed",
            "prompt_results",
        }:
            raise RepeatabilityGateError("repeatability model result is invalid")
        logical_id = model["logical_model_id"]
        if logical_id not in PRIMARY_LLM_LOGICAL_IDS or logical_id in seen_models:
            raise RepeatabilityGateError("repeatability model identities are invalid")
        seen_models.add(logical_id)
        if model["prompt_count"] != 20 or not isinstance(model["prompt_results"], list) or len(model["prompt_results"]) != 20:
            raise RepeatabilityGateError("repeatability model prompt count mismatch")
        identical_count = 0
        seen_prompts: set[str] = set()
        for prompt_result in model["prompt_results"]:
            if not isinstance(prompt_result, Mapping) or set(prompt_result) != {
                "prompt_id",
                "calls",
                "all_three_stripped_identical",
            }:
                raise RepeatabilityGateError("repeatability prompt result is invalid")
            prompt_id = prompt_result["prompt_id"]
            if not isinstance(prompt_id, str) or prompt_id in seen_prompts:
                raise RepeatabilityGateError("repeatability prompt IDs are invalid")
            seen_prompts.add(prompt_id)
            calls = prompt_result["calls"]
            if not isinstance(calls, list) or len(calls) != 3:
                raise RepeatabilityGateError("repeatability prompt requires three calls")
            raw_values = []
            for repetition, call in enumerate(calls, start=1):
                if not isinstance(call, Mapping) or set(call) != {
                    "repetition",
                    "raw_content",
                    "finish_reason",
                    "provider_refusal",
                    "provider_metadata",
                    "attempts",
                    "transport_exhausted",
                } or call["repetition"] != repetition:
                    raise RepeatabilityGateError("repeatability call record is invalid")
                raw = call["raw_content"]
                if raw is not None and not isinstance(raw, str):
                    raise RepeatabilityGateError("repeatability raw content is invalid")
                raw_values.append(None if raw is None else raw.strip())
            identical = all(value is not None for value in raw_values) and len(set(raw_values)) == 1
            if prompt_result["all_three_stripped_identical"] is not identical:
                raise RepeatabilityGateError("repeatability identity decision mismatch")
            identical_count += int(identical)
        if model["identical_prompt_count"] != identical_count:
            raise RepeatabilityGateError("repeatability identical count mismatch")
        passed = identical_count >= 19
        if model["passed"] is not passed:
            raise RepeatabilityGateError("repeatability model pass decision mismatch")
        model_passes.append(passed)
    if set(seen_models) != set(PRIMARY_LLM_LOGICAL_IDS):
        raise RepeatabilityGateError("repeatability gate is missing a primary model")
    if scientific["all_primary_models_passed"] is not all(model_passes):
        raise RepeatabilityGateError("repeatability aggregate pass decision mismatch")
    digest = stable_json_sha256(stored["scientific_payload"])
    if stored["scientific_sha256"] != digest or stored["gate_id"] != (
        f"generation-repeatability-gate:sha256:{digest}"
    ):
        raise RepeatabilityGateError("repeatability gate identity mismatch")
    return dict(stored)


def require_passing_repeatability_gate(
    path: Path,
    *,
    model_runtime_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gate = read_repeatability_gate(path)
    scientific = gate["scientific_payload"]
    if scientific.get("all_primary_models_passed") is not True:
        raise RepeatabilityGateError("generation matrix is blocked: gate did not pass")
    by_model = {item["logical_model_id"]: item for item in scientific["models"]}
    if set(by_model) != set(PRIMARY_LLM_LOGICAL_IDS):
        raise RepeatabilityGateError("generation matrix is blocked: gate model set mismatch")
    if set(model_runtime_identities) != set(PRIMARY_LLM_LOGICAL_IDS):
        raise RepeatabilityGateError("all three current runtime identities are required")
    for logical_id, identity in model_runtime_identities.items():
        model = by_model[logical_id]
        if model["passed"] is not True or model["runtime_identity"] != dict(identity):
            raise RepeatabilityGateError(
                f"generation matrix is blocked: {logical_id} runtime identity differs from gate"
            )
    return gate
