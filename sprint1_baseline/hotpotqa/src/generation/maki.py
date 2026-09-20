"""Canonical injectable Mannheim Maki adapter for frozen generation requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
import time
from typing import Any, Callable, Mapping, Protocol

from generation._io import stable_json_sha256
from generation.prompts import RenderedPrompt


MAKI_ADAPTER_VERSION = "sprint3.maki-openai-compatible.v1"
PRIMARY_LLM_LOGICAL_IDS = (
    "llama-3.3-70b",
    "gemma4-26b",
    "ministral-3-14b",
)
_FROZEN_REQUEST_KEYS = frozenset(
    {"model", "messages", "temperature", "max_tokens", "n", "stream", "seed"}
)
_BEARER_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/\-]+=*")
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_error_message(value: object) -> str:
    text = str(value)[:1000]
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text


class MakiInfrastructureError(RuntimeError):
    """A network/provider/runtime failure eligible for infrastructure retry."""


class MakiTransport(Protocol):
    def __call__(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class RequestsMakiTransport:
    """Thin network transport; never instantiated or called by import."""

    def __call__(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        import requests

        try:
            response = requests.post(
                url,
                headers=dict(headers),
                json=dict(payload),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            value = response.json()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            suffix = "" if status is None else f" (HTTP {status})"
            raise MakiInfrastructureError(
                sanitize_error_message(f"Maki transport failed{suffix}: {exc}")
            ) from exc
        except (TypeError, ValueError) as exc:
            raise MakiInfrastructureError(
                sanitize_error_message(f"Maki returned invalid JSON: {exc}")
            ) from exc
        if not isinstance(value, Mapping):
            raise MakiInfrastructureError("Maki response JSON is not an object")
        return value


@dataclass(frozen=True)
class MakiConfig:
    base_url: str
    logical_model_id: str
    physical_model_id: str
    model_revision: str | None
    model_revision_kind: str
    direct_mode_status: str
    direct_mode_control: Mapping[str, Any]
    seed: int | None = None
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be explicit and non-empty")
        if self.logical_model_id not in PRIMARY_LLM_LOGICAL_IDS:
            raise ValueError("logical_model_id is not a primary frozen generator")
        if not isinstance(self.physical_model_id, str) or not self.physical_model_id.strip():
            raise ValueError("physical_model_id is mandatory; no model default exists")
        if self.model_revision_kind == "NOT_PROVIDED_BY_PROVIDER":
            if self.model_revision is not None:
                raise ValueError("NOT_PROVIDED_BY_PROVIDER requires null revision")
        elif self.model_revision_kind not in {
            "IMMUTABLE_REVISION",
            "PROVIDER_SNAPSHOT",
        }:
            raise ValueError("model_revision_kind is invalid")
        elif not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model revision is required for the selected revision kind")
        if self.direct_mode_status not in {
            "SUPPORTED_AND_ENABLED",
            "NOT_SUPPORTED_BY_PROVIDER",
        }:
            raise ValueError("direct_mode_status is invalid")
        if not isinstance(self.direct_mode_control, Mapping):
            raise TypeError("direct_mode_control must be an object")
        overlap = _FROZEN_REQUEST_KEYS.intersection(self.direct_mode_control)
        if overlap:
            raise ValueError(f"direct-mode controls cannot override frozen keys: {sorted(overlap)}")
        if self.direct_mode_status == "NOT_SUPPORTED_BY_PROVIDER" and self.direct_mode_control:
            raise ValueError("unsupported direct mode must use an empty control object")
        stable_json_sha256(dict(self.direct_mode_control))
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or null")
        if isinstance(self.timeout_seconds, bool) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def runtime_identity(self) -> dict[str, Any]:
        return {
            "adapter_version": MAKI_ADAPTER_VERSION,
            "base_url": self.base_url.rstrip("/"),
            "logical_model_id": self.logical_model_id,
            "physical_model_id": self.physical_model_id,
            "model_revision": self.model_revision,
            "model_revision_kind": self.model_revision_kind,
            "direct_mode_status": self.direct_mode_status,
            "direct_mode_control": dict(self.direct_mode_control),
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class MakiCompletion:
    raw_content: str | None
    finish_reason: str | None
    provider_refusal: bool
    provider_metadata: Mapping[str, Any]
    attempts: tuple[Mapping[str, Any], ...]
    transport_exhausted: bool


def _safe_provider_metadata(response: Mapping[str, Any], choice: Mapping[str, Any]) -> dict[str, Any]:
    message = choice.get("message")
    message_mapping = message if isinstance(message, Mapping) else {}
    usage = response.get("usage")
    safe_usage = dict(usage) if isinstance(usage, Mapping) else None
    return {
        "response_id": response.get("id"),
        "object": response.get("object"),
        "created": response.get("created"),
        "system_fingerprint": response.get("system_fingerprint"),
        "service_tier": response.get("service_tier"),
        "usage": safe_usage,
        "choice_index": choice.get("index"),
        "message_role": message_mapping.get("role"),
        "provider_refusal": message_mapping.get("refusal"),
    }


def _parse_success(response: Mapping[str, Any]) -> tuple[str, str | None, bool, dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise MakiInfrastructureError("Maki response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise MakiInfrastructureError("Maki choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise MakiInfrastructureError("Maki choice has no message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise MakiInfrastructureError("Maki message content is not a string")
    finish = choice.get("finish_reason")
    if finish is not None and not isinstance(finish, str):
        raise MakiInfrastructureError("Maki finish_reason is not a string or null")
    refusal_value = message.get("refusal")
    provider_refusal = bool(refusal_value) if refusal_value is not None else False
    return content, finish, provider_refusal, _safe_provider_metadata(response, choice)


class CanonicalMakiAdapter:
    def __init__(
        self,
        config: MakiConfig,
        *,
        transport: MakiTransport | None = None,
        api_key_env: str = "MAKI_API_KEY",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], str] = utc_now,
        retry_delay_seconds: float = 1.0,
        max_tokens: int = 256,
    ) -> None:
        if max_tokens not in {256, 512}:
            raise ValueError("max_tokens must be one of the frozen dataset values: 256 or 512")
        self.config = config
        self.max_tokens = max_tokens
        self.transport = RequestsMakiTransport() if transport is None else transport
        self.api_key_env = api_key_env
        self.sleep = sleep
        self.clock = clock
        self.retry_delay_seconds = retry_delay_seconds

    def _api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "")
        if not value.strip():
            raise MakiInfrastructureError(
                f"required environment variable {self.api_key_env} is not set"
            )
        if _BEARER_TOKEN_RE.fullmatch(value) is None:
            raise MakiInfrastructureError(
                f"required environment variable {self.api_key_env} cannot safely "
                "form a Bearer authorization header"
            )
        return value

    def require_api_key(self) -> None:
        """Fail preflight without returning or serializing the secret value."""
        self._api_key()

    def request_payload(self, prompt: RenderedPrompt) -> dict[str, Any]:
        payload = {
            "model": self.config.physical_model_id,
            "messages": [dict(message) for message in prompt.messages],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stream": False,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        payload.update(dict(self.config.direct_mode_control))
        return payload

    def complete(self, prompt: RenderedPrompt) -> MakiCompletion:
        """Make at most three infrastructure attempts; never retry content."""
        attempts: list[dict[str, Any]] = []
        try:
            api_key = self._api_key()
        except MakiInfrastructureError as exc:
            # Missing credentials is an execution blocker, not three fake network calls.
            raise exc
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self.request_payload(prompt)
        for attempt_number in range(1, 4):
            started_at = self.clock()
            try:
                response = self.transport(
                    url=self.config.chat_url,
                    headers=headers,
                    payload=payload,
                    timeout_seconds=self.config.timeout_seconds,
                )
                content, finish, refusal, metadata = _parse_success(response)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "started_at": started_at,
                        "completed_at": self.clock(),
                        "outcome": "SUCCESS",
                        "error_type": None,
                        "error_message": None,
                    }
                )
                return MakiCompletion(
                    raw_content=content,
                    finish_reason=finish,
                    provider_refusal=refusal,
                    provider_metadata=metadata,
                    attempts=tuple(attempts),
                    transport_exhausted=False,
                )
            except Exception as exc:
                if not isinstance(exc, MakiInfrastructureError):
                    exc = MakiInfrastructureError(sanitize_error_message(exc))
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "started_at": started_at,
                        "completed_at": self.clock(),
                        "outcome": "INFRASTRUCTURE_ERROR",
                        "error_type": type(exc).__name__,
                        "error_message": sanitize_error_message(exc),
                    }
                )
                if attempt_number < 3:
                    self.sleep(self.retry_delay_seconds)
        return MakiCompletion(
            raw_content=None,
            finish_reason=None,
            provider_refusal=False,
            provider_metadata={},
            attempts=tuple(attempts),
            transport_exhausted=True,
        )
