"""Frozen PubMedQA response parsing and generation-status classification."""

from __future__ import annotations

import re
from typing import Any

from generation.artifacts import GenerationStatus


_DECISION_FIELD = re.compile(r"^Decision:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
_STANDALONE_REFUSAL = re.compile(
    r"^\s*(?:i\s+(?:cannot|can't|am\s+unable\s+to|won't)\s+"
    r"(?:answer|provide\s+an\s+answer|help)|"
    r"i\s+(?:must|have\s+to)\s+(?:decline|refuse)|"
    r"sorry,?\s+i\s+(?:cannot|can't|am\s+unable\s+to)\s+"
    r"(?:answer|help)).*?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TRUNCATION_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "token_limit"}
)


def parse_pubmedqa_decision(raw_content: str) -> str | None:
    """Apply the exact Decision-field rules from ANSWER_CORRECTNESS_PROTOCOL."""
    if not isinstance(raw_content, str):
        raise TypeError("raw_content must be a string")
    normalized = raw_content.replace("\r\n", "\n").replace("\r", "\n")
    matches = _DECISION_FIELD.findall(normalized)
    if len(matches) != 1:
        return None
    value = matches[0].strip().lower()
    return value if value in {"yes", "no", "maybe"} else None


def classify_pubmedqa_response(
    *,
    raw_content: str | None,
    finish_reason: str | None,
    provider_refusal: bool = False,
    transport_exhausted: bool = False,
) -> tuple[GenerationStatus, dict[str, Any] | None]:
    """Classify one response without retrying or semantically repairing content."""
    if transport_exhausted:
        if raw_content is not None:
            raise ValueError("transport-exhausted ERROR cannot carry response content")
        return GenerationStatus.ERROR, None
    if not isinstance(raw_content, str):
        raise ValueError("successful provider response must carry raw string content")
    normalized_finish = "" if finish_reason is None else finish_reason.strip().lower()
    if normalized_finish in _TRUNCATION_REASONS:
        return GenerationStatus.TRUNCATED, None
    if provider_refusal or _STANDALONE_REFUSAL.fullmatch(raw_content) is not None:
        return GenerationStatus.REFUSAL, None
    decision = parse_pubmedqa_decision(raw_content)
    if decision is None:
        return GenerationStatus.PARSE_FAILURE, None
    return GenerationStatus.OK, {"decision": decision}
