"""Frozen ASQA passage-aspect alias matcher.

Implements docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable


_APOSTROPHE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\uff07": "'",
}

_SEPARATOR_CATEGORIES = {
    "Pd", "Po", "Ps", "Pe", "Pi", "Pf", "Pc",
    "Sm", "Sc", "Sk", "So",
    "Zs", "Zl", "Zp",
}


def _map_apostrophes(text: str) -> str:
    return "".join(_APOSTROPHE_MAP.get(ch, ch) for ch in text)


def normalize_tokens(text: str) -> tuple[str, ...]:
    """Apply the frozen conservative ASQA base normalization."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFC", text)
    text = _map_apostrophes(text)

    chars: list[str] = []

    for i, ch in enumerate(text):
        if ch == "'":
            prev_is_alnum = i > 0 and text[i - 1].isalnum()
            next_is_alnum = i + 1 < len(text) and text[i + 1].isalnum()

            if prev_is_alnum and next_is_alnum:
                chars.append(ch)
            else:
                chars.append(" ")
            continue

        if unicodedata.category(ch) in _SEPARATOR_CATEGORIES:
            chars.append(" ")
        else:
            chars.append(ch)

    normalized = " ".join("".join(chars).split())

    if not normalized:
        return ()

    return tuple(normalized.split(" "))


def normalize_alias(alias: str) -> tuple[str, ...]:
    """Normalize one official alias using the frozen case policy."""
    tokens = normalize_tokens(alias)

    if len(tokens) <= 1:
        return tokens

    return tuple(token.casefold() for token in tokens)


def passage_tokens_for_alias(
    passage_body: str,
    *,
    alias_token_count: int,
) -> tuple[str, ...]:
    """Return passage tokens under the alias-specific frozen case policy."""
    tokens = normalize_tokens(passage_body)

    if alias_token_count <= 1:
        return tokens

    return tuple(token.casefold() for token in tokens)


def contiguous_token_match(
    passage_body: str,
    alias: str,
) -> bool:
    """Return True iff alias matches an exact contiguous token sequence."""
    alias_tokens = normalize_alias(alias)

    if not alias_tokens:
        return False

    passage_tokens = passage_tokens_for_alias(
        passage_body,
        alias_token_count=len(alias_tokens),
    )

    width = len(alias_tokens)

    for start in range(0, len(passage_tokens) - width + 1):
        if passage_tokens[start : start + width] == alias_tokens:
            return True

    return False


def aspect_matches(
    passage_body: str,
    aliases: Iterable[str],
) -> bool:
    """Return J_q(d,i)=1 iff any surviving official alias matches."""
    seen: set[tuple[str, ...]] = set()

    for raw_alias in aliases:
        if not isinstance(raw_alias, str):
            raise TypeError("aliases must contain only strings")

        if not raw_alias:
            continue

        normalized = normalize_alias(raw_alias)

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        if contiguous_token_match(passage_body, raw_alias):
            return True

    return False
