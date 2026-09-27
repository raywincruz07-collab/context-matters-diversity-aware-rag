"""Exact frozen Sprint-3 HotpotQA generation prompt assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from generation._io import sha256_text, stable_json_sha256
from generation.prompts import SYSTEM_INSTRUCTION, render_context_block


HOTPOTQA_PROMPT_PROTOCOL_VERSION = "sprint3.generation.hotpotqa.v1"

HOTPOTQA_WITH_CONTEXT_TEMPLATE = """Question:
{question}

Context:
{context_block}

Output format:
Answer: <short answer>
Explanation: <1-3 concise factual sentences>"""

HOTPOTQA_WITHOUT_CONTEXT_TEMPLATE = """Question:
{question}

Output format:
Answer: <short answer>
Explanation: <1-3 concise factual sentences>"""


HOTPOTQA_PROMPT_BUNDLE_SHA256 = stable_json_sha256(
    {
        "prompt_protocol_version": HOTPOTQA_PROMPT_PROTOCOL_VERSION,
        "assets": {
            "system_instruction": SYSTEM_INSTRUCTION,
            "hotpotqa_with_context": HOTPOTQA_WITH_CONTEXT_TEMPLATE,
            "hotpotqa_without_context": HOTPOTQA_WITHOUT_CONTEXT_TEMPLATE,
        },
    }
)


@dataclass(frozen=True)
class HotpotQARenderedPrompt:
    prompt_protocol_version: str
    context_mode: str
    system_message: str
    user_template: str
    user_message: str
    context_block: str | None

    def __post_init__(self) -> None:
        if self.prompt_protocol_version != HOTPOTQA_PROMPT_PROTOCOL_VERSION:
            raise ValueError("unexpected HotpotQA prompt protocol version")

        if self.context_mode not in {
            "with_context",
            "without_context",
        }:
            raise ValueError("context_mode is not canonical")

        if self.system_message != SYSTEM_INSTRUCTION:
            raise ValueError("system message differs from frozen asset")

        expected_template = (
            HOTPOTQA_WITH_CONTEXT_TEMPLATE
            if self.context_mode == "with_context"
            else HOTPOTQA_WITHOUT_CONTEXT_TEMPLATE
        )

        if self.user_template != expected_template:
            raise ValueError("user template differs from frozen HotpotQA asset")

        if (
            self.context_mode == "with_context"
            and self.context_block is None
        ):
            raise ValueError("WITH_CONTEXT requires context")

        if (
            self.context_mode == "without_context"
            and self.context_block is not None
        ):
            raise ValueError("WITHOUT_CONTEXT forbids context")

    @property
    def system_message_sha256(self) -> str:
        return sha256_text(self.system_message)

    @property
    def user_template_sha256(self) -> str:
        return sha256_text(self.user_template)

    @property
    def user_message_sha256(self) -> str:
        return sha256_text(self.user_message)

    @property
    def context_block_sha256(self) -> str | None:
        if self.context_block is None:
            return None
        return sha256_text(self.context_block)

    @property
    def messages(self) -> tuple[dict[str, str], dict[str, str]]:
        return (
            {
                "role": "system",
                "content": self.system_message,
            },
            {
                "role": "user",
                "content": self.user_message,
            },
        )

    @property
    def rendered_prompt_sha256(self) -> str:
        return stable_json_sha256(list(self.messages))

    def provenance_payload(self) -> dict[str, Any]:
        return {
            "prompt_protocol_version": self.prompt_protocol_version,
            "prompt_bundle_sha256": HOTPOTQA_PROMPT_BUNDLE_SHA256,
            "context_mode": self.context_mode,
            "system_message": self.system_message,
            "system_message_sha256": self.system_message_sha256,
            "user_template": self.user_template,
            "user_template_sha256": self.user_template_sha256,
            "user_message": self.user_message,
            "user_message_sha256": self.user_message_sha256,
            "context_block": self.context_block,
            "context_block_sha256": self.context_block_sha256,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
        }


def _substitute(
    template: str,
    *,
    question: str,
    context_block: str | None,
) -> str:
    if not isinstance(question, str) or not question:
        raise ValueError(
            "question must be exact non-empty canonical text"
        )

    if template.count("{question}") != 1:
        raise RuntimeError(
            "frozen HotpotQA template has invalid question placeholder"
        )

    rendered = template.replace("{question}", question)

    if context_block is None:
        if "{context_block}" in rendered:
            raise RuntimeError(
                "context placeholder remained in WITHOUT_CONTEXT prompt"
            )
    else:
        if rendered.count("{context_block}") != 1:
            raise RuntimeError(
                "frozen HotpotQA template has invalid context placeholder"
            )
        rendered = rendered.replace(
            "{context_block}",
            context_block,
        )

    return rendered


def render_hotpotqa_prompt(
    *,
    question: str,
    passage_bodies: tuple[str, ...] | None = None,
) -> HotpotQARenderedPrompt:

    if passage_bodies is None:
        mode = "without_context"
        template = HOTPOTQA_WITHOUT_CONTEXT_TEMPLATE
        context_block = None
    else:
        mode = "with_context"
        template = HOTPOTQA_WITH_CONTEXT_TEMPLATE

        # Critical HotpotQA rule:
        # passage_bodies are BEIR `text` bodies only.
        # Titles must NOT be injected into generation context.
        context_block = render_context_block(passage_bodies)

    return HotpotQARenderedPrompt(
        prompt_protocol_version=HOTPOTQA_PROMPT_PROTOCOL_VERSION,
        context_mode=mode,
        system_message=SYSTEM_INSTRUCTION,
        user_template=template,
        user_message=_substitute(
            template,
            question=question,
            context_block=context_block,
        ),
        context_block=context_block,
    )
