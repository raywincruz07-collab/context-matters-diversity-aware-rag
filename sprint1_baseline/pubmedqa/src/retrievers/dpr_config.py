"""Dependency-neutral authoritative DPR scientific configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class DPRConfig:
    """Structured scientific definition of the repository's DPR method."""

    question_model_id: str
    question_model_revision: str
    question_tokenizer_id: str
    question_tokenizer_revision: str
    context_model_id: str
    context_model_revision: str
    context_tokenizer_id: str
    context_tokenizer_revision: str
    query_preprocessing: str
    document_preprocessing: str
    query_max_length: int
    context_max_length: int
    truncation_enabled: bool
    paired_truncation_strategy: str
    padding_semantics: str
    context_title_policy: str
    representation: str
    embedding_dimension: int
    embedding_dtype: str
    normalization: str
    score_semantics: str
    ranking_direction: str
    index_type: str
    score_sign_filtering: str
    context_batch_size: int
    query_batch_size: int

    def __post_init__(self) -> None:
        for name in (
            "question_model_id",
            "question_tokenizer_id",
            "context_model_id",
            "context_tokenizer_id",
            "query_preprocessing",
            "document_preprocessing",
            "paired_truncation_strategy",
            "padding_semantics",
            "context_title_policy",
            "representation",
            "embedding_dtype",
            "normalization",
            "score_semantics",
            "ranking_direction",
            "index_type",
            "score_sign_filtering",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        for name in (
            "question_model_revision",
            "question_tokenizer_revision",
            "context_model_revision",
            "context_tokenizer_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _FULL_GIT_SHA.fullmatch(value) is None:
                raise ValueError(
                    f"{name} must be a full lowercase 40-character hexadecimal SHA"
                )

        for name in (
            "query_max_length",
            "context_max_length",
            "embedding_dimension",
            "context_batch_size",
            "query_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive non-boolean integer")

        if not isinstance(self.truncation_enabled, bool):
            raise TypeError("truncation_enabled must be a bool")

    def scientific_payload(self) -> dict:
        """Return every scientifically relevant DPR method choice."""
        return {
            "context_batch_size": self.context_batch_size,
            "context_max_length": self.context_max_length,
            "context_model_id": self.context_model_id,
            "context_model_revision": self.context_model_revision,
            "context_title_policy": self.context_title_policy,
            "context_tokenizer_id": self.context_tokenizer_id,
            "context_tokenizer_revision": self.context_tokenizer_revision,
            "document_preprocessing": self.document_preprocessing,
            "embedding_dimension": self.embedding_dimension,
            "embedding_dtype": self.embedding_dtype,
            "index_type": self.index_type,
            "normalization": self.normalization,
            "padding_semantics": self.padding_semantics,
            "paired_truncation_strategy": self.paired_truncation_strategy,
            "query_batch_size": self.query_batch_size,
            "query_max_length": self.query_max_length,
            "query_preprocessing": self.query_preprocessing,
            "question_model_id": self.question_model_id,
            "question_model_revision": self.question_model_revision,
            "question_tokenizer_id": self.question_tokenizer_id,
            "question_tokenizer_revision": self.question_tokenizer_revision,
            "ranking_direction": self.ranking_direction,
            "representation": self.representation,
            "score_semantics": self.score_semantics,
            "score_sign_filtering": self.score_sign_filtering,
            "truncation_enabled": self.truncation_enabled,
        }

    def scientific_json(self) -> str:
        """Serialize the DPR scientific definition as canonical JSON."""
        return json.dumps(
            self.scientific_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


DPR_CONFIG = DPRConfig(
    question_model_id="facebook/dpr-question_encoder-single-nq-base",
    question_model_revision="d04a52f6d2f96c60117a925e8c24c4043a75f265",
    question_tokenizer_id="facebook/dpr-question_encoder-single-nq-base",
    question_tokenizer_revision="d04a52f6d2f96c60117a925e8c24c4043a75f265",
    context_model_id="facebook/dpr-ctx_encoder-single-nq-base",
    context_model_revision="bb21a3c2b1656d60c6a8e920283bc40dabddadb8",
    context_tokenizer_id="facebook/dpr-ctx_encoder-single-nq-base",
    context_tokenizer_revision="bb21a3c2b1656d60c6a8e920283bc40dabddadb8",
    query_preprocessing="exact query string passed to tokenizer",
    document_preprocessing="exact passage text passed to tokenizer paired with empty title",
    query_max_length=256,
    context_max_length=256,
    truncation_enabled=True,
    paired_truncation_strategy="delegated to tokenizer/Transformers default",
    padding_semantics="dynamic padding to longest encoded input in each batch",
    context_title_policy="empty title paired with passage text",
    representation="pooler_output",
    embedding_dimension=768,
    embedding_dtype="float32",
    normalization="none",
    score_semantics="raw dot product / inner product of unnormalized embeddings",
    ranking_direction="higher score is better",
    index_type="faiss.IndexFlatIP",
    score_sign_filtering="none",
    context_batch_size=16,
    query_batch_size=1,
)
