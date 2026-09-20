"""Dependency-neutral authoritative Contriever scientific configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ContrieverConfig:
    """Structured scientific definition of the canonical Contriever method."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    model_loader: str
    tokenizer_loader: str
    query_preprocessing: str
    document_preprocessing: str
    pooling: str
    embedding_dimension: int
    query_max_length: int
    document_max_length: int
    truncation_enabled: bool
    truncation_semantics: str
    padding_semantics: str
    normalization: str
    compute_dtype: str
    autocast: bool
    embedding_dtype: str
    score_semantics: str
    ranking_direction: str
    index_type: str
    index_device: str
    score_sign_filtering: str
    query_batch_size: int
    document_batch_size: int

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "model_loader",
            "tokenizer_loader",
            "query_preprocessing",
            "document_preprocessing",
            "pooling",
            "truncation_semantics",
            "padding_semantics",
            "normalization",
            "compute_dtype",
            "embedding_dtype",
            "score_semantics",
            "ranking_direction",
            "index_type",
            "index_device",
            "score_sign_filtering",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        for name in ("model_revision", "tokenizer_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or _FULL_GIT_SHA.fullmatch(value) is None:
                raise ValueError(
                    f"{name} must be a full lowercase 40-character hexadecimal SHA"
                )

        for name in (
            "embedding_dimension",
            "query_max_length",
            "document_max_length",
            "query_batch_size",
            "document_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive non-boolean integer")

        if not isinstance(self.truncation_enabled, bool):
            raise TypeError("truncation_enabled must be a bool")
        if not isinstance(self.autocast, bool):
            raise TypeError("autocast must be a bool")

    def scientific_payload(self) -> dict:
        """Return every scientifically relevant Contriever method choice."""
        return {
            "autocast": self.autocast,
            "compute_dtype": self.compute_dtype,
            "document_batch_size": self.document_batch_size,
            "document_max_length": self.document_max_length,
            "document_preprocessing": self.document_preprocessing,
            "embedding_dimension": self.embedding_dimension,
            "embedding_dtype": self.embedding_dtype,
            "index_device": self.index_device,
            "index_type": self.index_type,
            "model_id": self.model_id,
            "model_loader": self.model_loader,
            "model_revision": self.model_revision,
            "normalization": self.normalization,
            "padding_semantics": self.padding_semantics,
            "pooling": self.pooling,
            "query_batch_size": self.query_batch_size,
            "query_max_length": self.query_max_length,
            "query_preprocessing": self.query_preprocessing,
            "ranking_direction": self.ranking_direction,
            "score_semantics": self.score_semantics,
            "score_sign_filtering": self.score_sign_filtering,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_loader": self.tokenizer_loader,
            "tokenizer_revision": self.tokenizer_revision,
            "truncation_enabled": self.truncation_enabled,
            "truncation_semantics": self.truncation_semantics,
        }

    def scientific_json(self) -> str:
        """Serialize the scientific definition as deterministic canonical JSON."""
        return json.dumps(
            self.scientific_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


CONTRIEVER_CONFIG = ContrieverConfig(
    model_id="facebook/contriever",
    model_revision="2bd46a25019aeea091fd42d1f0fd4801675cf699",
    tokenizer_id="facebook/contriever",
    tokenizer_revision="2bd46a25019aeea091fd42d1f0fd4801675cf699",
    model_loader="AutoModel",
    tokenizer_loader="AutoTokenizer",
    query_preprocessing=(
        "exact query string passed to tokenizer; no manual lowercasing or external "
        "text normalization; tokenizer-native behavior preserved"
    ),
    document_preprocessing=(
        "exact validated CorpusRecord.retrieval_content passed to tokenizer; no "
        "manual lowercasing or external text normalization; tokenizer-native "
        "behavior preserved"
    ),
    pooling="attention-mask-aware mean pooling of last_hidden_state",
    embedding_dimension=768,
    query_max_length=512,
    document_max_length=512,
    truncation_enabled=True,
    truncation_semantics=(
        "single-sequence truncation delegated to tokenizer/Transformers"
    ),
    padding_semantics="dynamic padding to longest encoded input in each batch",
    normalization="none",
    compute_dtype="float32",
    autocast=False,
    embedding_dtype="float32",
    score_semantics="raw dot product / inner product of unnormalized embeddings",
    ranking_direction="higher score is better",
    index_type="faiss.IndexFlatIP",
    index_device="cpu",
    score_sign_filtering="none",
    query_batch_size=64,
    document_batch_size=64,
)
