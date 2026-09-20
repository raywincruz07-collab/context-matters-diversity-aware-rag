"""Dependency-neutral authoritative BM25 configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class BM25Config:
    """Structured runtime and scientific configuration for BM25."""

    tokenizer: None = None
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    query_preprocessing: str = "lowercase then split on whitespace"
    document_preprocessing: str = (
        "lowercase then split exact document text on whitespace"
    )
    normalization: str = "none"
    score_semantics: str = "rank_bm25.BM25Okapi native get_scores score"
    ranking_semantics: str = (
        "native score descending; exact score ties by frozen corpus position "
        "ascending; no score-sign filtering"
    )
    index_type: str = "in-memory rank_bm25.BM25Okapi"

    def __post_init__(self) -> None:
        if self.tokenizer is not None:
            raise ValueError("tokenizer must be None for the current BM25 method")
        for name in ("k1", "b", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "query_preprocessing",
            "document_preprocessing",
            "normalization",
            "score_semantics",
            "ranking_semantics",
            "index_type",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def scientific_payload(self) -> dict:
        """Return the complete deterministic configuration structure."""
        return {
            "b": float(self.b),
            "document_preprocessing": self.document_preprocessing,
            "epsilon": float(self.epsilon),
            "index_type": self.index_type,
            "k1": float(self.k1),
            "normalization": self.normalization,
            "query_preprocessing": self.query_preprocessing,
            "ranking_semantics": self.ranking_semantics,
            "score_semantics": self.score_semantics,
            "tokenizer": self.tokenizer,
        }

    def scientific_json(self) -> str:
        """Serialize explicit BM25 parameters and semantics canonically."""
        return json.dumps(
            self.scientific_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


BM25_CONFIG = BM25Config()
