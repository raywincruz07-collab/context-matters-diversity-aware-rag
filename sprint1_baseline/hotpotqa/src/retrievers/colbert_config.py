"""Dependency-neutral authoritative ColBERTv2 scientific configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
import re


_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ColBERTConfig:
    """Structured scientific definition of the canonical ColBERTv2 method."""

    retriever_name: str
    backend: str
    colbert_ai_version: str
    checkpoint_id: str
    checkpoint_revision: str
    checkpoint_loading: str
    dim: int
    query_maxlen: int
    doc_maxlen: int
    similarity: str
    interaction: str
    mask_punctuation: bool
    attend_to_mask_tokens: bool
    query_preprocessing: str
    document_preprocessing: str
    index_engine: str
    nbits: int
    kmeans_niters: int
    index_bsize: int
    nranks: int
    amp: bool
    seed: int
    candidate_pool_size: int
    search_ncells: int
    search_centroid_score_threshold: float
    search_ndocs: int
    ranking: str
    post_filtering: str
    post_reranking: str
    tie_manipulation: str

    def __post_init__(self) -> None:
        for name in (
            "retriever_name",
            "backend",
            "colbert_ai_version",
            "checkpoint_id",
            "checkpoint_revision",
            "checkpoint_loading",
            "similarity",
            "interaction",
            "query_preprocessing",
            "document_preprocessing",
            "index_engine",
            "ranking",
            "post_filtering",
            "post_reranking",
            "tie_manipulation",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if _FULL_GIT_SHA.fullmatch(self.checkpoint_revision) is None:
            raise ValueError(
                "checkpoint_revision must be a full lowercase 40-character "
                "hexadecimal SHA"
            )

        for name in (
            "dim",
            "query_maxlen",
            "doc_maxlen",
            "nbits",
            "kmeans_niters",
            "index_bsize",
            "nranks",
            "candidate_pool_size",
            "search_ncells",
            "search_ndocs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive non-boolean integer")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative non-boolean integer")

        threshold = self.search_centroid_score_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold))
        ):
            raise ValueError(
                "search_centroid_score_threshold must be a finite non-boolean real"
            )

        for name in ("mask_punctuation", "attend_to_mask_tokens", "amp"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def scientific_payload(self) -> dict:
        """Return the complete ColBERTv2 retrieval-method identity.

        This full identity includes model, indexing, and search behavior. A future
        physical PLAID index-cache identity must instead use only fields capable
        of affecting checkpoint encoding or index construction. In particular,
        it must not include candidate_pool_size, search_ncells,
        search_centroid_score_threshold, search_ndocs, ranking, post_filtering,
        post_reranking, or tie_manipulation merely because they occur here.
        """
        return {
            "attend_to_mask_tokens": self.attend_to_mask_tokens,
            "amp": self.amp,
            "backend": self.backend,
            "candidate_pool_size": self.candidate_pool_size,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_loading": self.checkpoint_loading,
            "checkpoint_revision": self.checkpoint_revision,
            "colbert_ai_version": self.colbert_ai_version,
            "dim": self.dim,
            "doc_maxlen": self.doc_maxlen,
            "document_preprocessing": self.document_preprocessing,
            "index_bsize": self.index_bsize,
            "index_engine": self.index_engine,
            "interaction": self.interaction,
            "kmeans_niters": self.kmeans_niters,
            "mask_punctuation": self.mask_punctuation,
            "nbits": self.nbits,
            "nranks": self.nranks,
            "post_filtering": self.post_filtering,
            "post_reranking": self.post_reranking,
            "query_maxlen": self.query_maxlen,
            "query_preprocessing": self.query_preprocessing,
            "ranking": self.ranking,
            "retriever_name": self.retriever_name,
            "search_centroid_score_threshold": float(
                self.search_centroid_score_threshold
            ),
            "search_ncells": self.search_ncells,
            "search_ndocs": self.search_ndocs,
            "seed": self.seed,
            "similarity": self.similarity,
            "tie_manipulation": self.tie_manipulation,
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


COLBERT_CONFIG = ColBERTConfig(
    retriever_name="colbertv2",
    backend="direct Stanford ColBERT",
    colbert_ai_version="0.2.22",
    checkpoint_id="colbert-ir/colbertv2.0",
    checkpoint_revision="0855eac81381e0323a846f1ed7d8452d4c648b50",
    checkpoint_loading=(
        "resolve the pinned revision without fallback; validate and fingerprint "
        "the physical checkpoint snapshot before canonical indexing"
    ),
    dim=128,
    query_maxlen=32,
    doc_maxlen=180,
    similarity="cosine",
    interaction="ColBERT late interaction / MaxSim",
    mask_punctuation=True,
    attend_to_mask_tokens=False,
    query_preprocessing=(
        "exact query string; no manual lowercasing or external text normalization; "
        "tokenizer-native behavior preserved"
    ),
    document_preprocessing=(
        "exact validated CorpusRecord.retrieval_content; no manual lowercasing or "
        "external text normalization; tokenizer-native behavior preserved"
    ),
    index_engine="Stanford ColBERT PLAID",
    nbits=2,
    kmeans_niters=4,
    index_bsize=64,
    nranks=1,
    amp=True,
    seed=12345,
    candidate_pool_size=20,
    search_ncells=2,
    search_centroid_score_threshold=0.45,
    search_ndocs=1024,
    ranking="native ColBERT ranking, higher score is better",
    post_filtering="none",
    post_reranking="none",
    tie_manipulation="none",
)
