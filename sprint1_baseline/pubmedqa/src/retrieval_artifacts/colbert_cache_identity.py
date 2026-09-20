"""Canonical scientific and physical identities for Stanford PLAID indexes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from retrieval_artifacts.corpus_manifest import CorpusManifest
from retrievers.colbert_config import COLBERT_CONFIG, ColBERTConfig


COLBERT_CACHE_IDENTITY_SCHEMA_VERSION = "sprint3.colbert-cache-identity.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def colbert_index_settings(config: ColBERTConfig = COLBERT_CONFIG) -> dict:
    """Return only settings capable of changing the physical PLAID index."""
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    return {
        "attend_to_mask_tokens": config.attend_to_mask_tokens,
        "amp": config.amp,
        "backend": config.backend,
        "checkpoint_id": config.checkpoint_id,
        "checkpoint_revision": config.checkpoint_revision,
        "colbert_ai_version": config.colbert_ai_version,
        "dim": config.dim,
        "doc_maxlen": config.doc_maxlen,
        "document_preprocessing": config.document_preprocessing,
        "index_bsize": config.index_bsize,
        "index_engine": config.index_engine,
        "kmeans_niters": config.kmeans_niters,
        "mask_punctuation": config.mask_punctuation,
        "nbits": config.nbits,
        "nranks": config.nranks,
        "seed": config.seed,
        "similarity": config.similarity,
    }


def colbert_search_settings(config: ColBERTConfig = COLBERT_CONFIG) -> dict:
    """Return search/output controls which must not alter index identity."""
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    return {
        "candidate_pool_size": config.candidate_pool_size,
        "post_filtering": config.post_filtering,
        "post_reranking": config.post_reranking,
        "query_maxlen": config.query_maxlen,
        "query_preprocessing": config.query_preprocessing,
        "ranking": config.ranking,
        "search_centroid_score_threshold": float(
            config.search_centroid_score_threshold
        ),
        "search_ncells": config.search_ncells,
        "search_ndocs": config.search_ndocs,
        "tie_manipulation": config.tie_manipulation,
    }


@dataclass(frozen=True)
class ColBERTCacheIdentity:
    schema_version: str
    corpus_manifest: CorpusManifest
    checkpoint_snapshot_manifest_sha256: str
    colbert_config: ColBERTConfig

    def __post_init__(self) -> None:
        if self.schema_version != COLBERT_CACHE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported ColBERT cache identity schema_version")
        if not isinstance(self.corpus_manifest, CorpusManifest):
            raise TypeError("corpus_manifest must be a CorpusManifest")
        _require_sha256(
            self.checkpoint_snapshot_manifest_sha256,
            "checkpoint_snapshot_manifest_sha256",
        )
        if not isinstance(self.colbert_config, ColBERTConfig):
            raise TypeError("colbert_config must be a ColBERTConfig")

    def scientific_payload(self) -> dict:
        return {
            "checkpoint_snapshot_manifest_sha256": (
                self.checkpoint_snapshot_manifest_sha256
            ),
            "corpus": {
                "corpus_manifest_id": self.corpus_manifest.corpus_manifest_id,
                "corpus_manifest_sha256": self.corpus_manifest.sha256,
                "document_count": self.corpus_manifest.document_count,
            },
            "index_settings": colbert_index_settings(self.colbert_config),
            "schema_version": self.schema_version,
        }

    @property
    def fingerprint_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.scientific_payload()).encode("utf-8")
        ).hexdigest()

    @property
    def index_name(self) -> str:
        return f"pubmedqa_colbertv2_{self.fingerprint_sha256}"


def build_colbert_cache_identity(
    *,
    corpus_manifest: CorpusManifest,
    checkpoint_snapshot_manifest_sha256: str,
    colbert_config: ColBERTConfig = COLBERT_CONFIG,
) -> ColBERTCacheIdentity:
    return ColBERTCacheIdentity(
        schema_version=COLBERT_CACHE_IDENTITY_SCHEMA_VERSION,
        corpus_manifest=corpus_manifest,
        checkpoint_snapshot_manifest_sha256=checkpoint_snapshot_manifest_sha256,
        colbert_config=colbert_config,
    )


def fingerprint_colbert_index_directory(index_directory: Path) -> str:
    """Hash every regular file in a PLAID directory in stable path order."""
    directory = Path(index_directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"PLAID index directory does not exist: {directory}")
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not files:
        raise ValueError("PLAID index directory contains no files")
    manifest = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({
            "relative_path": path.relative_to(directory).as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
        })
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
