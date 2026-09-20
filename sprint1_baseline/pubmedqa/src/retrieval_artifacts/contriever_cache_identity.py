"""Scientific identity contract for future Contriever embedding/FAISS caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from retrieval_artifacts.corpus_manifest import CorpusManifest
from retrievers.contriever_config import CONTRIEVER_CONFIG, ContrieverConfig


CONTRIEVER_CACHE_IDENTITY_SCHEMA_VERSION = (
    "sprint3.contriever-cache-identity.v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CORPUS_MANIFEST_ID_RE = re.compile(r"^corpus-manifest:sha256:([0-9a-f]{64})$")


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_fingerprint(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("fingerprint must be a lowercase 64-character SHA-256")
    return value


def contriever_embedding_cache_filename(fingerprint_sha256: str) -> str:
    """Return the machine-independent Contriever embedding-cache basename."""
    fingerprint = _require_fingerprint(fingerprint_sha256)
    return f"contriever_embeddings_{fingerprint}.npy"


def contriever_faiss_cache_filename(fingerprint_sha256: str) -> str:
    """Return the machine-independent Contriever FAISS-cache basename."""
    fingerprint = _require_fingerprint(fingerprint_sha256)
    return f"contriever_index_{fingerprint}.faiss"


@dataclass(frozen=True)
class ContrieverCacheIdentity:
    """Corpus- and configuration-bound scientific Contriever cache identity."""

    schema_version: str
    corpus_manifest: CorpusManifest
    contriever_config: ContrieverConfig

    def __post_init__(self) -> None:
        if self.schema_version != CONTRIEVER_CACHE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported Contriever cache identity schema_version")
        if not isinstance(self.corpus_manifest, CorpusManifest):
            raise TypeError("corpus_manifest must be a CorpusManifest")
        if not isinstance(self.contriever_config, ContrieverConfig):
            raise TypeError("contriever_config must be a ContrieverConfig")

        manifest_sha256 = self.corpus_manifest.sha256
        manifest_id = self.corpus_manifest.corpus_manifest_id
        if not isinstance(manifest_sha256, str) or _SHA256_RE.fullmatch(
            manifest_sha256
        ) is None:
            raise ValueError("CorpusManifest SHA must be a lowercase SHA-256")
        match = (
            _CORPUS_MANIFEST_ID_RE.fullmatch(manifest_id)
            if isinstance(manifest_id, str)
            else None
        )
        if match is None or match.group(1) != manifest_sha256:
            raise ValueError("CorpusManifest ID must match its scientific SHA")

    def scientific_payload(self) -> dict:
        """Return the exact scientific corpus/config binding for cache reuse."""
        return {
            "corpus": {
                "corpus_manifest_id": self.corpus_manifest.corpus_manifest_id,
                "corpus_manifest_sha256": self.corpus_manifest.sha256,
                "document_count": self.corpus_manifest.document_count,
            },
            "contriever_config": self.contriever_config.scientific_payload(),
            "schema_version": self.schema_version,
        }

    def scientific_json(self) -> str:
        return _canonical_json(self.scientific_payload())

    @property
    def fingerprint_sha256(self) -> str:
        fingerprint = hashlib.sha256(
            self.scientific_json().encode("utf-8")
        ).hexdigest()
        return _require_fingerprint(fingerprint)

    @property
    def embedding_cache_filename(self) -> str:
        return contriever_embedding_cache_filename(self.fingerprint_sha256)

    @property
    def faiss_cache_filename(self) -> str:
        return contriever_faiss_cache_filename(self.fingerprint_sha256)


def build_contriever_cache_identity(
    *,
    corpus_manifest: CorpusManifest,
    contriever_config: ContrieverConfig = CONTRIEVER_CONFIG,
) -> ContrieverCacheIdentity:
    """Build the authoritative scientific Contriever cache binding."""
    return ContrieverCacheIdentity(
        schema_version=CONTRIEVER_CACHE_IDENTITY_SCHEMA_VERSION,
        corpus_manifest=corpus_manifest,
        contriever_config=contriever_config,
    )
