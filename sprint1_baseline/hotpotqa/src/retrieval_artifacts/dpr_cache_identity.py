"""Scientific identity contract for future DPR embedding and FAISS caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from retrieval_artifacts.corpus_manifest import CorpusManifest
from retrievers.dpr_config import DPR_CONFIG, DPRConfig


DPR_CACHE_IDENTITY_SCHEMA_VERSION = "sprint3.dpr-cache-identity.v1"

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


def dpr_embedding_cache_filename(fingerprint_sha256: str) -> str:
    """Return the machine-independent embedding-cache basename."""
    fingerprint = _require_fingerprint(fingerprint_sha256)
    return f"dpr_embeddings_{fingerprint}.npy"


def dpr_faiss_cache_filename(fingerprint_sha256: str) -> str:
    """Return the machine-independent FAISS-cache basename."""
    fingerprint = _require_fingerprint(fingerprint_sha256)
    return f"dpr_index_{fingerprint}.faiss"


@dataclass(frozen=True)
class DPRCacheIdentity:
    """Corpus- and configuration-bound scientific DPR cache identity.

    Runtime environment and physical artifact metadata deliberately remain
    outside this identity. The validated CorpusManifest already commits to the
    exact ordered retrieval content, so raw document text is not rehashed here.
    """

    schema_version: str
    corpus_manifest: CorpusManifest
    dpr_config: DPRConfig

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        if self.schema_version != DPR_CACHE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported DPR cache identity schema_version")
        if not isinstance(self.corpus_manifest, CorpusManifest):
            raise TypeError("corpus_manifest must be a CorpusManifest")
        if not isinstance(self.dpr_config, DPRConfig):
            raise TypeError("dpr_config must be a DPRConfig")

        manifest_sha256 = self.corpus_manifest.sha256
        manifest_id = self.corpus_manifest.corpus_manifest_id
        if _SHA256_RE.fullmatch(manifest_sha256) is None:
            raise ValueError("CorpusManifest SHA must be a lowercase SHA-256")
        match = _CORPUS_MANIFEST_ID_RE.fullmatch(manifest_id)
        if match is None or match.group(1) != manifest_sha256:
            raise ValueError("CorpusManifest ID must match its scientific SHA")

    def scientific_payload(self) -> dict:
        """Return the inspectable corpus/config binding used for cache reuse."""
        return {
            "corpus": {
                "corpus_manifest_id": self.corpus_manifest.corpus_manifest_id,
                "corpus_manifest_sha256": self.corpus_manifest.sha256,
                "document_count": self.corpus_manifest.document_count,
            },
            "dpr_config": self.dpr_config.scientific_payload(),
            "schema_version": self.schema_version,
        }

    def scientific_json(self) -> str:
        return _canonical_json(self.scientific_payload())

    @property
    def fingerprint_sha256(self) -> str:
        value = hashlib.sha256(self.scientific_json().encode("utf-8")).hexdigest()
        return _require_fingerprint(value)

    @property
    def embedding_cache_filename(self) -> str:
        return dpr_embedding_cache_filename(self.fingerprint_sha256)

    @property
    def faiss_cache_filename(self) -> str:
        return dpr_faiss_cache_filename(self.fingerprint_sha256)


def build_dpr_cache_identity(
    *,
    corpus_manifest: CorpusManifest,
    dpr_config: DPRConfig = DPR_CONFIG,
) -> DPRCacheIdentity:
    """Build the authoritative scientific binding for both DPR cache artifacts."""
    return DPRCacheIdentity(
        schema_version=DPR_CACHE_IDENTITY_SCHEMA_VERSION,
        corpus_manifest=corpus_manifest,
        dpr_config=dpr_config,
    )
