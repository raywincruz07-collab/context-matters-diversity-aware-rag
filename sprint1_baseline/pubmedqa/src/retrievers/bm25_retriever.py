"""BM25 sparse retriever using a corpus-bound, validated runtime cache.

The runtime cache fingerprint only proves that a reused local BM25 object matches
the exact ordered corpus and runtime configuration. CandidateArtifact creation and
scientific retrieval provenance remain the responsibility of
``retrieval_artifacts.producer``.
"""

from collections.abc import Mapping
import hashlib
from importlib import metadata as importlib_metadata
import json
from numbers import Integral
import os
import pickle
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from config import INDEX_DIR
from retrievers import BaseRetriever
from retrievers.bm25_config import BM25_CONFIG, BM25Config


BM25_CACHE_SCHEMA_VERSION = "sprint3.bm25-cache.v1"


def _rank_bm25_version() -> str:
    return importlib_metadata.version("rank-bm25")


def _canonical_json(payload) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_doc_id(document_id):
    return document_id if isinstance(document_id, str) else int(document_id)


class BM25Retriever(BaseRetriever):
    def __init__(self, runtime_config: BM25Config = BM25_CONFIG):
        super().__init__("bm25")
        if not isinstance(runtime_config, BM25Config):
            raise TypeError("runtime_config must be BM25Config")
        self._runtime_config = runtime_config
        self._runtime_index_fingerprint = None
        self._cache_path = None
        self.bm25 = None
        self.tokenized_corpus = None

    @property
    def runtime_index_fingerprint(self):
        return self._runtime_index_fingerprint

    @property
    def cache_path(self):
        return self._cache_path

    @property
    def runtime_config(self):
        return self._runtime_config

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def _validate_corpus(self, corpus: List[Dict]) -> None:
        if not isinstance(corpus, list):
            raise TypeError("corpus must be a list")
        if not corpus:
            raise ValueError("corpus must not be empty")
        document_ids = []
        for position, document in enumerate(corpus):
            if not isinstance(document, Mapping):
                raise TypeError(f"corpus document at position {position} must be a mapping")
            if "doc_id" not in document:
                raise ValueError(f"corpus document at position {position} is missing doc_id")
            document_id = document["doc_id"]
            if isinstance(document_id, (bool, np.bool_)) or not isinstance(
                document_id, (str, Integral)
            ):
                raise TypeError("doc_id must be a string or non-boolean integer")
            if isinstance(document_id, str) and not document_id.strip():
                raise ValueError("doc_id strings must be non-empty")
            if "text" not in document:
                raise ValueError(f"corpus document at position {position} is missing text")
            if not isinstance(document["text"], str):
                raise TypeError("document text must be a string")
            document_ids.append(document_id)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("corpus doc_id values must be unique")

    def _runtime_fingerprint(self, corpus: List[Dict], library_version: str) -> str:
        corpus_payload = [
            {
                "corpus_position": position,
                "doc_id": _canonical_doc_id(document["doc_id"]),
                "document_content_sha256": hashlib.sha256(
                    document["text"].encode("utf-8")
                ).hexdigest(),
            }
            for position, document in enumerate(corpus)
        ]
        payload = {
            "bm25_runtime_config": self._runtime_config.scientific_payload(),
            "corpus": corpus_payload,
            "corpus_document_count": len(corpus),
            "library_name": "rank_bm25",
            "library_version": library_version,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _expected_cache_metadata(
        self, fingerprint: str, library_version: str, document_count: int
    ) -> dict:
        return {
            "cache_schema_version": BM25_CACHE_SCHEMA_VERSION,
            "runtime_fingerprint": fingerprint,
            "rank_bm25_version": library_version,
            "bm25_runtime_config": self._runtime_config.scientific_payload(),
            "corpus_document_count": document_count,
        }

    def _load_validated_cache(
        self, path: str, expected: dict, expected_tokenized_corpus: List[List[str]]
    ) -> bool:
        try:
            with open(path, "rb") as file:
                payload = pickle.load(file)
            if not isinstance(payload, dict):
                return False
            for key, value in expected.items():
                if payload.get(key) != value:
                    return False
            bm25 = payload.get("bm25")
            tokenized_corpus = payload.get("tokenized_corpus")
            document_count = expected["corpus_document_count"]
            if not isinstance(bm25, BM25Okapi):
                return False
            if not isinstance(tokenized_corpus, list):
                return False
            if len(tokenized_corpus) != document_count:
                return False
            if tokenized_corpus != expected_tokenized_corpus:
                return False
            if getattr(bm25, "corpus_size", None) != document_count:
                return False
        except Exception:
            # A local cache may reference unavailable Python modules/classes as well
            # as contain malformed pickle bytes. No cache read failure is trusted.
            return False
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        return True

    def _build_index(self, tokenized_corpus: List[List[str]]) -> None:
        self.tokenized_corpus = tokenized_corpus
        config = self._runtime_config
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            tokenizer=config.tokenizer,
            k1=config.k1,
            b=config.b,
            epsilon=config.epsilon,
        )

    def _write_cache_atomically(self, path: str, metadata: dict) -> None:
        payload = {
            **metadata,
            "bm25": self.bm25,
            "tokenized_corpus": self.tokenized_corpus,
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".bm25_", suffix=".tmp", dir=INDEX_DIR, delete=False
            ) as file:
                temporary_path = file.name
                pickle.dump(payload, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def index(self, corpus: List[Dict]):
        """Build or safely reuse an index bound to this exact ordered corpus."""
        self._validate_corpus(corpus)
        self.corpus = corpus
        os.makedirs(INDEX_DIR, exist_ok=True)
        library_version = _rank_bm25_version()
        fingerprint = self._runtime_fingerprint(corpus, library_version)
        cache_path = os.path.join(INDEX_DIR, f"bm25_{fingerprint}.pkl")
        self._runtime_index_fingerprint = fingerprint
        self._cache_path = cache_path
        metadata = self._expected_cache_metadata(
            fingerprint, library_version, len(corpus)
        )
        expected_tokenized_corpus = [
            self._tokenize(document["text"]) for document in corpus
        ]

        legacy_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
        if os.path.exists(legacy_path):
            print("Ignoring unverifiable legacy BM25 cache: bm25_index.pkl")

        if os.path.exists(cache_path) and self._load_validated_cache(
            cache_path, metadata, expected_tokenized_corpus
        ):
            print("Loading validated BM25 index from disk...")
        else:
            print("Building BM25 index...")
            self._build_index(expected_tokenized_corpus)
            self._write_cache_atomically(cache_path, metadata)
            print("BM25 index saved.")

        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using BM25 scoring."""
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        corpus_positions = np.arange(len(scores))
        ranked_indices = np.lexsort((corpus_positions, -scores))
        top_indices = ranked_indices[:top_k]
        return [
            (self.corpus[idx], float(scores[idx]))
            for idx in top_indices
        ]
