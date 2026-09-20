"""Canonical in-memory Contriever runtime backed by ContrieverConfig."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import tempfile
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import EMBEDDINGS_DIR, INDEX_DIR
from retrievers import BaseRetriever
from retrievers.contriever_config import CONTRIEVER_CONFIG, ContrieverConfig
from retrieval_artifacts.contriever_cache_identity import (
    build_contriever_cache_identity,
)
from retrieval_artifacts.runtime_corpus import (
    contriever_runtime_documents_from_corpus_records,
)


CONTRIEVER_CACHE_METADATA_SCHEMA_VERSION = (
    "sprint3.contriever-cache-metadata.v1"
)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _physical_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContrieverRetriever(BaseRetriever):
    """Pinned, unnormalized Contriever retrieval with an in-memory CPU index."""

    def __init__(self, config: ContrieverConfig = CONTRIEVER_CONFIG):
        super().__init__("contriever")
        if not isinstance(config, ContrieverConfig):
            raise TypeError("config must be a ContrieverConfig")
        self.config = config
        self._validate_runtime_config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.faiss_index = None
        self.embedding_dim = config.embedding_dimension
        self.cache_identity = None
        self.embedding_cache_path = None
        self.faiss_cache_path = None
        self.cache_metadata_path = None
        self.embedding_artifact_sha256 = None
        self.index_artifact_sha256 = None

    def _validate_runtime_config(self) -> None:
        supported = {
            "model_loader": "AutoModel",
            "tokenizer_loader": "AutoTokenizer",
            "query_preprocessing": (
                "exact query string passed to tokenizer; no manual lowercasing or "
                "external text normalization; tokenizer-native behavior preserved"
            ),
            "document_preprocessing": (
                "exact validated CorpusRecord.retrieval_content passed to tokenizer; "
                "no manual lowercasing or external text normalization; "
                "tokenizer-native behavior preserved"
            ),
            "pooling": "attention-mask-aware mean pooling of last_hidden_state",
            "truncation_enabled": True,
            "truncation_semantics": (
                "single-sequence truncation delegated to tokenizer/Transformers"
            ),
            "padding_semantics": (
                "dynamic padding to longest encoded input in each batch"
            ),
            "normalization": "none",
            "compute_dtype": "float32",
            "autocast": False,
            "embedding_dtype": "float32",
            "score_semantics": (
                "raw dot product / inner product of unnormalized embeddings"
            ),
            "ranking_direction": "higher score is better",
            "index_type": "faiss.IndexFlatIP",
            "index_device": "cpu",
            "score_sign_filtering": "none",
        }
        for field, expected in supported.items():
            if getattr(self.config, field) != expected:
                raise ValueError(
                    f"unsupported Contriever {field}: "
                    f"{getattr(self.config, field)!r}"
                )

    def _load_model(self) -> None:
        """Lazily load the exact pinned tokenizer and float32 model."""
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_id,
            revision=self.config.tokenizer_revision,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
        )
        self.model = self.model.to(self.device)
        self.model = self.model.float()
        self.model.eval()

    @staticmethod
    def _mean_pool(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(model_output, "last_hidden_state"):
            raise ValueError("Contriever model output must contain last_hidden_state")
        last_hidden = model_output.last_hidden_state
        if not isinstance(last_hidden, torch.Tensor):
            raise TypeError("last_hidden_state must be a torch.Tensor")
        if last_hidden.ndim != 3:
            raise ValueError("last_hidden_state must have rank 3")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if attention_mask.shape != last_hidden.shape[:2]:
            raise ValueError("attention_mask shape must match last_hidden_state")
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denominator = mask.sum(dim=1)
        if torch.any(denominator == 0):
            raise ValueError("Contriever attention mask contains a zero-token row")
        return (last_hidden * mask).sum(dim=1) / denominator

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be an ordered sequence of strings")
        if not texts:
            raise ValueError("texts must not be empty")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must contain only strings")

        self._load_model()
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=self.config.truncation_enabled,
                max_length=max_length,
                return_tensors="pt",
            )
            if "attention_mask" not in inputs:
                raise ValueError("Contriever tokenizer output requires attention_mask")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs)
                pooled = self._mean_pool(output, inputs["attention_mask"])
            if pooled.dtype != torch.float32:
                raise ValueError("Contriever pooled embeddings must be float32")
            chunks.append(pooled.cpu().numpy())

        embeddings = np.vstack(chunks)
        self._validate_embeddings(embeddings, document_count=len(texts))
        return embeddings

    def _encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode_texts(
            queries,
            batch_size=self.config.query_batch_size,
            max_length=self.config.query_max_length,
        )

    def _encode_documents(self, documents: Sequence[Mapping]) -> np.ndarray:
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be an ordered sequence")
        texts = []
        for position, document in enumerate(documents):
            if not isinstance(document, Mapping):
                raise TypeError(f"corpus document {position} must be a mapping")
            if "retrieval_content" not in document:
                raise ValueError(
                    f"corpus document {position} is missing retrieval_content"
                )
            content = document["retrieval_content"]
            if not isinstance(content, str):
                raise TypeError(
                    f"corpus document {position} retrieval_content must be a string"
                )
            texts.append(content)
        return self._encode_texts(
            texts,
            batch_size=self.config.document_batch_size,
            max_length=self.config.document_max_length,
        )

    def _validate_embeddings(self, embeddings, *, document_count: int) -> None:
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Contriever embeddings must be a numpy.ndarray")
        expected_shape = (document_count, self.config.embedding_dimension)
        if embeddings.ndim != 2 or embeddings.shape != expected_shape:
            raise ValueError(f"Contriever embedding shape must be {expected_shape}")
        if embeddings.dtype != np.dtype(self.config.embedding_dtype):
            raise ValueError(
                f"Contriever embedding dtype must be {self.config.embedding_dtype}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Contriever embeddings must contain only finite values")

    def _validate_faiss_index(self, index, *, document_count: int) -> None:
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Contriever FAISS index must be faiss.IndexFlatIP")
        if index.d != self.config.embedding_dimension:
            raise ValueError("Contriever FAISS dimension does not match config")
        if index.ntotal != document_count:
            raise ValueError("Contriever FAISS ntotal does not match corpus")

    def index(self, corpus: List[Dict]):
        raise ValueError(
            "Contriever indexing requires index_from_corpus_records() with a "
            "validated CorpusManifest and CorpusRecords"
        )

    def index_from_corpus_records(self, *, corpus_manifest, corpus_records) -> None:
        runtime_documents = contriever_runtime_documents_from_corpus_records(
            corpus_manifest=corpus_manifest,
            corpus_records=corpus_records,
        )
        identity = build_contriever_cache_identity(
            corpus_manifest=corpus_manifest,
            contriever_config=self.config,
        )
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)

        fingerprint = identity.fingerprint_sha256
        embedding_path = os.path.join(
            EMBEDDINGS_DIR, identity.embedding_cache_filename
        )
        faiss_path = os.path.join(INDEX_DIR, identity.faiss_cache_filename)
        metadata_path = os.path.join(
            INDEX_DIR, f"contriever_cache_{fingerprint}.json"
        )

        cached = self._load_validated_cache(
            corpus_manifest=corpus_manifest,
            cache_identity=identity,
            embedding_path=embedding_path,
            faiss_path=faiss_path,
            metadata_path=metadata_path,
        )
        if cached is None:
            embeddings = self._encode_documents(runtime_documents)
            self._validate_embeddings(
                embeddings,
                document_count=corpus_manifest.document_count,
            )
            index = faiss.IndexFlatIP(self.config.embedding_dimension)
            index.add(embeddings)
            self._validate_faiss_index(
                index,
                document_count=corpus_manifest.document_count,
            )
            embedding_artifact_sha256, index_artifact_sha256 = (
                self._write_cache_pair(
                    corpus_manifest=corpus_manifest,
                    cache_identity=identity,
                    embeddings=embeddings,
                    index=index,
                    embedding_path=embedding_path,
                    faiss_path=faiss_path,
                    metadata_path=metadata_path,
                )
            )
            new_index = index
        else:
            _, new_index, metadata = cached
            embedding_artifact_sha256 = metadata["embedding"]["sha256"]
            index_artifact_sha256 = metadata["faiss"]["sha256"]

        (
            self.corpus,
            self.cache_identity,
            self.embedding_cache_path,
            self.faiss_cache_path,
            self.cache_metadata_path,
            self.faiss_index,
            self.embedding_artifact_sha256,
            self.index_artifact_sha256,
            self.is_indexed,
        ) = (
            runtime_documents,
            identity,
            embedding_path,
            faiss_path,
            metadata_path,
            new_index,
            embedding_artifact_sha256,
            index_artifact_sha256,
            True,
        )

    def _environment_metadata(self) -> dict:
        return {
            "device": self.device,
            "faiss_version": getattr(faiss, "__version__", None)
            or _package_version("faiss-cpu"),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": _package_version("transformers"),
        }

    def _expected_metadata_binding(self, corpus_manifest, *, cache_identity) -> dict:
        return {
            "cache_fingerprint_sha256": cache_identity.fingerprint_sha256,
            "cache_identity_schema_version": cache_identity.schema_version,
            "cache_schema_version": CONTRIEVER_CACHE_METADATA_SCHEMA_VERSION,
            "corpus_manifest_id": corpus_manifest.corpus_manifest_id,
            "corpus_manifest_sha256": corpus_manifest.sha256,
            "contriever_scientific_json": self.config.scientific_json(),
            "scientific_payload": cache_identity.scientific_payload(),
        }

    def _load_validated_cache(
        self,
        *,
        corpus_manifest,
        cache_identity,
        embedding_path,
        faiss_path,
        metadata_path,
    ):
        if not all(
            os.path.isfile(path)
            for path in (metadata_path, embedding_path, faiss_path)
        ):
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            if not isinstance(metadata, dict):
                return None
            for key, value in self._expected_metadata_binding(
                corpus_manifest, cache_identity=cache_identity
            ).items():
                if metadata.get(key) != value:
                    return None

            embedding_metadata = metadata.get("embedding")
            faiss_metadata = metadata.get("faiss")
            if not isinstance(embedding_metadata, dict) or not isinstance(
                faiss_metadata, dict
            ):
                return None
            expected_count = corpus_manifest.document_count
            expected_shape = [expected_count, self.config.embedding_dimension]
            if embedding_metadata != {
                "document_count": expected_count,
                "dtype": self.config.embedding_dtype,
                "embedding_dimension": self.config.embedding_dimension,
                "filename": os.path.basename(embedding_path),
                "sha256": embedding_metadata.get("sha256"),
                "shape": expected_shape,
            }:
                return None
            if faiss_metadata != {
                "dimension": self.config.embedding_dimension,
                "filename": os.path.basename(faiss_path),
                "index_type": self.config.index_type,
                "ntotal": expected_count,
                "sha256": faiss_metadata.get("sha256"),
            }:
                return None
            if not isinstance(metadata.get("environment"), dict):
                return None
            if _physical_sha256(embedding_path) != embedding_metadata["sha256"]:
                return None
            if _physical_sha256(faiss_path) != faiss_metadata["sha256"]:
                return None

            embeddings = np.load(embedding_path, allow_pickle=False)
            self._validate_embeddings(embeddings, document_count=expected_count)
            index = faiss.read_index(faiss_path)
            self._validate_faiss_index(index, document_count=expected_count)
            return embeddings, index, metadata
        except Exception:
            return None

    def _write_cache_pair(
        self,
        *,
        corpus_manifest,
        cache_identity,
        embeddings,
        index,
        embedding_path,
        faiss_path,
        metadata_path,
    ) -> tuple[str, str]:
        self._write_numpy_atomically(embedding_path, embeddings)
        self._write_faiss_atomically(faiss_path, index)
        embedding_sha256 = _physical_sha256(embedding_path)
        index_sha256 = _physical_sha256(faiss_path)
        metadata = {
            **self._expected_metadata_binding(
                corpus_manifest, cache_identity=cache_identity
            ),
            "embedding": {
                "document_count": corpus_manifest.document_count,
                "dtype": self.config.embedding_dtype,
                "embedding_dimension": self.config.embedding_dimension,
                "filename": os.path.basename(embedding_path),
                "sha256": embedding_sha256,
                "shape": list(embeddings.shape),
            },
            "environment": self._environment_metadata(),
            "faiss": {
                "dimension": index.d,
                "filename": os.path.basename(faiss_path),
                "index_type": self.config.index_type,
                "ntotal": index.ntotal,
                "sha256": index_sha256,
            },
        }
        self._write_json_atomically(metadata_path, metadata)
        return embedding_sha256, index_sha256

    @staticmethod
    def _write_numpy_atomically(path: str, embeddings: np.ndarray) -> None:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".contriever_embeddings_",
                suffix=".npy.tmp",
                dir=os.path.dirname(path),
                delete=False,
            ) as file:
                temporary_path = file.name
                np.save(file, embeddings, allow_pickle=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _write_faiss_atomically(path: str, index) -> None:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".contriever_index_",
                suffix=".faiss.tmp",
                dir=os.path.dirname(path),
                delete=False,
            ) as file:
                temporary_path = file.name
            faiss.write_index(index, temporary_path)
            with open(temporary_path, "rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _write_json_atomically(path: str, metadata: dict) -> None:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".contriever_cache_",
                suffix=".json.tmp",
                dir=os.path.dirname(path),
                delete=False,
            ) as file:
                temporary_path = file.name
                json.dump(
                    metadata,
                    file,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index_from_corpus_records() first.")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive non-boolean integer")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self._encode_queries([query])
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if int(index) == -1:
                continue
            results.append((self.corpus[int(index)], float(score)))
        return results

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
