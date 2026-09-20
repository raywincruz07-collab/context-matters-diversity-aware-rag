"""Original DPR dual-encoder runtime backed by authoritative DPRConfig."""

import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
from collections.abc import Mapping
import tempfile
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

from config import EMBEDDINGS_DIR, INDEX_DIR
from retrievers import BaseRetriever
from retrievers.dpr_config import DPR_CONFIG, DPRConfig
from retrieval_artifacts.dpr_cache_identity import build_dpr_cache_identity
from retrieval_artifacts.runtime_corpus import (
    dpr_runtime_documents_from_corpus_records,
)


DPR_CACHE_METADATA_SCHEMA_VERSION = "sprint3.dpr-cache-metadata.v1"


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


class OriginalDPRRetriever(BaseRetriever):
    def __init__(self, config: DPRConfig = DPR_CONFIG):
        super().__init__("dpr")
        if not isinstance(config, DPRConfig):
            raise TypeError("config must be a DPRConfig")
        self.config = config
        self._validate_runtime_config()

        self.q_tokenizer = None
        self.ctx_tokenizer = None
        self.q_encoder = None
        self.ctx_encoder = None

        self.faiss_index = None
        self.embedding_dim = config.embedding_dimension
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_identity = None
        self.embedding_cache_path = None
        self.faiss_cache_path = None
        self.cache_metadata_path = None
        self.embedding_artifact_sha256 = None
        self.index_artifact_sha256 = None

    def _validate_runtime_config(self) -> None:
        """Reject scientific modes this frozen DPR runtime does not implement."""
        supported = {
            "query_preprocessing": "exact query string passed to tokenizer",
            "document_preprocessing": (
                "exact passage text passed to tokenizer paired with empty title"
            ),
            "paired_truncation_strategy": (
                "delegated to tokenizer/Transformers default"
            ),
            "padding_semantics": (
                "dynamic padding to longest encoded input in each batch"
            ),
            "context_title_policy": "empty title paired with passage text",
            "representation": "pooler_output",
            "embedding_dtype": "float32",
            "normalization": "none",
            "score_semantics": (
                "raw dot product / inner product of unnormalized embeddings"
            ),
            "ranking_direction": "higher score is better",
            "index_type": "faiss.IndexFlatIP",
            "score_sign_filtering": "none",
        }
        for field, expected in supported.items():
            if getattr(self.config, field) != expected:
                raise ValueError(
                    f"unsupported DPR {field}: {getattr(self.config, field)!r}"
                )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        if self.q_encoder is not None and self.ctx_encoder is not None:
            return

        print("=" * 70)
        print("Loading ORIGINAL DPR models (facebook dual-encoder)")
        print(f"  Question encoder : {self.config.question_model_id}")
        print(f"  Context encoder  : {self.config.context_model_id}")
        print(f"  Device           : {self.device}")
        print("=" * 70)

        self.q_tokenizer = AutoTokenizer.from_pretrained(
            self.config.question_tokenizer_id,
            revision=self.config.question_tokenizer_revision,
        )
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(
            self.config.context_tokenizer_id,
            revision=self.config.context_tokenizer_revision,
        )

        self.q_encoder = DPRQuestionEncoder.from_pretrained(
            self.config.question_model_id,
            revision=self.config.question_model_revision,
        ).to(self.device)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(
            self.config.context_model_id,
            revision=self.config.context_model_revision,
        ).to(self.device)

        self.q_encoder.eval()
        self.ctx_encoder.eval()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_questions(
        self, questions: List[str], batch_size: int | None = None
    ) -> np.ndarray:
        self._load_models()
        all_emb = []
        effective_batch_size = batch_size or self.config.query_batch_size

        for start in range(0, len(questions), effective_batch_size):
            batch = questions[start : start + effective_batch_size]
            inputs = self.q_tokenizer(
                batch,
                padding=True,
                truncation=self.config.truncation_enabled,
                max_length=self.config.query_max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = self._extract_representation(self.q_encoder(**inputs))

            array = emb.cpu().numpy()
            if array.dtype != np.dtype(self.config.embedding_dtype):
                raise ValueError("DPR question embedding dtype does not match config")
            all_emb.append(array)

        return np.vstack(all_emb)

    def _encode_contexts(
        self, corpus: List[Dict], batch_size: int | None = None
    ) -> np.ndarray:
        texts = []
        for position, document in enumerate(corpus):
            if not isinstance(document, Mapping):
                raise TypeError(f"corpus document {position} must be a mapping")
            if "retrieval_content" not in document:
                raise ValueError(
                    f"corpus document {position} is missing retrieval_content"
                )
            retrieval_content = document["retrieval_content"]
            if not isinstance(retrieval_content, str):
                raise TypeError(
                    f"corpus document {position} retrieval_content must be a string"
                )
            texts.append(retrieval_content)

        self._load_models()
        all_emb = []
        effective_batch_size = batch_size or self.config.context_batch_size
        titles = [""] * len(texts)

        for start in tqdm(
            range(0, len(texts), effective_batch_size),
            desc="Encoding DPR contexts",
        ):
            batch_titles = titles[start : start + effective_batch_size]
            batch_texts = texts[start : start + effective_batch_size]

            # DPR context encoder takes (title, text) as a token-type-separated pair
            inputs = self.ctx_tokenizer(
                batch_titles,
                batch_texts,
                padding=True,
                truncation=self.config.truncation_enabled,
                max_length=self.config.context_max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = self._extract_representation(self.ctx_encoder(**inputs))

            array = emb.cpu().numpy()
            if array.dtype != np.dtype(self.config.embedding_dtype):
                raise ValueError("DPR context embedding dtype does not match config")
            all_emb.append(array)

        return np.vstack(all_emb)

    def _extract_representation(self, encoder_output):
        if self.config.representation != "pooler_output":
            raise ValueError(
                f"unsupported DPR representation: {self.config.representation!r}"
            )
        return encoder_output.pooler_output

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, corpus: List[Dict]):
        raise ValueError(
            "DPR indexing requires index_from_corpus_records() with a validated "
            "CorpusManifest and CorpusRecords"
        )

    def index_from_corpus_records(self, *, corpus_manifest, corpus_records) -> None:
        """Build or reuse DPR caches only after exact manifest/record validation."""
        runtime_documents = dpr_runtime_documents_from_corpus_records(
            corpus_manifest=corpus_manifest,
            corpus_records=corpus_records,
        )
        identity = build_dpr_cache_identity(
            corpus_manifest=corpus_manifest,
            dpr_config=self.config,
        )
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)

        fingerprint = identity.fingerprint_sha256
        embedding_path = os.path.join(
            EMBEDDINGS_DIR, identity.embedding_cache_filename
        )
        faiss_path = os.path.join(INDEX_DIR, identity.faiss_cache_filename)
        metadata_path = os.path.join(INDEX_DIR, f"dpr_cache_{fingerprint}.json")

        cached = self._load_validated_cache(
            corpus_manifest=corpus_manifest,
            cache_identity=identity,
            embedding_path=embedding_path,
            faiss_path=faiss_path,
            metadata_path=metadata_path,
        )
        if cached is None:
            embeddings = self._encode_contexts(runtime_documents)
            self._validate_embeddings(
                embeddings, document_count=corpus_manifest.document_count
            )
            index = faiss.IndexFlatIP(self.config.embedding_dimension)
            index.add(embeddings)
            self._validate_faiss_index(
                index, document_count=corpus_manifest.document_count
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

    def _validate_embeddings(self, embeddings, *, document_count: int) -> None:
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("DPR embeddings must be a numpy.ndarray")
        expected_shape = (document_count, self.config.embedding_dimension)
        if embeddings.ndim != 2 or embeddings.shape != expected_shape:
            raise ValueError(f"DPR embedding shape must be {expected_shape}")
        if embeddings.dtype != np.dtype(self.config.embedding_dtype):
            raise ValueError(
                f"DPR embedding dtype must be {self.config.embedding_dtype}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("DPR embeddings must contain only finite values")

    def _validate_faiss_index(self, index, *, document_count: int) -> None:
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("DPR FAISS index must be faiss.IndexFlatIP")
        if index.d != self.config.embedding_dimension:
            raise ValueError("DPR FAISS dimension does not match config")
        if index.ntotal != document_count:
            raise ValueError("DPR FAISS ntotal does not match corpus")

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
            "cache_schema_version": DPR_CACHE_METADATA_SCHEMA_VERSION,
            "corpus_manifest_id": corpus_manifest.corpus_manifest_id,
            "corpus_manifest_sha256": corpus_manifest.sha256,
            "dpr_scientific_json": self.config.scientific_json(),
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
        if not all(os.path.isfile(path) for path in (metadata_path, embedding_path, faiss_path)):
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
            if _physical_sha256(embedding_path) != embedding_metadata["sha256"]:
                return None
            if _physical_sha256(faiss_path) != faiss_metadata["sha256"]:
                return None

            embeddings = np.load(embedding_path, allow_pickle=False)
            self._validate_embeddings(embeddings, document_count=expected_count)
            index = faiss.read_index(faiss_path)
            self._validate_faiss_index(index, document_count=expected_count)
            if not isinstance(metadata.get("environment"), dict):
                return None
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
                prefix=".dpr_embeddings_",
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
                prefix=".dpr_index_",
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
                prefix=".dpr_cache_",
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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index(corpus) first.")

        query_emb = self._encode_questions([query])
        scores, indices = self.faiss_index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.corpus[int(idx)], float(score)))

        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload_model(self):
        self.q_tokenizer = None
        self.ctx_tokenizer = None
        self.q_encoder = None
        self.ctx_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
