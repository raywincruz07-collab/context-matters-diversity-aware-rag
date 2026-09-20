"""Controlled direct-Stanford-ColBERT configuration and checkpoint adapter."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any

from retrievers.colbert_checkpoint import ResolvedColBERTCheckpoint
from retrievers.colbert_checkpoint_provenance import (
    ColBERTCheckpointPhysicalProvenance,
)
from retrievers.colbert_config import COLBERT_CONFIG, ColBERTConfig
from retrieval_artifacts.colbert_cache_identity import (
    ColBERTCacheIdentity,
    build_colbert_cache_identity,
    fingerprint_colbert_index_directory,
)
from retrieval_artifacts.corpus_manifest import CorpusManifest
from retrieval_artifacts.producer import CorpusRecord, RawCandidateResult
from retrieval_artifacts.runtime_corpus import (
    colbert_runtime_collection_from_corpus_records,
)


_SCIENTIFIC_INTERACTION = "ColBERT late interaction / MaxSim"
_STANFORD_INTERACTION = "colbert"
_TRANSFORMERS_VERSION = "4.57.6"
_HUGGINGFACE_HUB_VERSION = "0.36.2"
_INDEX_COMPLETION_SCHEMA_VERSION = "sprint3.colbert-index-completion.v1"


@dataclass(frozen=True)
class StanfordColBERTEffectiveConfig:
    """Validated effective Stanford settings retained as runtime provenance."""

    checkpoint: str
    dim: int
    query_maxlen: int
    doc_maxlen: int
    similarity: str
    interaction: str
    mask_punctuation: bool
    attend_to_mask_tokens: bool
    index_bsize: int
    nbits: int
    kmeans_niters: int
    nranks: int
    amp: bool
    ncells: int
    centroid_score_threshold: float
    ndocs: int

    def payload(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class InitializedColBERTCheckpoint:
    """Initialized Stanford checkpoint plus validated effective runtime state."""

    checkpoint: Any
    resolved_checkpoint: ResolvedColBERTCheckpoint
    effective_config: StanfordColBERTEffectiveConfig
    colbert_ai_version: str
    transformers_version: str
    huggingface_hub_version: str


def _stanford_config_class():
    from colbert.infra import ColBERTConfig as StanfordColBERTConfig

    return StanfordColBERTConfig


def _stanford_checkpoint_class():
    from colbert.modeling.checkpoint import Checkpoint

    return Checkpoint


def _stanford_indexer_class():
    from colbert import Indexer

    return Indexer


def _stanford_searcher_class():
    from colbert import Searcher

    return Searcher


def _stanford_run_classes():
    from colbert.infra import Run, RunConfig

    return Run, RunConfig


def apply_colbert_index_seed(seed: int) -> None:
    """Initialize controlled RNGs; this does not promise bit-exact GPU output."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative non-boolean integer")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require_colbert_runtime_prerequisites() -> None:
    """Match PyTorch's practical Ninja availability check without installing it."""
    try:
        subprocess.check_output(
            ["ninja", "--version"], stderr=subprocess.STDOUT
        )
    except Exception as error:
        raise RuntimeError("ColBERT runtime prerequisite missing: ninja") from error


def _require_distribution_version(distribution: str, expected: str) -> str:
    try:
        installed = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"required distribution {distribution}=={expected} is not installed"
        ) from error
    if installed != expected:
        raise RuntimeError(
            f"{distribution} version mismatch: expected {expected}, got {installed}"
        )
    return installed


def _require_runtime_versions(config: ColBERTConfig) -> dict[str, str]:
    return {
        "colbert_ai_version": _require_distribution_version(
            "colbert-ai", config.colbert_ai_version
        ),
        "transformers_version": _require_distribution_version(
            "transformers", _TRANSFORMERS_VERSION
        ),
        "huggingface_hub_version": _require_distribution_version(
            "huggingface-hub", _HUGGINGFACE_HUB_VERSION
        ),
    }


def _validated_snapshot_path(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    config: ColBERTConfig,
) -> Path:
    if not isinstance(resolved_checkpoint, ResolvedColBERTCheckpoint):
        raise TypeError("resolved_checkpoint must be a ResolvedColBERTCheckpoint")
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    if resolved_checkpoint.checkpoint_id != config.checkpoint_id:
        raise ValueError("resolved checkpoint ID does not match ColBERTConfig")
    if resolved_checkpoint.checkpoint_revision != config.checkpoint_revision:
        raise ValueError("resolved checkpoint revision does not match ColBERTConfig")

    snapshot_path = resolved_checkpoint.snapshot_path.resolve()
    if not snapshot_path.is_dir():
        raise NotADirectoryError("resolved ColBERT snapshot must be a directory")
    if snapshot_path.name != config.checkpoint_revision:
        raise ValueError("resolved snapshot directory does not match checkpoint revision")
    return snapshot_path


def _expected_stanford_values(
    snapshot_path: Path,
    config: ColBERTConfig,
) -> dict[str, Any]:
    if config.interaction != _SCIENTIFIC_INTERACTION:
        raise ValueError("unsupported ColBERT scientific interaction")
    return {
        "checkpoint": str(snapshot_path),
        "dim": config.dim,
        "query_maxlen": config.query_maxlen,
        "doc_maxlen": config.doc_maxlen,
        "similarity": config.similarity,
        # Stanford names the frozen ColBERT late-interaction/MaxSim mode "colbert".
        "interaction": _STANFORD_INTERACTION,
        "mask_punctuation": config.mask_punctuation,
        "attend_to_mask_tokens": config.attend_to_mask_tokens,
        "index_bsize": config.index_bsize,
        "nbits": config.nbits,
        "kmeans_niters": config.kmeans_niters,
        "nranks": config.nranks,
        "amp": config.amp,
        "ncells": config.search_ncells,
        "centroid_score_threshold": float(
            config.search_centroid_score_threshold
        ),
        "ndocs": config.search_ndocs,
    }


def validate_effective_stanford_config(
    stanford_config: Any,
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
) -> StanfordColBERTEffectiveConfig:
    """Validate every translated value after Stanford config merging."""
    snapshot_path = _validated_snapshot_path(resolved_checkpoint, config)
    expected = _expected_stanford_values(snapshot_path, config)
    for field, expected_value in expected.items():
        if not hasattr(stanford_config, field):
            raise ValueError(f"Stanford ColBERTConfig is missing {field}")
        actual = getattr(stanford_config, field)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"effective Stanford ColBERTConfig {field} mismatch: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    return StanfordColBERTEffectiveConfig(**expected)


def build_stanford_colbert_config(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
):
    """Translate the frozen project method into explicit Stanford settings.

    Stanford performs the learned 768→128 projection, token-vector L2
    normalization, document padding/punctuation masking, and ColBERT MaxSim sum.
    This adapter does not normalize, lowercase, or score text itself.

    Candidate-pool/ranking/post-processing controls are intentionally not
    Stanford constructor parameters. The project seed is likewise deferred to
    the later indexing milestone because Stanford ColBERTConfig has no project
    seed field.
    """
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    _require_runtime_versions(config)
    return _build_stanford_colbert_config(resolved_checkpoint, config=config)


def _build_stanford_colbert_config(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig,
):
    snapshot_path = _validated_snapshot_path(resolved_checkpoint, config)
    values = _expected_stanford_values(snapshot_path, config)
    stanford_config = _stanford_config_class()(**values)
    validate_effective_stanford_config(
        stanford_config,
        resolved_checkpoint,
        config=config,
    )
    return stanford_config


def _build_stanford_colbert_index_config(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig,
):
    """Build the physical-index config without project search-only overrides."""
    snapshot_path = _validated_snapshot_path(resolved_checkpoint, config)
    values = _expected_stanford_values(snapshot_path, config)
    for field in (
        "query_maxlen", "ncells", "centroid_score_threshold", "ndocs"
    ):
        values.pop(field)
    return _stanford_config_class()(**values)


def initialize_colbert_checkpoint(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
    verbose: int = 0,
) -> InitializedColBERTCheckpoint:
    """Initialize only from the validated local immutable snapshot path."""
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    runtime_versions = _require_runtime_versions(config)
    snapshot_path = _validated_snapshot_path(resolved_checkpoint, config)
    stanford_config = _build_stanford_colbert_config(
        resolved_checkpoint,
        config=config,
    )
    checkpoint = _stanford_checkpoint_class()(
        str(snapshot_path),
        colbert_config=stanford_config,
        verbose=verbose,
    )
    if not hasattr(checkpoint, "colbert_config"):
        raise ValueError("initialized Checkpoint is missing colbert_config")
    effective_config = validate_effective_stanford_config(
        checkpoint.colbert_config,
        resolved_checkpoint,
        config=config,
    )
    return InitializedColBERTCheckpoint(
        checkpoint=checkpoint,
        resolved_checkpoint=resolved_checkpoint,
        effective_config=effective_config,
        **runtime_versions,
    )


class CanonicalColBERTRetriever:
    """Manifest-bound direct Stanford PLAID index/search adapter."""

    def __init__(
        self,
        *,
        resolved_checkpoint: ResolvedColBERTCheckpoint,
        checkpoint_provenance: ColBERTCheckpointPhysicalProvenance,
        index_root: Path,
        experiment: str = "sprint3_pubmedqa_colbertv2",
        config: ColBERTConfig = COLBERT_CONFIG,
    ) -> None:
        if not isinstance(checkpoint_provenance, ColBERTCheckpointPhysicalProvenance):
            raise TypeError("checkpoint_provenance must be physical provenance")
        _validated_snapshot_path(resolved_checkpoint, config)
        if (
            checkpoint_provenance.checkpoint_id != config.checkpoint_id
            or checkpoint_provenance.checkpoint_revision != config.checkpoint_revision
            or checkpoint_provenance.snapshot_path.resolve()
            != resolved_checkpoint.snapshot_path.resolve()
        ):
            raise ValueError("checkpoint physical provenance does not match resolution")
        self.resolved_checkpoint = resolved_checkpoint
        self.checkpoint_provenance = checkpoint_provenance
        self.index_root = Path(index_root).resolve()
        self.experiment = experiment
        self.config = config
        self.cache_identity: ColBERTCacheIdentity | None = None
        self.index_artifact_sha256: str | None = None
        self.pid_to_document_id: tuple[str | int, ...] = ()
        self._collection: tuple[str, ...] = ()
        self._searcher: Any = None

    @property
    def is_indexed(self) -> bool:
        return self._searcher is not None

    @property
    def index_directory(self) -> Path:
        if self.cache_identity is None:
            raise RuntimeError("ColBERT cache identity is not initialized")
        return (
            self.index_root / self.experiment / "indexes" /
            self.cache_identity.index_name
        )

    @property
    def completion_record_path(self) -> Path:
        """Project-owned sidecar kept outside Stanford's index directory."""
        if self.cache_identity is None:
            raise RuntimeError("ColBERT cache identity is not initialized")
        return self.index_directory.parent / (
            f".{self.cache_identity.index_name}.sprint3-complete.json"
        )

    def _completion_record_payload(self) -> dict[str, Any]:
        if self.cache_identity is None:
            raise RuntimeError("ColBERT cache identity is not initialized")
        return {
            "checkpoint_snapshot_manifest_sha256": (
                self.cache_identity.checkpoint_snapshot_manifest_sha256
            ),
            "corpus_manifest_sha256": self.cache_identity.corpus_manifest.sha256,
            "document_count": self.cache_identity.corpus_manifest.document_count,
            "index_fingerprint_sha256": self.cache_identity.fingerprint_sha256,
            "index_name": self.cache_identity.index_name,
            "schema_version": _INDEX_COMPLETION_SCHEMA_VERSION,
        }

    def _require_valid_completion_record(self) -> None:
        path = self.completion_record_path
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"invalid ColBERT completion record: {path}; explicit cleanup/"
                "rebuild is required"
            ) from error
        if payload != self._completion_record_payload():
            raise RuntimeError(
                f"ColBERT completion record identity mismatch: {path}; explicit "
                "cleanup/rebuild is required"
            )

    def _write_completion_record(self) -> None:
        """Publish an atomic, non-overwriting sidecar after successful indexing."""
        path = self.completion_record_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(
                    self._completion_record_payload(),
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def index_from_corpus_records(
        self,
        *,
        corpus_manifest: CorpusManifest,
        corpus_records: tuple[CorpusRecord, ...],
    ) -> None:
        collection, pid_map = colbert_runtime_collection_from_corpus_records(
            corpus_manifest=corpus_manifest,
            corpus_records=corpus_records,
        )
        identity = build_colbert_cache_identity(
            corpus_manifest=corpus_manifest,
            checkpoint_snapshot_manifest_sha256=(
                self.checkpoint_provenance.snapshot_manifest_sha256
            ),
            colbert_config=self.config,
        )
        if self.cache_identity is not None and self.cache_identity != identity:
            raise RuntimeError("ColBERT retriever cannot be rebound to another index")
        self.cache_identity = identity
        self._collection = collection
        self.pid_to_document_id = pid_map

        directory_exists = self.index_directory.exists()
        completion_exists = self.completion_record_path.exists()
        if directory_exists and not completion_exists:
            raise RuntimeError(
                f"incomplete ColBERT index: {self.index_directory}; explicit "
                "cleanup/rebuild is required"
            )
        if completion_exists:
            self._require_valid_completion_record()
            if not self.index_directory.is_dir():
                raise RuntimeError(
                    f"ColBERT completion record has no index directory: "
                    f"{self.index_directory}; explicit cleanup/rebuild is required"
                )

        # Required for both new indexing and completed-index Searcher reuse.
        require_colbert_runtime_prerequisites()

        search_config = build_stanford_colbert_config(
            self.resolved_checkpoint, config=self.config
        )
        index_config = _build_stanford_colbert_index_config(
            self.resolved_checkpoint, config=self.config
        )
        Run, RunConfig = _stanford_run_classes()
        run_config = RunConfig(
            nranks=self.config.nranks,
            amp=self.config.amp,
            experiment=self.experiment,
            root=str(self.index_root),
        )
        with Run().context(run_config):
            if not self.index_directory.exists():
                apply_colbert_index_seed(self.config.seed)
                indexer = _stanford_indexer_class()(
                    checkpoint=str(self.resolved_checkpoint.snapshot_path.resolve()),
                    config=index_config,
                )
                indexer.index(
                    name=identity.index_name,
                    collection=list(collection),
                    overwrite=False,
                )
                if not self.index_directory.is_dir():
                    raise RuntimeError(
                        "Stanford ColBERT did not create the expected index"
                    )
                self._write_completion_record()
            if not self.index_directory.is_dir():
                raise RuntimeError("Stanford ColBERT did not create the expected index")
            self.index_artifact_sha256 = fingerprint_colbert_index_directory(
                self.index_directory
            )
            self._searcher = _stanford_searcher_class()(
                index=identity.index_name,
                config=search_config,
                collection=list(collection),
            )

    def retrieve(self, query: str, *, top_k: int) -> tuple[RawCandidateResult, ...]:
        if not self.is_indexed:
            raise RuntimeError("ColBERT index must be initialized before retrieval")
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty exact string")
        if top_k != self.config.candidate_pool_size:
            raise ValueError("top_k must equal the frozen candidate pool size")
        pids, ranks, scores = self._searcher.search(query, k=top_k)
        if not (len(pids) == len(ranks) == len(scores) == top_k):
            raise ValueError("ColBERT returned a non-canonical candidate count")
        if tuple(int(rank) for rank in ranks) != tuple(range(1, top_k + 1)):
            raise ValueError("ColBERT returned non-contiguous native ranks")
        normalized_pids = tuple(int(pid) for pid in pids)
        if len(set(normalized_pids)) != top_k:
            raise ValueError("ColBERT returned duplicate PIDs")
        if any(pid < 0 or pid >= len(self.pid_to_document_id) for pid in normalized_pids):
            raise ValueError("ColBERT returned a PID outside the canonical corpus")
        numeric_scores = tuple(float(score) for score in scores)
        if any(not math.isfinite(score) for score in numeric_scores):
            raise ValueError("ColBERT returned a non-finite score")
        if any(left < right for left, right in zip(numeric_scores, numeric_scores[1:])):
            raise ValueError("ColBERT scores are not in native descending order")
        return tuple(
            RawCandidateResult(
                document_id=self.pid_to_document_id[pid],
                native_score=score,
            )
            for pid, score in zip(normalized_pids, numeric_scores)
        )
