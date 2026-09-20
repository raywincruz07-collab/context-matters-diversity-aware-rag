"""Physical validation and provenance for a resolved ColBERT checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from retrievers.colbert_checkpoint import ResolvedColBERTCheckpoint
from retrievers.colbert_config import COLBERT_CONFIG, ColBERTConfig


_REQUIRED_FILES = frozenset(
    {
        "artifact.metadata",
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
)
_MODEL_METADATA_FIELDS = (
    "dim",
    "query_maxlen",
    "doc_maxlen",
    "similarity",
    "mask_punctuation",
    "attend_to_mask_tokens",
)
_HISTORICAL_INDEX_FIELDS = ("nbits", "kmeans_niters", "nranks")


@dataclass(frozen=True)
class ColBERTCheckpointFileProvenance:
    relative_path: str
    size_bytes: int
    sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ColBERTCheckpointMetadataProvenance:
    dim: int
    query_maxlen: int
    doc_maxlen: int
    similarity: str
    mask_punctuation: bool
    attend_to_mask_tokens: bool
    tokenizer_do_lower_case: bool
    historical_nbits: int | None
    historical_kmeans_niters: int | None
    historical_nranks: int | None

    def payload(self) -> dict[str, Any]:
        return {
            "attend_to_mask_tokens": self.attend_to_mask_tokens,
            "dim": self.dim,
            "doc_maxlen": self.doc_maxlen,
            "historical_kmeans_niters": self.historical_kmeans_niters,
            "historical_nbits": self.historical_nbits,
            "historical_nranks": self.historical_nranks,
            "mask_punctuation": self.mask_punctuation,
            "query_maxlen": self.query_maxlen,
            "similarity": self.similarity,
            "tokenizer_do_lower_case": self.tokenizer_do_lower_case,
        }


@dataclass(frozen=True)
class ColBERTCheckpointPhysicalProvenance:
    checkpoint_id: str
    checkpoint_revision: str
    snapshot_path: Path
    files: tuple[ColBERTCheckpointFileProvenance, ...]
    file_count: int
    total_bytes: int
    snapshot_manifest_sha256: str
    metadata: ColBERTCheckpointMetadataProvenance

    def payload(self) -> dict[str, Any]:
        """Return physical/runtime provenance, including the machine-local path."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_revision": self.checkpoint_revision,
            "file_count": self.file_count,
            "files": [file.payload() for file in self.files],
            "metadata": self.metadata.payload(),
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "snapshot_path": str(self.snapshot_path),
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class ColBERTWeightStructureProvenance:
    tensor_count: int
    projection_name: str
    projection_shape: tuple[int, ...]
    word_embeddings_shape: tuple[int, ...]
    position_embeddings_shape: tuple[int, ...]
    dtype: str

    def payload(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "position_embeddings_shape": list(self.position_embeddings_shape),
            "projection_name": self.projection_name,
            "projection_shape": list(self.projection_shape),
            "tensor_count": self.tensor_count,
            "word_embeddings_shape": list(self.word_embeddings_shape),
        }


def _physical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return value


def _artifact_config(metadata: dict[str, Any]) -> dict[str, Any]:
    nested = metadata.get("config")
    if nested is None:
        return metadata
    if not isinstance(nested, dict):
        raise ValueError("artifact.metadata config must be a JSON object")
    return nested


def _require_exact_metadata(field: str, actual: Any, expected: Any) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"artifact.metadata {field} does not match ColBERTConfig: "
            f"expected {expected!r}, got {actual!r}"
        )


def _historical_value(metadata: Mapping[str, Any], field: str) -> int | None:
    value = metadata.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"artifact.metadata historical {field} must be an integer")
    return value


def build_colbert_checkpoint_physical_provenance(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
) -> ColBERTCheckpointPhysicalProvenance:
    """Hash and validate one concrete immutable ColBERT snapshot."""
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

    paths = sorted(
        (path for path in snapshot_path.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot_path).as_posix(),
    )
    relative_paths = tuple(path.relative_to(snapshot_path).as_posix() for path in paths)
    missing = sorted(_REQUIRED_FILES.difference(relative_paths))
    if missing:
        raise ValueError(f"ColBERT snapshot is missing required files: {missing}")

    files = tuple(
        ColBERTCheckpointFileProvenance(
            relative_path=relative_path,
            size_bytes=path.stat().st_size,
            sha256=_physical_sha256(path),
        )
        for path, relative_path in zip(paths, relative_paths, strict=True)
    )
    manifest = "".join(
        f"{file.sha256}  {file.size_bytes}  {file.relative_path}\n"
        for file in files
    )
    manifest_sha256 = hashlib.sha256(manifest.encode("utf-8")).hexdigest()

    raw_metadata = _load_json_object(
        snapshot_path / "artifact.metadata", "artifact.metadata"
    )
    model_metadata = _artifact_config(raw_metadata)
    for field in _MODEL_METADATA_FIELDS:
        _require_exact_metadata(field, model_metadata.get(field), getattr(config, field))

    tokenizer_config = _load_json_object(
        snapshot_path / "tokenizer_config.json", "tokenizer_config.json"
    )
    if tokenizer_config.get("do_lower_case") is not True:
        raise ValueError("tokenizer_config.json do_lower_case must be true")

    metadata = ColBERTCheckpointMetadataProvenance(
        dim=model_metadata["dim"],
        query_maxlen=model_metadata["query_maxlen"],
        doc_maxlen=model_metadata["doc_maxlen"],
        similarity=model_metadata["similarity"],
        mask_punctuation=model_metadata["mask_punctuation"],
        attend_to_mask_tokens=model_metadata["attend_to_mask_tokens"],
        tokenizer_do_lower_case=True,
        historical_nbits=_historical_value(model_metadata, "nbits"),
        historical_kmeans_niters=_historical_value(
            model_metadata, "kmeans_niters"
        ),
        historical_nranks=_historical_value(model_metadata, "nranks"),
    )
    return ColBERTCheckpointPhysicalProvenance(
        checkpoint_id=resolved_checkpoint.checkpoint_id,
        checkpoint_revision=resolved_checkpoint.checkpoint_revision,
        snapshot_path=snapshot_path,
        files=files,
        file_count=len(files),
        total_bytes=sum(file.size_bytes for file in files),
        snapshot_manifest_sha256=manifest_sha256,
        metadata=metadata,
    )


def validate_colbert_checkpoint_weight_structure(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
) -> ColBERTWeightStructureProvenance:
    """Optionally load weights on CPU and validate the learned tensor structure."""
    if not isinstance(resolved_checkpoint, ResolvedColBERTCheckpoint):
        raise TypeError("resolved_checkpoint must be a ResolvedColBERTCheckpoint")
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")

    import torch

    state_dict = torch.load(
        resolved_checkpoint.snapshot_path / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, Mapping):
        raise ValueError("pytorch_model.bin must contain a state-dict mapping")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("pytorch_model.bin state dict must contain only tensors")

    projection_shape = (config.dim, 768)
    projection_candidates = tuple(
        name for name, tensor in state_dict.items() if tuple(tensor.shape) == projection_shape
    )
    if len(projection_candidates) != 1:
        raise ValueError(
            "pytorch_model.bin must contain exactly one 128x768 projection tensor"
        )
    if projection_candidates[0] != "linear.weight":
        raise ValueError("ColBERT projection tensor must be linear.weight")

    expected = {
        "linear.weight": projection_shape,
        "bert.embeddings.word_embeddings.weight": (30522, 768),
        "bert.embeddings.position_embeddings.weight": (512, 768),
    }
    for name, shape in expected.items():
        tensor = state_dict.get(name)
        if tensor is None or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype float32")

    return ColBERTWeightStructureProvenance(
        tensor_count=len(state_dict),
        projection_name="linear.weight",
        projection_shape=projection_shape,
        word_embeddings_shape=expected["bert.embeddings.word_embeddings.weight"],
        position_embeddings_shape=expected[
            "bert.embeddings.position_embeddings.weight"
        ],
        dtype="float32",
    )
