"""Resolve the exact pinned ColBERT checkpoint to a local HF snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from retrievers.colbert_config import ColBERTConfig


@dataclass(frozen=True)
class ResolvedColBERTCheckpoint:
    """Physical local snapshot for one immutable scientific checkpoint."""

    checkpoint_id: str
    checkpoint_revision: str
    snapshot_path: Path


def resolve_colbert_checkpoint(
    config: ColBERTConfig,
    *,
    cache_dir: str | Path,
    local_files_only: bool,
) -> ResolvedColBERTCheckpoint:
    """Resolve exactly ``config.checkpoint_revision`` without any fallback."""
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    if not isinstance(cache_dir, (str, Path)) or not str(cache_dir).strip():
        raise ValueError("cache_dir must be a non-empty string or Path")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool")

    downloaded_path = snapshot_download(
        repo_id=config.checkpoint_id,
        revision=config.checkpoint_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if not isinstance(downloaded_path, (str, Path)):
        raise TypeError("snapshot_download must return a filesystem path")

    snapshot_path = Path(downloaded_path).expanduser().resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"ColBERT snapshot path does not exist: {snapshot_path}")
    if not snapshot_path.is_dir():
        raise NotADirectoryError(
            f"ColBERT snapshot path is not a directory: {snapshot_path}"
        )
    # This invariant depends on Hugging Face's normal cache_dir snapshot layout
    # without local_dir=; adding local_dir in future must revisit this validation.
    if snapshot_path.name != config.checkpoint_revision:
        raise ValueError(
            "resolved ColBERT snapshot revision does not match the configured "
            f"revision: expected {config.checkpoint_revision}, got "
            f"{snapshot_path.name}"
        )

    return ResolvedColBERTCheckpoint(
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=config.checkpoint_revision,
        snapshot_path=snapshot_path,
    )
