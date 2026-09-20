"""Offline data loading and secret-free execution provenance for generation CLIs."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from generation._io import file_sha256, stable_json_sha256
from generation.maki import CanonicalMakiAdapter, MakiConfig, PRIMARY_LLM_LOGICAL_IDS
from retrieval_artifacts.colbert_cache_identity import (
    fingerprint_colbert_index_directory,
)
from run_registry import (
    append_run_record,
    artifact_ref,
    read_registry,
    run_identity_payload,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_REGISTRY_PATH = Path("artifacts/run_registry/run_registry_v1.jsonl")
_CANONICAL_EVIDENCE_AUTHORITY_PATH = Path(
    "artifacts/run_registry/evidence_manifest_authority_v1.json"
)
_PUBMEDQA_GENERATION_ROOT = Path("results/sprint3/raw/pubmedqa/generation")
_PUBMEDQA_GENERATION_RETRIEVERS = frozenset(
    {"bm25", "dpr", "contriever", "colbertv2"}
)
_GENERATION_SAMPLE_NAME_RE = re.compile(r"sample_([0-9]{4})\.json")
_RUN_ID_RE = re.compile(r"run-[a-z0-9][a-z0-9-]*-[0-9a-f]{24}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def neural_index_artifact_ref(
    *,
    repository_root: Path,
    retriever: str,
    artifact_path: Path,
    index_fingerprint_sha256: str,
    index_artifact_sha256: str,
) -> dict[str, Any]:
    """Reference one governed neural index with retriever-specific integrity."""
    if retriever not in {"dpr", "contriever", "colbertv2"}:
        raise ValueError(f"unsupported neural retriever: {retriever}")
    for value, name in (
        (index_fingerprint_sha256, "index_fingerprint_sha256"),
        (index_artifact_sha256, "index_artifact_sha256"),
    ):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")

    artifact_id = f"index:sha256:{index_fingerprint_sha256}"
    if retriever in {"dpr", "contriever"}:
        reference = artifact_ref(
            artifact_path,
            repository_root=repository_root,
            artifact_id=artifact_id,
        )
        if reference["sha256"] != index_artifact_sha256:
            raise ValueError(
                "neural index physical SHA does not match "
                "candidate index_artifact_sha256"
            )
        return reference

    root = Path(repository_root).resolve()
    resolved = Path(artifact_path).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "neural index artifact path is outside repository_root"
        ) from exc
    if not resolved.is_dir():
        raise ValueError(f"ColBERT index artifact must be a directory: {relative}")
    physical_sha256 = fingerprint_colbert_index_directory(resolved)
    if physical_sha256 != index_artifact_sha256:
        raise ValueError(
            "ColBERT index directory fingerprint does not match "
            "candidate index_artifact_sha256"
        )
    return {
        "path": relative,
        "sha256": physical_sha256,
        "artifact_id": artifact_id,
    }


def load_pubmedqa_runtime_local_only(*, cache_dir: Path | None = None):
    """Load the pinned dataset from an existing cache with network disabled."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    from scripts.build_corpus_manifests import (
        PUBMEDQA_CONFIG,
        PUBMEDQA_REVISION,
        PUBMEDQA_SOURCE,
        PUBMEDQA_SPLIT,
        load_validated_pubmedqa_runtime_corpus,
    )

    rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if cache_dir is None else str(cache_dir),
        download_mode="reuse_dataset_if_exists",
    )
    return load_validated_pubmedqa_runtime_corpus(tuple(rows))


def git_registry_identity(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not bool(status)
    diff_sha = None
    if not clean:
        tracked = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(tracked)
        for raw_path in sorted(part for part in untracked.split(b"\0") if part):
            digest.update(raw_path)
            digest.update(b"\0")
            path = repository_root / raw_path.decode("utf-8")
            if path.is_file():
                digest.update(path.read_bytes())
            digest.update(b"\0")
        diff_sha = digest.hexdigest()
    return {
        "commit": commit,
        "branch": branch,
        "worktree_clean": clean,
        "worktree_diff_sha256": diff_sha,
    }


def require_pubmedqa_generation_replacement_parent(
    records: Sequence[Mapping[str, Any]], parent_run_id: str
) -> Mapping[str, Any]:
    """Return an eligible terminal parent without changing registry evidence."""
    matching = [record for record in records if record["run_id"] == parent_run_id]
    if not matching:
        raise ValueError("replacement parent run is absent from the governed registry")
    parent = matching[-1]
    if (
        parent["run_type"] != "GENERATION"
        or parent["data"]["dataset"] != "pubmedqa"
    ):
        raise ValueError("replacement parent must be a PubMedQA GENERATION run")
    if parent["execution"]["status"] not in {"COMPLETE", "FAILED"}:
        raise ValueError("replacement parent must be terminal")
    if parent["output"]["failed_row_count"] <= 0:
        raise ValueError("replacement parent must contain failed/error rows")
    return parent


def pubmedqa_generation_replacement_output_directory(
    *, parent_run_id: str, replacement_git_commit: str
) -> Path:
    """Derive the only governed output directory for one replacement run."""
    if _RUN_ID_RE.fullmatch(parent_run_id) is None:
        raise ValueError("replacement parent run ID is invalid")
    if _GIT_SHA_RE.fullmatch(replacement_git_commit) is None:
        raise ValueError("replacement Git commit must be a full lowercase SHA")
    return (
        _PUBMEDQA_GENERATION_ROOT
        / "replacements"
        / parent_run_id
        / replacement_git_commit
    )


def validate_pubmedqa_generation_replacement(
    records: Sequence[Mapping[str, Any]], replacement: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate parent lineage, scientific identity, and the replacement path."""
    if (
        replacement["run_type"] != "GENERATION"
        or replacement["data"]["dataset"] != "pubmedqa"
        or replacement["execution"]["status"] != "PLANNED"
    ):
        raise ValueError("replacement must be a planned PubMedQA GENERATION run")
    parent_run_id = replacement["execution"]["parent_run_id"]
    if parent_run_id is None:
        raise ValueError("replacement must record execution.parent_run_id")
    parent = require_pubmedqa_generation_replacement_parent(records, parent_run_id)
    if replacement["git"]["commit"] == parent["git"]["commit"]:
        raise ValueError("replacement must use the later amendment Git commit")
    if replacement["run_id"] == parent["run_id"]:
        raise ValueError("replacement must have a distinct governed run ID")

    parent_identity = run_identity_payload(parent)
    replacement_identity = run_identity_payload(replacement)
    parent_identity.pop("git")
    replacement_identity.pop("git")
    if replacement_identity != parent_identity:
        raise ValueError("replacement scientific identity differs from its parent block")

    expected_output_directory = pubmedqa_generation_replacement_output_directory(
        parent_run_id=parent_run_id,
        replacement_git_commit=replacement["git"]["commit"],
    )
    if Path(replacement["output"]["output_directory"]) != expected_output_directory:
        raise ValueError("replacement output directory is not canonical")
    return parent


def _validated_pubmedqa_generation_output_directories(
    *, registry_path: Path, evidence_authority_path: Path
) -> frozenset[Path] | None:
    try:
        records = read_registry(
            registry_path,
            evidence_authority_path=evidence_authority_path,
        )
        with tempfile.TemporaryDirectory(
            prefix="context-matters-registry-validation-"
        ) as temporary_directory:
            replay_path = Path(temporary_directory) / "registry.jsonl"
            for record in records:
                if not append_run_record(
                    replay_path,
                    record,
                    evidence_authority_path=evidence_authority_path,
                ):
                    return None
    except (OSError, UnicodeError, ValueError):
        return None

    result: set[Path] = set()
    first_record_indexes: dict[str, int] = {}
    for index, record in enumerate(records):
        first_record_indexes.setdefault(record["run_id"], index)
    for index, record in enumerate(records):
        if (
            record["run_type"] != "GENERATION"
            or record["data"]["dataset"] != "pubmedqa"
            or first_record_indexes[record["run_id"]] != index
        ):
            continue
        parent_run_id = record["execution"]["parent_run_id"]
        if parent_run_id is not None:
            try:
                validate_pubmedqa_generation_replacement(records[:index], record)
            except (KeyError, TypeError, ValueError):
                continue
            result.add(Path(record["output"]["output_directory"]))
            continue
        logical_model_id = record["generation"]["llm_logical_id"]
        if logical_model_id not in PRIMARY_LLM_LOGICAL_IDS:
            continue
        context_mode = record["context_mode"]
        retrieval = record["retrieval"]
        if context_mode == "without_context" and retrieval is None:
            expected_output_directory = (
                _PUBMEDQA_GENERATION_ROOT
                / "without_context"
                / logical_model_id
            )
        elif context_mode == "with_context" and retrieval is not None:
            retriever = retrieval["retriever"]
            if retriever not in _PUBMEDQA_GENERATION_RETRIEVERS:
                continue
            expected_output_directory = (
                _PUBMEDQA_GENERATION_ROOT
                / "with_context"
                / retriever
                / logical_model_id
            )
        else:
            continue
        output_directory = Path(record["output"]["output_directory"])
        if output_directory == expected_output_directory:
            result.add(output_directory)
    return frozenset(result)


def _is_registered_generation_output(
    path: Path, *, allowed_output_directories: frozenset[Path]
) -> bool:
    if path.parent not in allowed_output_directories:
        return False
    if path.name == "generation_output_inventory_v1.json":
        return True
    match = _GENERATION_SAMPLE_NAME_RE.fullmatch(path.name)
    return match is not None and int(match.group(1)) <= 999


def canonical_generation_git_identity(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Return source identity while excluding validated canonical run evidence.

    This specialization accepts only append-only changes to the canonical run
    registry and untracked PubMedQA generation outputs whose directories have
    registry lineage. All other changes retain :func:`git_registry_identity`'s
    strict dirty-worktree result.
    """
    repository_root = Path(repository_root).resolve()
    registry_path = repository_root / _CANONICAL_REGISTRY_PATH
    evidence_authority_path = repository_root / _CANONICAL_EVIDENCE_AUTHORITY_PATH

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ),
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    if not status:
        return {
            "commit": commit,
            "branch": branch,
            "worktree_clean": True,
            "worktree_diff_sha256": None,
        }

    registry_modified = False
    untracked_paths: list[Path] = []
    for entry in status.split(b"\0"):
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            return git_registry_identity(repository_root)
        state = entry[:2]
        raw_path = entry[3:]
        path = Path(os.fsdecode(raw_path))
        if path.is_absolute() or ".." in path.parts:
            return git_registry_identity(repository_root)
        if path == _CANONICAL_REGISTRY_PATH and state == b" M":
            registry_modified = True
        elif state == b"??":
            untracked_paths.append(path)
        else:
            # This also rejects staged changes, tracked output modifications,
            # renames/copies, and every non-runtime repository path.
            return git_registry_identity(repository_root)

    if registry_modified:
        committed_registry = subprocess.run(
            ("git", "show", f"HEAD:{_CANONICAL_REGISTRY_PATH.as_posix()}"),
            cwd=repository_root,
            check=False,
            capture_output=True,
        )
        if committed_registry.returncode != 0:
            return git_registry_identity(repository_root)
        try:
            current_registry = registry_path.read_bytes()
        except OSError:
            return git_registry_identity(repository_root)
        if (
            len(current_registry) <= len(committed_registry.stdout)
            or not current_registry.startswith(committed_registry.stdout)
        ):
            return git_registry_identity(repository_root)

    allowed_output_directories = _validated_pubmedqa_generation_output_directories(
        registry_path=registry_path,
        evidence_authority_path=evidence_authority_path,
    )
    if allowed_output_directories is None:
        return git_registry_identity(repository_root)
    for path in untracked_paths:
        absolute_path = repository_root / path
        if (
            absolute_path.is_symlink()
            or not absolute_path.is_file()
            or not _is_registered_generation_output(
                path,
                allowed_output_directories=allowed_output_directories,
            )
        ):
            return git_registry_identity(repository_root)

    return {
        "commit": commit,
        "branch": branch,
        "worktree_clean": True,
        "worktree_diff_sha256": None,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "requests": _package_version("requests"),
        "generation_package_files": {
            path.name: file_sha256(path)
            for path in sorted((REPOSITORY_ROOT / "src/generation").glob("*.py"))
        },
    }


def load_model_bindings(path: Path) -> dict[str, MakiConfig]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping) or set(stored) != {"models"}:
        raise ValueError("model bindings must contain exactly a models object")
    models = stored["models"]
    if not isinstance(models, Mapping) or set(models) != set(PRIMARY_LLM_LOGICAL_IDS):
        raise ValueError("model bindings require exactly the three primary logical IDs")
    result: dict[str, MakiConfig] = {}
    for logical_id in PRIMARY_LLM_LOGICAL_IDS:
        value = models[logical_id]
        if not isinstance(value, Mapping):
            raise ValueError("each model binding must be an object")
        result[logical_id] = MakiConfig(
            base_url=value["base_url"],
            logical_model_id=logical_id,
            physical_model_id=value["physical_model_id"],
            model_revision=value.get("model_revision"),
            model_revision_kind=value["model_revision_kind"],
            direct_mode_status=value["direct_mode_status"],
            direct_mode_control=value["direct_mode_control"],
            seed=value.get("seed"),
            timeout_seconds=value.get("timeout_seconds", 300.0),
        )
    return result


def adapter_from_bindings(
    bindings: Mapping[str, MakiConfig],
    logical_id: str,
    *,
    max_tokens: int = 256,
) -> CanonicalMakiAdapter:
    return CanonicalMakiAdapter(
        bindings[logical_id],
        max_tokens=max_tokens,
    )


def runtime_provenance(adapter: CanonicalMakiAdapter) -> dict[str, Any]:
    return {
        "runtime_identity": adapter.config.runtime_identity(),
        "environment_variable_names": ["MAKI_API_KEY"],
        "secret_values_persisted": False,
    }


def provenance_hashes(adapter: CanonicalMakiAdapter) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    environment = environment_provenance()
    runtime = runtime_provenance(adapter)
    return environment, runtime, stable_json_sha256(environment), stable_json_sha256(runtime)
