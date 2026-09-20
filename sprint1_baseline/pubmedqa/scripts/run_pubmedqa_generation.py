#!/usr/bin/env python3
"""Run, resume, inspect, or enumerate governed PubMedQA generation blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
for value in (REPOSITORY_ROOT, SRC_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from generation._io import stable_json_sha256
from generation.cli_support import (
    adapter_from_bindings,
    canonical_generation_git_identity,
    git_registry_identity,
    load_model_bindings,
    load_pubmedqa_runtime_local_only,
    neural_index_artifact_ref,
    provenance_hashes,
    pubmedqa_generation_replacement_output_directory,
    require_pubmedqa_generation_replacement_parent,
    validate_pubmedqa_generation_replacement,
)
from generation.runner import (
    GenerationBlock,
    build_generation_planned_record,
    canonical_pubmedqa_generation_matrix,
    execute_generation_block,
    expected_matrix_row_count,
    inspect_generation_block,
    utc_now,
)
from generation.selected_context import read_selected_context_set
from retrieval_artifacts import read_candidate_artifact
from run_registry import (
    DEFAULT_EVIDENCE_AUTHORITY_PATH,
    DEFAULT_REGISTRY_PATH,
    artifact_ref,
    candidate_set_artifact_ref,
    read_registry,
)


CONFIRMATION = "I_UNDERSTAND_THIS_MAKES_MODEL_REQUESTS"


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm", required=True)
    parser.add_argument("--context-mode", choices=("with_context", "without_context"), required=True)
    parser.add_argument("--retriever", choices=("bm25", "dpr", "contriever", "colbertv2"))
    parser.add_argument("--model-bindings", type=Path, required=True)
    parser.add_argument("--repeatability-gate", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--candidate-directory", type=Path)
    parser.add_argument("--candidate-set", type=Path)
    parser.add_argument("--selected-context-directory", type=Path)
    parser.add_argument("--selected-context-set", type=Path)
    parser.add_argument("--index-artifact", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--output-inventory", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--evidence-authority",
        type=Path,
        default=DEFAULT_EVIDENCE_AUTHORITY_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "replace", "resume", "status"):
        child = subparsers.add_parser(action)
        _common_parser(child)
        if action in {"run", "replace", "resume"}:
            child.add_argument("--confirm-api-calls", required=True)
        if action == "replace":
            child.add_argument("--parent-run-id", required=True)
        if action in {"resume", "status"}:
            child.add_argument("--run-id", required=True)
    subparsers.add_parser("matrix")
    return parser.parse_args()


def _relative(path: Path) -> str:
    return Path(path).resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _default_output(block: GenerationBlock) -> Path:
    root = REPOSITORY_ROOT / "results/sprint3/raw/pubmedqa/generation"
    if block.context_mode == "without_context":
        return root / "without_context" / block.llm_logical_id
    return root / "with_context" / str(block.retriever) / block.llm_logical_id


def _registered_planned(args: argparse.Namespace):
    if args.action not in {"resume", "status"}:
        return None
    records = [
        record
        for record in read_registry(
            args.registry, evidence_authority_path=args.evidence_authority
        )
        if record["run_id"] == args.run_id
    ]
    if not records:
        raise ValueError("--run-id is absent from the governed registry")
    if records[0]["execution"]["status"] != "PLANNED":
        raise ValueError("registered run does not begin with PLANNED")
    return records[0]


def _build_inputs(args: argparse.Namespace, *, registered_planned=None):
    bindings = load_model_bindings(args.model_bindings)
    if args.llm not in bindings:
        raise ValueError("--llm must be one of the three frozen primary IDs")
    block = GenerationBlock(args.context_mode, args.llm, args.retriever)
    adapter = adapter_from_bindings(bindings, args.llm)
    runtime = load_pubmedqa_runtime_local_only(cache_dir=args.cache_dir)
    uses_canonical_generation_identity = (
        args.action in {"run", "replace"}
        and registered_planned is None
        and args.registry.resolve() == DEFAULT_REGISTRY_PATH.resolve()
        and (
            args.evidence_authority.resolve()
            == DEFAULT_EVIDENCE_AUTHORITY_PATH.resolve()
        )
    )
    git = (
        canonical_generation_git_identity()
        if uses_canonical_generation_identity
        else git_registry_identity()
    )
    if registered_planned is None and not git["worktree_clean"]:
        raise RuntimeError("canonical generation requires a clean worktree")
    sample_manifest_path = REPOSITORY_ROOT / "artifacts/sample_manifests/pubmedqa_sample_manifest_v2.json"
    corpus_manifest_path = REPOSITORY_ROOT / "artifacts/corpus_manifests/pubmedqa_corpus_manifest_v1.json"
    sample_ref = artifact_ref(
        sample_manifest_path,
        repository_root=REPOSITORY_ROOT,
        artifact_id=runtime.sample_manifest.manifest_id,
    )
    corpus_ref = retrieval = None
    if block.context_mode == "with_context":
        required = {
            "candidate_directory": args.candidate_directory,
            "candidate_set": args.candidate_set,
            "selected_context_directory": args.selected_context_directory,
            "selected_context_set": args.selected_context_set,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"WITH_CONTEXT missing required arguments: {missing}")
        corpus_ref = artifact_ref(
            corpus_manifest_path,
            repository_root=REPOSITORY_ROOT,
            artifact_id=runtime.corpus_manifest.corpus_manifest_id,
        )
        first = read_candidate_artifact(args.candidate_directory / "sample_0000.json")
        if first.retriever.retriever_name != block.retriever:
            raise ValueError("candidate directory retriever differs from block")
        candidate_ref = candidate_set_artifact_ref(
            args.candidate_set, repository_root=REPOSITORY_ROOT
        )
        context_set = read_selected_context_set(args.selected_context_set)
        selected_ref = artifact_ref(
            args.selected_context_set,
            repository_root=REPOSITORY_ROOT,
            artifact_id=context_set["selected_context_set_id"],
        )
        if candidate_ref["artifact_id"] != context_set["scientific_payload"]["candidate_set_id"]:
            raise ValueError("candidate and selected-context aggregate identities differ")
        if block.retriever == "bm25":
            if args.index_artifact is not None:
                raise ValueError("BM25 must not carry a neural index artifact")
            index_ref = None
        else:
            if args.index_artifact is None:
                raise ValueError("neural generation lineage requires --index-artifact")
            index_ref = neural_index_artifact_ref(
                repository_root=REPOSITORY_ROOT,
                retriever=block.retriever,
                artifact_path=args.index_artifact,
                index_fingerprint_sha256=(
                    first.retriever.index_fingerprint_sha256
                ),
                index_artifact_sha256=first.retriever.index_artifact_sha256,
            )
        retrieval = {
            "retriever": block.retriever,
            "config_sha256": stable_json_sha256(first.scientific_payload()["retriever"]),
            "index": index_ref,
            "candidate_set": candidate_ref,
            "selected_context": selected_ref,
            "candidate_pool": 20,
            "top_k": 5,
        }
    elif any(
        value is not None
        for value in (
            args.candidate_directory,
            args.candidate_set,
            args.selected_context_directory,
            args.selected_context_set,
            args.index_artifact,
        )
    ):
        raise ValueError("WITHOUT_CONTEXT forbids retrieval/context arguments")
    environment, runtime_provenance, environment_sha, runtime_sha = provenance_hashes(adapter)
    registered_output = (
        None
        if registered_planned is None
        else REPOSITORY_ROOT / registered_planned["output"]["output_directory"]
    )
    replacement_records = ()
    parent_run_id = None
    if args.action == "replace":
        if args.output_directory is not None or args.output_inventory is not None:
            raise ValueError("replacement output paths are derived and cannot be overridden")
        replacement_records = read_registry(
            args.registry,
            evidence_authority_path=args.evidence_authority,
        )
        parent = require_pubmedqa_generation_replacement_parent(
            replacement_records,
            args.parent_run_id,
        )
        if any(
            record["execution"]["parent_run_id"] == parent["run_id"]
            for record in replacement_records
        ):
            raise ValueError(
                "a replacement is already registered for this parent; use resume/status"
            )
        parent_run_id = parent["run_id"]
        output_directory = REPOSITORY_ROOT / (
            pubmedqa_generation_replacement_output_directory(
                parent_run_id=parent_run_id,
                replacement_git_commit=git["commit"],
            )
        )
        output_inventory = output_directory / "generation_output_inventory_v1.json"
    else:
        output_directory = (
            args.output_directory or registered_output or _default_output(block)
        )
        output_inventory = (
            args.output_inventory
            or output_directory / "generation_output_inventory_v1.json"
        )
    planned = build_generation_planned_record(
        created_at=utc_now(),
        block=block,
        adapter=adapter,
        git=git,
        sample_manifest_ref=sample_ref,
        corpus_manifest_ref=corpus_ref,
        retrieval=retrieval,
        environment_sha256=environment_sha,
        runtime_sha256=runtime_sha,
        hardware_summary=f"platform={platform.platform()}",
        output_directory=_relative(output_directory),
        parent_run_id=parent_run_id,
        evidence_authority_path=args.evidence_authority,
    )
    if args.action == "replace":
        validate_pubmedqa_generation_replacement(replacement_records, planned)
    if registered_planned is not None:
        for field in (
            "sprint",
            "stage",
            "run_type",
            "evidence_role",
            "origin",
            "data",
            "retrieval",
            "diversification",
            "context_mode",
            "generation",
        ):
            if planned[field] != registered_planned[field]:
                raise ValueError(
                    f"current execution inputs differ from registered {field} identity"
                )
        if _relative(output_directory) != registered_planned["output"]["output_directory"]:
            raise ValueError("resume/status output directory differs from registered run")
        planned = registered_planned
    return (
        bindings,
        block,
        adapter,
        runtime,
        planned,
        environment,
        runtime_provenance,
        output_directory,
        output_inventory,
    )


def main() -> None:
    args = parse_args()
    if args.action == "matrix":
        blocks = [
            {
                "context_mode": block.context_mode,
                "retriever": block.retriever,
                "llm": block.llm_logical_id,
                "expected_rows": 1000,
            }
            for block in canonical_pubmedqa_generation_matrix()
        ]
        print(json.dumps({"blocks": blocks, "total_rows": expected_matrix_row_count()}, indent=2))
        return
    registered_planned = _registered_planned(args)
    (
        bindings,
        block,
        adapter,
        runtime,
        planned,
        environment,
        runtime_provenance,
        output_directory,
        output_inventory,
    ) = _build_inputs(args, registered_planned=registered_planned)
    if args.action == "status":
        print(
            json.dumps(
                inspect_generation_block(
                    planned_record=planned,
                    output_directory=output_directory,
                    registry_path=args.registry,
                    evidence_authority_path=args.evidence_authority,
                ),
                indent=2,
            )
        )
        return
    if args.confirm_api_calls != CONFIRMATION:
        raise SystemExit(f"refusing API execution; pass --confirm-api-calls {CONFIRMATION}")
    terminal = execute_generation_block(
        planned_record=planned,
        runtime=runtime,
        adapter=adapter,
        repeatability_gate_path=args.repeatability_gate,
        all_model_runtime_identities={
            logical_id: config.runtime_identity()
            for logical_id, config in bindings.items()
        },
        output_directory=output_directory,
        output_inventory_path=output_inventory,
        environment=environment,
        runtime_provenance=runtime_provenance,
        hardware_summary=f"platform={platform.platform()}",
        selected_context_directory=args.selected_context_directory,
        selected_context_set_path=args.selected_context_set,
        registry_path=args.registry,
        evidence_authority_path=args.evidence_authority,
        repository_root=REPOSITORY_ROOT,
    )
    print(terminal["run_id"])
    print(terminal["execution"]["status"])


if __name__ == "__main__":
    main()
