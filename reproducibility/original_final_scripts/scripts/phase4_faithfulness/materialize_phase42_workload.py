#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from build_phase42_workload import (
    FINAL8,
    extract_pubmed_explanation,
    normalize_and_split,
    split_document_block,
    verify_passage_parity,
)


MODELS = {
    "llama-3.3-70b",
    "gemma4-26b",
    "ministral-3-14b",
}


class IndexedJsonl:
    def __init__(self, path: Path):
        self.path = path

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.offsets = {}

        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()

                if not line:
                    break

                if not line.strip():
                    continue

                row = json.loads(line)
                position = row.get("position")

                if not isinstance(position, int):
                    raise RuntimeError(
                        f"{self.path}: row without integer position"
                    )

                if position in self.offsets:
                    raise RuntimeError(
                        f"{self.path}: duplicate position {position}"
                    )

                self.offsets[position] = offset

        self.handle = self.path.open("rb")

    def get(self, position: int):
        if position not in self.offsets:
            raise RuntimeError(
                f"{self.path}: position {position} not found"
            )

        self.handle.seek(
            self.offsets[position]
        )

        line = self.handle.readline()

        return json.loads(line)

    def close(self):
        self.handle.close()


class JsonlCache:
    def __init__(self):
        self.cache = {}

    def get_row(self, path: Path, position: int):
        key = str(path)

        if key not in self.cache:
            self.cache[key] = IndexedJsonl(path)

        return self.cache[key].get(position)

    def close(self):
        for reader in self.cache.values():
            reader.close()


def relative_to_root(path: Path, root: Path):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def validate_record(record):
    required = [
        "dataset",
        "retriever",
        "condition",
        "logical_model_id",
        "physical_model_id",
        "position",
        "sample_id",
        "status",
        "claims",
        "passages",
        "source_generation_path",
    ]

    missing = [
        key
        for key in required
        if key not in record
    ]

    if missing:
        raise RuntimeError(
            f"Missing workload fields: {missing}"
        )

    if record["condition"] not in FINAL8:
        raise RuntimeError(
            f"Unexpected condition: {record['condition']}"
        )

    if record["logical_model_id"] not in MODELS:
        raise RuntimeError(
            f"Unexpected logical model: "
            f"{record['logical_model_id']}"
        )

    if (
        record["logical_model_id"]
        == "ministral-3-14b"
        and record["physical_model_id"]
        != "qwen3.6-36b"
    ):
        raise RuntimeError(
            "ministral-3-14b physical binding mismatch: "
            f"{record['physical_model_id']}"
        )

    if len(record["passages"]) != 5:
        raise RuntimeError(
            f"{record['dataset']}: expected 5 passages, "
            f"got {len(record['passages'])}"
        )

    if any(
        not isinstance(p, str)
        or not p.strip()
        for p in record["passages"]
    ):
        raise RuntimeError(
            f"{record['dataset']}: empty passage"
        )

    if not record["claims"]:
        raise RuntimeError(
            f"{record['dataset']}: zero claims"
        )

    if any(
        not isinstance(c, str)
        or not c.strip()
        for c in record["claims"]
    ):
        raise RuntimeError(
            f"{record['dataset']}: empty claim"
        )

    if record["dataset"] == "asqa":
        if record["retriever"] == "colbertv2":
            raise RuntimeError(
                "ASQA ColBERT is outside final design"
            )

    if record["dataset"] == "hotpotqa":
        if len(record["claims"]) != 1:
            raise RuntimeError(
                "HotpotQA must have exactly one hypothesis"
            )

    return record


def iter_pubmedqa(root: Path):
    parquet_path = (
        root
        / "results/sprint2/pubmedqa/"
          "correctness_20260925/"
          "pubmedqa_correctness_per_generation.parquet"
    )

    generation_root = (
        root
        / "data/pubmedqa/sprint2/"
          "generation_outputs_2026-09-25"
    )

    pf = pq.ParquetFile(parquet_path)

    columns = [
        "sample_id",
        "retriever",
        "condition",
        "physical_model_id",
        "status",
        "generation_relative_path",
    ]

    for batch in pf.iter_batches(
        batch_size=4096,
        columns=columns,
    ):
        for row in batch.to_pylist():
            if row["status"] != "OK":
                continue

            rel = Path(
                row["generation_relative_path"]
            )

            parts = rel.parts

            if len(parts) < 4:
                raise RuntimeError(
                    f"Unexpected PubMedQA relative path: {rel}"
                )

            logical_model_id = parts[2]

            gen_path = (
                generation_root
                / rel
            )

            obj = json.loads(
                gen_path.read_text(
                    encoding="utf-8"
                )
            )

            raw_content = (
                obj["observation"]["raw_content"]
            )

            context_block = (
                obj["request"]["prompt"]["context_block"]
            )

            claims = extract_pubmed_explanation(
                raw_content
            )

            passages = split_document_block(
                context_block
            )

            position = int(
                gen_path.stem.split("_", 1)[1]
            )

            record = {
                "dataset": "pubmedqa",
                "retriever": row["retriever"],
                "condition": row["condition"],
                "logical_model_id": logical_model_id,
                "physical_model_id": row["physical_model_id"],
                "position": position,
                "sample_id": row["sample_id"],
                "status": row["status"],
                "claims": claims,
                "passages": passages,
                "passage_ids": None,
                "source_generation_path": (
                    relative_to_root(
                        gen_path,
                        root,
                    )
                ),
                "source_context_path": None,
            }

            yield validate_record(record)


def hotpot_generation_root(
    root: Path,
    symbolic_root: str,
):
    mapping = {
        "refresh": (
            root
            / "data/hotpotqa/sprint2/"
              "baseline_refresh_outputs_2026-09-24"
        ),
        "fresh": (
            root
            / "data/hotpotqa/sprint2/"
              "generation_outputs_2026-09-23"
        ),
    }

    if symbolic_root not in mapping:
        raise RuntimeError(
            f"Unknown HotpotQA source_root: "
            f"{symbolic_root}"
        )

    return mapping[symbolic_root]


def iter_hotpotqa(
    root: Path,
    cache: JsonlCache,
):
    parquet_path = (
        root
        / "results/sprint2/hotpotqa/"
          "correctness_20260926/"
          "per_generation.parquet"
    )

    package_root = (
        root
        / "data/hotpotqa/sprint2/"
          "generation_package_2026-09-22"
    )

    pf = pq.ParquetFile(parquet_path)

    columns = [
        "sample_id",
        "retriever",
        "condition",
        "logical_model_id",
        "status",
        "measurable",
        "prediction",
        "source_root",
        "source_relative_path",
    ]

    for batch in pf.iter_batches(
        batch_size=4096,
        columns=columns,
    ):
        for row in batch.to_pylist():
            if not row["measurable"]:
                continue

            generation_root = (
                hotpot_generation_root(
                    root,
                    row["source_root"],
                )
            )

            gen_path = (
                generation_root
                / row["source_relative_path"]
            )

            obj = json.loads(
                gen_path.read_text(
                    encoding="utf-8"
                )
            )

            position = obj["position"]

            source_input = (
                package_root
                / obj["source_input_relative_path"]
            )

            source_row = cache.get_row(
                source_input,
                position,
            )

            passages = verify_passage_parity(
                obj,
                source_row,
            )

            question = obj["query_text"].strip()

            prediction = row["prediction"]

            if (
                prediction is not None
                and str(prediction).strip()
            ):
                answer = str(prediction).strip()
                answer_source = "correctness_prediction"

            elif (
                row["status"] == "REFUSAL"
                and isinstance(
                    obj.get("raw_content"),
                    str,
                )
                and obj["raw_content"].strip()
            ):
                answer = obj["raw_content"].strip()
                answer_source = "verbatim_refusal_raw_content"

            else:
                raise RuntimeError(
                    "HotpotQA measurable row has no usable "
                    "prediction and is not a valid REFUSAL fallback"
                )

            if not question or not answer:
                raise RuntimeError(
                    "HotpotQA empty question/answer"
                )

            claim = (
                f"The answer to {question} "
                f"is {answer}."
            )

            record = {
                "dataset": "hotpotqa",
                "retriever": row["retriever"],
                "condition": row["condition"],
                "logical_model_id": row["logical_model_id"],
                "physical_model_id": obj["physical_model_id"],
                "position": position,
                "sample_id": row["sample_id"],
                "query_id": obj.get("query_id"),
                "status": row["status"],
                "hotpot_answer_source": answer_source,
                "claims": [claim],
                "passages": passages,
                "passage_ids": [
                    str(x)
                    for x in obj["passage_ids"]
                ],
                "source_generation_path": (
                    relative_to_root(
                        gen_path,
                        root,
                    )
                ),
                "source_context_path": (
                    relative_to_root(
                        source_input,
                        root,
                    )
                ),
            }

            yield validate_record(record)


def asqa_source_input(
    root: Path,
    condition: str,
    retriever: str,
):
    if condition == "none":
        return (
            root
            / "data/asqa/"
              "generation_package_2026-09-09"
            / f"{retriever}_with_context.jsonl"
        )

    return (
        root
        / "data/asqa/sprint2/"
          "generation_package_2026-09-22"
        / retriever
        / f"{condition}.jsonl"
    )


def iter_asqa(
    root: Path,
    cache: JsonlCache,
):
    freeze = (
        root
        / "exports/"
          "asqa_sprint2_correctness_input_freeze_20260926/"
          "asqa_final_72cell_outputs.jsonl"
    )

    with freeze.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if not row["measurable"]:
                continue

            gen_path = (
                root
                / row["source_relative_path"]
            )

            obj = json.loads(
                gen_path.read_text(
                    encoding="utf-8"
                )
            )

            position = obj["position"]

            source_input = asqa_source_input(
                root,
                row["condition"],
                row["retriever"],
            )

            source_row = cache.get_row(
                source_input,
                position,
            )

            passages = verify_passage_parity(
                obj,
                source_row,
            )

            claims = normalize_and_split(
                row["raw_answer"]
            )

            if not claims:
                raise RuntimeError(
                    "ASQA answer yielded zero claims"
                )

            record = {
                "dataset": "asqa",
                "retriever": row["retriever"],
                "condition": row["condition"],
                "logical_model_id": row["logical_model_id"],
                "physical_model_id": obj["physical_model_id"],
                "position": position,
                "sample_id": row["sample_id"],
                "query_id": obj.get("query_id"),
                "status": row["status"],
                "claims": claims,
                "passages": passages,
                "passage_ids": [
                    str(x)
                    for x in obj["passage_ids"]
                ],
                "source_generation_path": (
                    relative_to_root(
                        gen_path,
                        root,
                    )
                ),
                "source_context_path": (
                    relative_to_root(
                        source_input,
                        root,
                    )
                ),
            }

            yield validate_record(record)


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def materialize_dataset(
    name,
    iterator,
    out_path,
):
    answer_count = 0
    claim_count = 0
    pair_count = 0

    by_condition = Counter()
    by_retriever = Counter()
    by_logical_model = Counter()
    by_physical_model = Counter()
    by_status = Counter()

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as out:
        for record in iterator:
            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            out.write("\n")

            n_claims = len(record["claims"])

            answer_count += 1
            claim_count += n_claims
            pair_count += (
                n_claims
                * len(record["passages"])
            )

            by_condition[
                record["condition"]
            ] += 1

            by_retriever[
                record["retriever"]
            ] += 1

            by_logical_model[
                record["logical_model_id"]
            ] += 1

            by_physical_model[
                record["physical_model_id"]
            ] += 1

            by_status[
                record["status"]
            ] += 1

    return {
        "dataset": name,
        "answers": answer_count,
        "claims": claim_count,
        "nli_pairs": pair_count,
        "by_condition": dict(
            sorted(by_condition.items())
        ),
        "by_retriever": dict(
            sorted(by_retriever.items())
        ),
        "by_logical_model": dict(
            sorted(by_logical_model.items())
        ),
        "by_physical_model": dict(
            sorted(by_physical_model.items())
        ),
        "by_status": dict(
            sorted(by_status.items())
        ),
        "file": out_path.name,
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
    }


def validate_iterator(
    name,
    iterator,
    limit,
):
    count = 0
    claims = 0
    pairs = 0
    first = None

    for record in iterator:
        if first is None:
            first = record

        count += 1
        claims += len(record["claims"])
        pairs += (
            len(record["claims"])
            * 5
        )

        if count >= limit:
            break

    if count != limit:
        raise RuntimeError(
            f"{name}: requested {limit} validation "
            f"rows but obtained {count}"
        )

    print(f"\n=== {name.upper()} VALIDATION ===")
    print("answers =", count)
    print("claims =", claims)
    print("nli_pairs =", pairs)

    print(
        "first_identity =",
        {
            key: first.get(key)
            for key in [
                "retriever",
                "condition",
                "logical_model_id",
                "physical_model_id",
                "position",
                "sample_id",
                "status",
            ]
        },
    )

    print(
        "first_claim =",
        repr(first["claims"][0][:500]),
    )

    print(
        "first_passage =",
        repr(first["passages"][0][:500]),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
    )

    args = parser.parse_args()

    cache = JsonlCache()

    try:
        if args.validate_only:
            validate_iterator(
                "pubmedqa",
                iter_pubmedqa(args.root),
                args.limit,
            )

            validate_iterator(
                "hotpotqa",
                iter_hotpotqa(
                    args.root,
                    cache,
                ),
                args.limit,
            )

            validate_iterator(
                "asqa",
                iter_asqa(
                    args.root,
                    cache,
                ),
                args.limit,
            )

            print(
                "\nPHASE_4_2_WORKLOAD_MATERIALIZER_VALIDATION=PASS"
            )
            print("FILES_WRITTEN=NO")
            print("MODEL_INFERENCE_RUN=NO")
            print("GPU_WORK_STARTED=NO")
            return

        if args.out_dir is None:
            raise RuntimeError(
                "--out-dir is required unless "
                "--validate-only is used"
            )

        if args.out_dir.exists():
            existing = list(
                args.out_dir.iterdir()
            )

            if existing:
                raise RuntimeError(
                    f"Refusing non-empty output directory: "
                    f"{args.out_dir}"
                )
        else:
            args.out_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

        summaries = []

        summaries.append(
            materialize_dataset(
                "pubmedqa",
                iter_pubmedqa(args.root),
                args.out_dir
                / "pubmedqa_phase42_workload.jsonl",
            )
        )

        summaries.append(
            materialize_dataset(
                "hotpotqa",
                iter_hotpotqa(
                    args.root,
                    cache,
                ),
                args.out_dir
                / "hotpotqa_phase42_workload.jsonl",
            )
        )

        summaries.append(
            materialize_dataset(
                "asqa",
                iter_asqa(
                    args.root,
                    cache,
                ),
                args.out_dir
                / "asqa_phase42_workload.jsonl",
            )
        )

        manifest = {
            "artifact_format": (
                "context-matters.phase42-faithfulness-workload.v1"
            ),
            "date": "2026-09-26",
            "protocol": (
                "FAITHFULNESS_PROTOCOL_AMENDMENT_05_"
                "SIMPLIFIED_KICKOFF_SCOPE.md"
            ),
            "preprocessing_contract": (
                "FAITHFULNESS_PREPROCESSING_CONTRACT_06.md"
            ),
            "segmentation_amendment": (
                "FAITHFULNESS_PREPROCESSING_AMENDMENT_07_"
                "SENTENCE_SEGMENTATION.md"
            ),
            "datasets": summaries,
            "total_answers": sum(
                x["answers"]
                for x in summaries
            ),
            "total_claims": sum(
                x["claims"]
                for x in summaries
            ),
            "total_nli_pairs": sum(
                x["nli_pairs"]
                for x in summaries
            ),
            "model_inference_run": False,
            "generation_reruns": 0,
        }

        manifest_path = (
            args.out_dir
            / "manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        print(
            "\nPHASE_4_2_WORKLOAD_MATERIALIZATION=COMPLETE"
        )
        print("MODEL_INFERENCE_RUN=NO")
        print("GPU_WORK_STARTED=NO")

    finally:
        cache.close()


if __name__ == "__main__":
    main()
