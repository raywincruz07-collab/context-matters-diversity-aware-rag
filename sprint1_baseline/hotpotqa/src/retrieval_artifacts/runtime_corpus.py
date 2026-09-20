"""Validated adapters from scientific corpus records to retriever inputs."""

from __future__ import annotations

from typing import Any

from retrieval_artifacts.corpus_manifest import CorpusManifest
from retrieval_artifacts.producer import (
    CorpusRecord,
    validate_corpus_records_against_manifest,
)


def bm25_runtime_documents_from_corpus_records(
    *,
    corpus_manifest: CorpusManifest,
    corpus_records: tuple[CorpusRecord, ...],
) -> list[dict[str, Any]]:
    """Map validated records to BM25 input without changing identity or content.

    The generic BM25 runtime calls its indexed field ``text``. Scientifically,
    that value is the exact prepared ``CorpusRecord.retrieval_content`` rather
    than the record's stored ``text`` field.
    """
    validate_corpus_records_against_manifest(corpus_manifest, corpus_records)
    return [
        {
            "doc_id": record.document_id,
            "text": record.retrieval_content,
        }
        for record in corpus_records
    ]


def dpr_runtime_documents_from_corpus_records(
    *,
    corpus_manifest: CorpusManifest,
    corpus_records: tuple[CorpusRecord, ...],
) -> list[dict[str, Any]]:
    """Map validated records to strict DPR input without changing content."""
    validate_corpus_records_against_manifest(corpus_manifest, corpus_records)
    return [
        {
            "doc_id": record.document_id,
            "retrieval_content": record.retrieval_content,
        }
        for record in corpus_records
    ]


def contriever_runtime_documents_from_corpus_records(
    *,
    corpus_manifest: CorpusManifest,
    corpus_records: tuple[CorpusRecord, ...],
) -> list[dict[str, Any]]:
    """Map validated records to strict Contriever input without changing content."""
    validate_corpus_records_against_manifest(corpus_manifest, corpus_records)
    return [
        {
            "doc_id": record.document_id,
            "retrieval_content": record.retrieval_content,
        }
        for record in corpus_records
    ]


def colbert_runtime_collection_from_corpus_records(
    *,
    corpus_manifest: CorpusManifest,
    corpus_records: tuple[CorpusRecord, ...],
) -> tuple[tuple[str, ...], tuple[str | int, ...]]:
    """Return exact collection text and the authoritative PID-to-ID mapping."""
    validate_corpus_records_against_manifest(corpus_manifest, corpus_records)
    collection = tuple(record.retrieval_content for record in corpus_records)
    pid_to_document_id = tuple(record.document_id for record in corpus_records)
    if len(collection) != corpus_manifest.document_count:
        raise ValueError("ColBERT collection count does not match CorpusManifest")
    return collection, pid_to_document_id
