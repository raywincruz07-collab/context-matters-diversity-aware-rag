"""Structured, non-aggregating query-level evaluation for Sprint 3.

Public API misuse (unknown metric IDs, wrong metric scope, or wrong input object)
raises. Scientific input failures and metric-computation failures are isolated in
``MetricResult`` values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from evaluation import (
    exact_match as legacy_exact_match,
    mrr as legacy_mrr,
    retrieval_diversity as legacy_retrieval_diversity,
    retrieval_recall_at_k as legacy_recall_at_k,
    rouge_l as legacy_rouge_l,
    token_f1 as legacy_token_f1,
)
from evaluation.contracts import (
    ComponentStatus,
    ContextMode,
    MetricResult,
    MetricStatus,
    ReasonCode,
)
from evaluation.metric_registry import (
    METRIC_REGISTRY,
    ApplicabilityContext,
    DatasetId,
    MetricScope,
)


DEFAULT_RETRIEVAL_METRICS = (
    "recall_at_k",
    "mrr_at_k",
    "retrieval_diversity",
)
DEFAULT_GENERATION_METRICS = ("exact_match", "token_f1", "rouge_l")

_IMPLEMENTED_RETRIEVAL_METRICS = frozenset(DEFAULT_RETRIEVAL_METRICS)
_IMPLEMENTED_GENERATION_METRICS = frozenset(DEFAULT_GENERATION_METRICS)


@dataclass(frozen=True)
class RetrievalEvaluationInput:
    """Inputs for metrics belonging to one frozen retrieval artifact.

    Document-ID sequences are converted to tuples to preserve order and prevent
    list mutation. The embedding array is retained by reference to avoid a large
    copy; freezing this dataclass does not make the NumPy buffer immutable.
    """

    dataset_id: DatasetId
    context_mode: ContextMode
    retrieved_doc_ids: tuple[Any, ...] | Sequence[Any]
    gold_doc_ids: tuple[Any, ...] | Sequence[Any] | None
    selected_embeddings: np.ndarray | None
    evaluator_available: bool
    protocol_frozen: bool

    def __post_init__(self) -> None:
        _validate_common_fields(
            self.dataset_id,
            self.context_mode,
            self.evaluator_available,
            self.protocol_frozen,
        )
        object.__setattr__(
            self,
            "retrieved_doc_ids",
            _ordered_tuple(self.retrieved_doc_ids, "retrieved_doc_ids"),
        )
        if self.gold_doc_ids is not None:
            object.__setattr__(
                self,
                "gold_doc_ids",
                _ordered_tuple(self.gold_doc_ids, "gold_doc_ids"),
            )


@dataclass(frozen=True)
class GenerationEvaluationInput:
    """Inputs for metrics belonging to one generated answer."""

    dataset_id: DatasetId
    context_mode: ContextMode
    generation_status: ComponentStatus
    prediction: str | None
    reference_answer: str | None
    reference_available: bool
    evaluator_available: bool
    protocol_frozen: bool

    def __post_init__(self) -> None:
        _validate_common_fields(
            self.dataset_id,
            self.context_mode,
            self.evaluator_available,
            self.protocol_frozen,
        )
        if not isinstance(self.generation_status, ComponentStatus):
            raise TypeError("generation_status must be a ComponentStatus")
        if type(self.reference_available) is not bool:
            raise TypeError("reference_available must be a bool")


def evaluate_retrieval_metric(
    metric_id: str, inputs: RetrievalEvaluationInput
) -> MetricResult:
    """Evaluate one retrieval-artifact metric without aggregation."""
    definition = METRIC_REGISTRY.get(metric_id)
    if definition.scope is not MetricScope.RETRIEVAL_ARTIFACT:
        raise ValueError(f"metric {metric_id!r} is not a retrieval-artifact metric")
    if not isinstance(inputs, RetrievalEvaluationInput):
        raise TypeError("inputs must be a RetrievalEvaluationInput")

    decision = METRIC_REGISTRY.decide(
        metric_id,
        ApplicabilityContext(
            dataset_id=inputs.dataset_id,
            context_mode=inputs.context_mode,
            generation_status=ComponentStatus.SUCCESS,
            reference_available=inputs.gold_doc_ids is not None,
            evaluator_available=inputs.evaluator_available,
            protocol_frozen=inputs.protocol_frozen,
        ),
    )
    if not decision.eligible:
        return _decision_result(decision.status, decision.reason)
    if metric_id not in _IMPLEMENTED_RETRIEVAL_METRICS:
        return _unavailable(
            MetricStatus.NOT_COMPUTED, ReasonCode.EVALUATOR_NOT_CONFIGURED
        )

    if metric_id in {"recall_at_k", "mrr_at_k"}:
        invalid = _validate_qrels(inputs.retrieved_doc_ids, inputs.gold_doc_ids)
        if invalid is not None:
            return invalid
        function = legacy_recall_at_k if metric_id == "recall_at_k" else legacy_mrr
        return _compute_metric(
            function, list(inputs.retrieved_doc_ids), list(inputs.gold_doc_ids)
        )

    embeddings, invalid = _validate_embeddings(
        inputs.selected_embeddings, len(inputs.retrieved_doc_ids)
    )
    if invalid is not None:
        return invalid
    return _compute_metric(legacy_retrieval_diversity, embeddings)


def evaluate_generation_metric(
    metric_id: str, inputs: GenerationEvaluationInput
) -> MetricResult:
    """Evaluate one generation or generation-context metric without aggregation."""
    definition = METRIC_REGISTRY.get(metric_id)
    if definition.scope not in {
        MetricScope.GENERATION,
        MetricScope.GENERATION_CONTEXT,
    }:
        raise ValueError(f"metric {metric_id!r} is not a generation metric")
    if not isinstance(inputs, GenerationEvaluationInput):
        raise TypeError("inputs must be a GenerationEvaluationInput")

    decision = METRIC_REGISTRY.decide(
        metric_id,
        ApplicabilityContext(
            dataset_id=inputs.dataset_id,
            context_mode=inputs.context_mode,
            generation_status=inputs.generation_status,
            reference_available=inputs.reference_available,
            evaluator_available=inputs.evaluator_available,
            protocol_frozen=inputs.protocol_frozen,
        ),
    )
    if not decision.eligible:
        return _decision_result(decision.status, decision.reason)
    if metric_id not in _IMPLEMENTED_GENERATION_METRICS:
        return _unavailable(
            MetricStatus.NOT_COMPUTED, ReasonCode.EVALUATOR_NOT_CONFIGURED
        )
    if not isinstance(inputs.prediction, str):
        return _unavailable(MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)
    if not isinstance(inputs.reference_answer, str):
        return _unavailable(MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)

    function = {
        "exact_match": legacy_exact_match,
        "token_f1": legacy_token_f1,
        "rouge_l": legacy_rouge_l,
    }[metric_id]
    return _compute_metric(function, inputs.prediction, inputs.reference_answer)


def evaluate_retrieval_metrics(
    inputs: RetrievalEvaluationInput,
    metric_ids: Sequence[str] = DEFAULT_RETRIEVAL_METRICS,
) -> dict[str, MetricResult]:
    """Evaluate retrieval metrics independently; no aggregation is performed."""
    return {
        metric_id: evaluate_retrieval_metric(metric_id, inputs)
        for metric_id in metric_ids
    }


def evaluate_generation_metrics(
    inputs: GenerationEvaluationInput,
    metric_ids: Sequence[str] = DEFAULT_GENERATION_METRICS,
) -> dict[str, MetricResult]:
    """Evaluate generation metrics independently; no aggregation is performed."""
    return {
        metric_id: evaluate_generation_metric(metric_id, inputs)
        for metric_id in metric_ids
    }


def _validate_common_fields(
    dataset_id: DatasetId,
    context_mode: ContextMode,
    evaluator_available: bool,
    protocol_frozen: bool,
) -> None:
    if not isinstance(dataset_id, DatasetId):
        raise TypeError("dataset_id must be a DatasetId")
    if not isinstance(context_mode, ContextMode):
        raise TypeError("context_mode must be a ContextMode")
    if type(evaluator_available) is not bool:
        raise TypeError("evaluator_available must be a bool")
    if type(protocol_frozen) is not bool:
        raise TypeError("protocol_frozen must be a bool")


def _ordered_tuple(values: Sequence[Any], field_name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be an ordered sequence")
    return tuple(values)


def _valid_doc_id(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, Integral):
        return True
    return isinstance(value, str) and bool(value.strip())


def _validate_qrels(
    retrieved_doc_ids: tuple[Any, ...], gold_doc_ids: tuple[Any, ...] | None
) -> MetricResult | None:
    if gold_doc_ids is None:
        return _unavailable(
            MetricStatus.NOT_COMPUTED, ReasonCode.REFERENCE_UNAVAILABLE
        )
    if not gold_doc_ids:
        return _unavailable(MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)
    if not all(_valid_doc_id(doc_id) for doc_id in retrieved_doc_ids):
        return _unavailable(MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)
    if not all(_valid_doc_id(doc_id) for doc_id in gold_doc_ids):
        return _unavailable(MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)
    if len(set(gold_doc_ids)) != len(gold_doc_ids):
        return _unavailable(MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)
    return None


def _validate_embeddings(
    selected_embeddings: np.ndarray | None,
    expected_rows: int,
) -> tuple[np.ndarray | None, MetricResult | None]:
    if selected_embeddings is None:
        return None, _unavailable(
            MetricStatus.NOT_COMPUTED, ReasonCode.MISSING_EMBEDDING
        )
    try:
        embeddings = np.asarray(selected_embeddings)
    except Exception:
        return None, _unavailable(
            MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT
        )
    if embeddings.ndim != 2 or embeddings.shape[1] <= 0:
        return None, _unavailable(
            MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT
        )
    if embeddings.shape[0] != expected_rows:
        return None, _unavailable(
            MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT
        )
    if embeddings.shape[0] < 2:
        return None, _unavailable(
            MetricStatus.NOT_COMPUTED, ReasonCode.INSUFFICIENT_ITEMS
        )
    if embeddings.dtype.kind not in {"i", "u", "f"} or not np.isfinite(
        embeddings
    ).all():
        return None, _unavailable(
            MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT
        )
    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        return None, _unavailable(
            MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT
        )
    return embeddings, None


def _compute_metric(function, *args) -> MetricResult:
    try:
        value = function(*args)
    except Exception:
        return _unavailable(MetricStatus.FAILED, ReasonCode.METRIC_EXCEPTION)
    try:
        return MetricResult(value=value, status=MetricStatus.MEASURED)
    except (TypeError, ValueError):
        return _unavailable(MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)


def _decision_result(
    status: MetricStatus | None, reason: ReasonCode | None
) -> MetricResult:
    if status is None or reason is None:
        raise RuntimeError("an ineligible registry decision must have status and reason")
    return MetricResult(value=None, status=status, reason=reason)


def _unavailable(status: MetricStatus, reason: ReasonCode) -> MetricResult:
    return MetricResult(value=None, status=status, reason=reason)
