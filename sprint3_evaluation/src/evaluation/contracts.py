"""Validated data contracts for Sprint 3 evaluation results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

import numpy as np


SCHEMA_VERSION = "sprint3.eval.v1"


class MetricStatus(str, Enum):
    """Availability and computation status of one metric value."""

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    NOT_COMPUTED = "not_computed"
    FAILED = "failed"


class ComponentStatus(str, Enum):
    """Execution status of a pipeline component, distinct from metric status."""

    SUCCESS = "success"
    NOT_RUN = "not_run"
    FAILED = "failed"


class ReasonCode(str, Enum):
    """Controlled explanations for unavailable metric values."""

    REFERENCE_UNAVAILABLE = "reference_unavailable"
    PROTOCOL_NOT_FROZEN = "protocol_not_frozen"
    EVALUATOR_NOT_CONFIGURED = "evaluator_not_configured"
    EVALUATOR_UNAVAILABLE = "evaluator_unavailable"
    GENERATION_NOT_ATTEMPTED = "generation_not_attempted"
    GENERATION_FAILED = "generation_failed"
    RETRIEVAL_FAILED = "retrieval_failed"
    INSUFFICIENT_ITEMS = "insufficient_items"
    MISSING_EMBEDDING = "missing_embedding"
    INVALID_METRIC_INPUT = "invalid_metric_input"
    METRIC_EXCEPTION = "metric_exception"
    NOT_DEFINED_FOR_CONTEXT_MODE = "not_defined_for_context_mode"
    NOT_DEFINED_FOR_DATASET = "not_defined_for_dataset"


class ContextMode(str, Enum):
    """Whether retrieved evidence is supplied to the generator."""

    WITH_CONTEXT = "with_context"
    WITHOUT_CONTEXT = "without_context"


@dataclass(frozen=True)
class MetricResult:
    """One metric value with explicit availability and reason metadata."""

    value: Real | None
    status: MetricStatus
    reason: ReasonCode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, MetricStatus):
            raise TypeError("status must be a MetricStatus")

        if self.status is MetricStatus.MEASURED:
            if isinstance(self.value, (bool, np.bool_)) or not isinstance(
                self.value, Real
            ):
                raise TypeError("a measured metric value must be a real number")
            if not math.isfinite(float(self.value)):
                raise ValueError("a measured metric value must be finite")
            if self.reason is not None:
                raise ValueError("a measured metric result must not have a reason")
            return

        if self.value is not None:
            raise ValueError("a non-measured metric result must have value=None")
        if not isinstance(self.reason, ReasonCode):
            raise ValueError(
                "a non-measured metric result must have a controlled reason code"
            )


@dataclass(frozen=True)
class RowIdentity:
    """Minimum identity required to distinguish a Sprint 3 evaluation row."""

    schema_version: str
    run_id: str
    dataset_id: str
    dataset_split: str
    sample_id: str
    retrieval_artifact_id: str | None
    model_provider: str
    model_id: str
    model_revision: str | None
    context_mode: ContextMode
    generation_replica: int

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")

        required_strings = (
            "run_id",
            "dataset_id",
            "dataset_split",
            "sample_id",
            "model_provider",
            "model_id",
        )
        for field_name in required_strings:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ValueError("model_revision must be None or a non-empty string")
        if not isinstance(self.context_mode, ContextMode):
            raise TypeError("context_mode must be a ContextMode")
        if self.context_mode is ContextMode.WITH_CONTEXT:
            if (
                not isinstance(self.retrieval_artifact_id, str)
                or not self.retrieval_artifact_id.strip()
            ):
                raise ValueError(
                    "with_context identity requires a non-empty retrieval_artifact_id"
                )
        elif self.retrieval_artifact_id is not None:
            raise ValueError(
                "without_context identity requires retrieval_artifact_id=None"
            )
        if (
            isinstance(self.generation_replica, (bool, np.bool_))
            or not isinstance(self.generation_replica, Integral)
            or self.generation_replica < 0
        ):
            raise ValueError(
                "generation_replica must be a non-boolean integer >= 0"
            )

    def logical_key(self) -> tuple[Any, ...]:
        """Return a deterministic tuple suitable for later duplicate detection."""
        return (
            self.schema_version,
            self.run_id,
            self.dataset_id,
            self.dataset_split,
            self.sample_id,
            self.retrieval_artifact_id,
            self.model_provider,
            self.model_id,
            self.model_revision,
            self.context_mode.value,
            int(self.generation_replica),
        )


def metric_result_to_json(result: MetricResult) -> dict[str, Any]:
    """Return a JSON-compatible value/status/reason mapping."""
    return {
        "value": float(result.value) if result.value is not None else None,
        "status": result.status.value,
        "reason": result.reason.value if result.reason is not None else None,
    }


def metric_result_to_csv(result: MetricResult) -> dict[str, Any]:
    """Return a flat CSV projection with NaN for unavailable numeric values."""
    return {
        "value": float(result.value) if result.value is not None else math.nan,
        "status": result.status.value,
        "reason": result.reason.value if result.reason is not None else None,
    }
