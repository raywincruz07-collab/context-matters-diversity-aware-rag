"""Declarative metric applicability registry for Sprint 3 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from evaluation.contracts import (
    ComponentStatus,
    ContextMode,
    MetricStatus,
    ReasonCode,
)


class MetricScope(str, Enum):
    """Scientific object to which a metric belongs."""

    RETRIEVAL_ARTIFACT = "retrieval_artifact"
    GENERATION = "generation"
    GENERATION_CONTEXT = "generation_context"


class DatasetId(str, Enum):
    """Canonical dataset identifiers supported by the Sprint 3 registry."""

    PUBMEDQA = "pubmedqa"
    HOTPOTQA = "hotpotqa"
    ASQA = "asqa"


@dataclass(frozen=True)
class ApplicabilityContext:
    """Runtime facts used to decide whether a metric may be computed."""

    dataset_id: DatasetId
    context_mode: ContextMode
    generation_status: ComponentStatus
    reference_available: bool
    evaluator_available: bool
    protocol_frozen: bool

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be a DatasetId")
        if not isinstance(self.context_mode, ContextMode):
            raise TypeError("context_mode must be a ContextMode")
        if not isinstance(self.generation_status, ComponentStatus):
            raise TypeError("generation_status must be a ComponentStatus")
        for field_name in (
            "reference_available",
            "evaluator_available",
            "protocol_frozen",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")


@dataclass(frozen=True)
class ApplicabilityDecision:
    """Eligibility or an explicit non-measured status; never a metric value."""

    eligible: bool
    status: MetricStatus | None = None
    reason: ReasonCode | None = None

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be a bool")
        if self.eligible:
            if self.status is not None or self.reason is not None:
                raise ValueError("an eligible decision has no status or reason")
            return
        if not isinstance(self.status, MetricStatus):
            raise TypeError("an ineligible decision requires a MetricStatus")
        if self.status is MetricStatus.MEASURED:
            raise ValueError("applicability cannot mark a metric as measured")
        if not isinstance(self.reason, ReasonCode):
            raise TypeError("an ineligible decision requires a ReasonCode")


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable scientific and operational requirements for one metric."""

    metric_id: str
    version: str
    scope: MetricScope
    allowed_datasets: frozenset[DatasetId]
    allowed_context_modes: frozenset[ContextMode]
    requires_generation: bool
    requires_reference: bool
    requires_evaluator: bool
    protocol_frozen_datasets: frozenset[DatasetId]

    def __post_init__(self) -> None:
        if not isinstance(self.metric_id, str) or not self.metric_id.strip():
            raise ValueError("metric_id must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if not isinstance(self.scope, MetricScope):
            raise TypeError("scope must be a MetricScope")
        for field_name, enum_type in (
            ("allowed_datasets", DatasetId),
            ("allowed_context_modes", ContextMode),
            ("protocol_frozen_datasets", DatasetId),
        ):
            values = getattr(self, field_name)
            if not isinstance(values, frozenset) or not all(
                isinstance(value, enum_type) for value in values
            ):
                raise TypeError(f"{field_name} must be a frozenset of {enum_type.__name__}")
        if not self.allowed_datasets:
            raise ValueError("allowed_datasets must not be empty")
        if not self.allowed_context_modes:
            raise ValueError("allowed_context_modes must not be empty")
        if not self.protocol_frozen_datasets <= self.allowed_datasets:
            raise ValueError("frozen protocol datasets must also be allowed")
        for field_name in (
            "requires_generation",
            "requires_reference",
            "requires_evaluator",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")


@dataclass(frozen=True, init=False)
class MetricRegistry:
    """Immutable collection of metric definitions with strict lookup."""

    _definitions: Mapping[str, MetricDefinition]

    def __init__(self, definitions: Iterable[MetricDefinition]) -> None:
        by_id: dict[str, MetricDefinition] = {}
        for definition in definitions:
            if not isinstance(definition, MetricDefinition):
                raise TypeError("registry entries must be MetricDefinition objects")
            if definition.metric_id in by_id:
                raise ValueError(f"duplicate metric_id: {definition.metric_id!r}")
            by_id[definition.metric_id] = definition
        object.__setattr__(self, "_definitions", MappingProxyType(by_id))

    @property
    def definitions(self) -> Mapping[str, MetricDefinition]:
        return self._definitions

    def get(self, metric_id: str) -> MetricDefinition:
        try:
            return self._definitions[metric_id]
        except KeyError as exc:
            raise KeyError(f"unknown metric_id: {metric_id!r}") from exc

    def decide(
        self, metric_id: str, context: ApplicabilityContext
    ) -> ApplicabilityDecision:
        definition = self.get(metric_id)
        if not isinstance(context, ApplicabilityContext):
            raise TypeError("context must be an ApplicabilityContext")

        if context.dataset_id not in definition.allowed_datasets:
            return _unavailable(
                MetricStatus.NOT_APPLICABLE, ReasonCode.NOT_DEFINED_FOR_DATASET
            )
        if context.context_mode not in definition.allowed_context_modes:
            return _unavailable(
                MetricStatus.NOT_APPLICABLE,
                ReasonCode.NOT_DEFINED_FOR_CONTEXT_MODE,
            )
        if (
            context.dataset_id not in definition.protocol_frozen_datasets
            or not context.protocol_frozen
        ):
            return _unavailable(
                MetricStatus.NOT_COMPUTED, ReasonCode.PROTOCOL_NOT_FROZEN
            )
        if definition.requires_reference and not context.reference_available:
            return _unavailable(
                MetricStatus.NOT_COMPUTED, ReasonCode.REFERENCE_UNAVAILABLE
            )
        if definition.requires_evaluator and not context.evaluator_available:
            return _unavailable(
                MetricStatus.NOT_COMPUTED, ReasonCode.EVALUATOR_UNAVAILABLE
            )
        if definition.requires_generation:
            if context.generation_status is ComponentStatus.NOT_RUN:
                return _unavailable(
                    MetricStatus.NOT_COMPUTED,
                    ReasonCode.GENERATION_NOT_ATTEMPTED,
                )
            if context.generation_status is ComponentStatus.FAILED:
                return _unavailable(
                    MetricStatus.FAILED, ReasonCode.GENERATION_FAILED
                )
        return ApplicabilityDecision(eligible=True)


def _unavailable(status: MetricStatus, reason: ReasonCode) -> ApplicabilityDecision:
    return ApplicabilityDecision(eligible=False, status=status, reason=reason)


_BOTH_CONTEXT_MODES = frozenset(ContextMode)
_ALL_DATASETS = frozenset(DatasetId)
_PUBMED_HOTPOT = frozenset({DatasetId.PUBMEDQA, DatasetId.HOTPOTQA})
_PUBMED_ONLY = frozenset({DatasetId.PUBMEDQA})
_ASQA_ONLY = frozenset({DatasetId.ASQA})


def _definition(
    metric_id: str,
    version: str,
    scope: MetricScope,
    allowed_datasets: frozenset[DatasetId],
    *,
    allowed_context_modes: frozenset[ContextMode] = _BOTH_CONTEXT_MODES,
    requires_generation: bool = False,
    requires_reference: bool = False,
    requires_evaluator: bool = False,
    protocol_frozen_datasets: frozenset[DatasetId],
) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        version=version,
        scope=scope,
        allowed_datasets=allowed_datasets,
        allowed_context_modes=allowed_context_modes,
        requires_generation=requires_generation,
        requires_reference=requires_reference,
        requires_evaluator=requires_evaluator,
        protocol_frozen_datasets=protocol_frozen_datasets,
    )


METRIC_REGISTRY = MetricRegistry(
    [
        _definition(
            "recall_at_k",
            "sprint3.recall_at_k.v1",
            MetricScope.RETRIEVAL_ARTIFACT,
            _ALL_DATASETS,
            protocol_frozen_datasets=_PUBMED_HOTPOT,
        ),
        _definition(
            "mrr_at_k",
            "sprint3.mrr_at_k.v1",
            MetricScope.RETRIEVAL_ARTIFACT,
            _ALL_DATASETS,
            protocol_frozen_datasets=_PUBMED_HOTPOT,
        ),
        _definition(
            "retrieval_diversity",
            "sprint3.retrieval_diversity.v1",
            MetricScope.RETRIEVAL_ARTIFACT,
            _ALL_DATASETS,
            requires_evaluator=True,
            protocol_frozen_datasets=_ALL_DATASETS,
        ),
        _definition(
            "pubmedqa_decision_accuracy",
            "sprint3.pubmedqa_decision_accuracy.v1",
            MetricScope.GENERATION,
            _PUBMED_ONLY,
            requires_generation=True,
            requires_reference=True,
            protocol_frozen_datasets=_PUBMED_ONLY,
        ),
        *[
            _definition(
                metric_id,
                version,
                MetricScope.GENERATION,
                _ALL_DATASETS,
                requires_generation=True,
                requires_reference=True,
                protocol_frozen_datasets=_PUBMED_HOTPOT,
            )
            for metric_id, version in (
                ("exact_match", "sprint3.exact_match.v1"),
                ("token_f1", "sprint3.token_f1.v1"),
                ("rouge_l", "sprint3.rouge_l.v1"),
            )
        ],
        _definition(
            "faithfulness_to_context",
            "sprint3.faithfulness_to_context.v1",
            MetricScope.GENERATION_CONTEXT,
            _ALL_DATASETS,
            allowed_context_modes=frozenset({ContextMode.WITH_CONTEXT}),
            requires_generation=True,
            requires_evaluator=True,
            protocol_frozen_datasets=frozenset(),
        ),
        _definition(
            "asqa_alpha_ndcg",
            "sprint3.asqa_alpha_ndcg.v1",
            MetricScope.RETRIEVAL_ARTIFACT,
            _ASQA_ONLY,
            protocol_frozen_datasets=frozenset(),
        ),
        _definition(
            "asqa_official_answer",
            "sprint3.asqa_official_answer.v1",
            MetricScope.GENERATION,
            _ASQA_ONLY,
            requires_generation=True,
            requires_reference=True,
            requires_evaluator=True,
            protocol_frozen_datasets=frozenset(),
        ),
    ]
)
