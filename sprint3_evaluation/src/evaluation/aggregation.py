"""Pure metric aggregation and paired context comparison for Sprint 3."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from evaluation.contracts import ContextMode, MetricResult, MetricStatus
from evaluation.metric_registry import METRIC_REGISTRY, DatasetId, MetricScope


DIFFERENCE_DIRECTION = "with_context_minus_without_context"


@dataclass(frozen=True)
class MetricObservation:
    """One query-level metric result with its complete logical identity."""

    sample_id: str
    metric_id: str
    metric_result: MetricResult
    dataset_id: DatasetId
    dataset_split: str
    retriever: str | None
    diversification_condition: str | None
    model_provider: str | None
    model_id: str | None
    model_revision: str | None
    context_mode: ContextMode | None
    generation_replica: int | None
    retrieval_artifact_id: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "sample_id",
            "metric_id",
            "dataset_split",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        definition = METRIC_REGISTRY.get(self.metric_id)
        if not isinstance(self.metric_result, MetricResult):
            raise TypeError("metric_result must be a MetricResult")
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be a DatasetId")
        retrieval_values = (
            self.retriever,
            self.diversification_condition,
            self.retrieval_artifact_id,
        )
        if definition.scope is MetricScope.RETRIEVAL_ARTIFACT:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in retrieval_values
            ):
                raise ValueError(
                    "retrieval metadata must be non-empty for retrieval-artifact "
                    "and with-context observations"
                )
            if any(
                value is not None
                for value in (
                    self.model_provider,
                    self.model_id,
                    self.model_revision,
                    self.context_mode,
                    self.generation_replica,
                )
            ):
                raise ValueError(
                    "retrieval-artifact observations require generation metadata=None"
                )
            return

        for field_name in ("model_provider", "model_id"):
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
            if any(
                not isinstance(value, str) or not value.strip()
                for value in retrieval_values
            ):
                raise ValueError(
                    "retrieval metadata must be non-empty for retrieval-artifact "
                    "and with-context observations"
                )
        elif any(value is not None for value in retrieval_values):
            raise ValueError(
                "without-context generation observations require retrieval metadata=None"
            )
        if (
            isinstance(self.generation_replica, bool)
            or not isinstance(self.generation_replica, Integral)
            or self.generation_replica < 0
        ):
            raise ValueError(
                "generation_replica must be a non-boolean integer >= 0"
            )


@dataclass(frozen=True)
class MetricSummary:
    """Status-aware descriptive summary for one logical metric group."""

    metric_id: str
    metric_scope: MetricScope
    dataset_id: DatasetId
    dataset_split: str
    retriever: str | None
    diversification_condition: str | None
    model_provider: str | None
    model_id: str | None
    model_revision: str | None
    context_mode: ContextMode | None
    generation_replica: int | None
    n_total_expected: int
    n_rows_present: int
    n_measured: int
    n_not_applicable: int
    n_not_computed: int
    n_failed: int
    n_missing_rows: int
    mean: float | None
    std: float | None
    measured_sample_ids: tuple[str, ...]
    missing_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairedMetricSummary:
    """Complete-pair comparison of one generation metric across context modes."""

    metric_id: str
    metric_scope: MetricScope
    dataset_id: DatasetId
    dataset_split: str
    retriever: str
    diversification_condition: str
    model_provider: str
    model_id: str
    model_revision: str | None
    generation_replica: int
    n_pair_expected: int
    n_with_context_present: int
    n_without_context_present: int
    n_pair_rows_present: int
    n_pair_complete: int
    n_pair_incomplete: int
    missing_with_context_sample_ids: tuple[str, ...]
    missing_without_context_sample_ids: tuple[str, ...]
    incomplete_pair_sample_ids: tuple[str, ...]
    paired_sample_ids: tuple[str, ...]
    with_context_mean_on_pairs: float | None
    without_context_mean_on_pairs: float | None
    mean_paired_difference: float | None
    difference_direction: str = DIFFERENCE_DIRECTION


def aggregate_metric_group(
    records: Sequence[MetricObservation], expected_sample_ids: Sequence[str]
) -> MetricSummary:
    """Aggregate one validated logical metric group over an explicit manifest."""
    expected = _validate_expected_sample_ids(expected_sample_ids)
    observations = _validate_observations(records, expected, "records")
    if not observations:
        raise ValueError("records must contain at least one observation")

    definition = METRIC_REGISTRY.get(observations[0].metric_id)
    group_key = _group_key(observations[0], definition.scope)
    for record in observations[1:]:
        if _group_key(record, definition.scope) != group_key:
            raise ValueError("records do not belong to one logical metric group")

    by_sample = {record.sample_id: record for record in observations}
    measured_ids = tuple(
        sample_id
        for sample_id in expected
        if sample_id in by_sample
        and by_sample[sample_id].metric_result.status is MetricStatus.MEASURED
    )
    missing_ids = tuple(sample_id for sample_id in expected if sample_id not in by_sample)
    status_counts = {
        status: sum(
            record.metric_result.status is status for record in observations
        )
        for status in MetricStatus
    }
    measured_values = [
        float(by_sample[sample_id].metric_result.value) for sample_id in measured_ids
    ]
    mean = statistics.fmean(measured_values) if measured_values else None
    std = statistics.stdev(measured_values) if len(measured_values) >= 2 else None

    first = observations[0]
    is_retrieval = definition.scope is MetricScope.RETRIEVAL_ARTIFACT
    return MetricSummary(
        metric_id=observations[0].metric_id,
        metric_scope=definition.scope,
        dataset_id=first.dataset_id,
        dataset_split=first.dataset_split,
        retriever=first.retriever,
        diversification_condition=first.diversification_condition,
        model_provider=None if is_retrieval else first.model_provider,
        model_id=None if is_retrieval else first.model_id,
        model_revision=None if is_retrieval else first.model_revision,
        context_mode=None if is_retrieval else first.context_mode,
        generation_replica=None if is_retrieval else int(first.generation_replica),
        n_total_expected=len(expected),
        n_rows_present=len(observations),
        n_measured=status_counts[MetricStatus.MEASURED],
        n_not_applicable=status_counts[MetricStatus.NOT_APPLICABLE],
        n_not_computed=status_counts[MetricStatus.NOT_COMPUTED],
        n_failed=status_counts[MetricStatus.FAILED],
        n_missing_rows=len(missing_ids),
        mean=mean,
        std=std,
        measured_sample_ids=measured_ids,
        missing_sample_ids=missing_ids,
    )


def compare_context_modes(
    with_context_records: Sequence[MetricObservation],
    without_context_records: Sequence[MetricObservation],
    expected_sample_ids: Sequence[str],
) -> PairedMetricSummary:
    """Compare context modes using only the exact complete-pair intersection."""
    expected = _validate_expected_sample_ids(expected_sample_ids)
    with_records = _validate_observations(
        with_context_records, expected, "with_context_records"
    )
    without_records = _validate_observations(
        without_context_records, expected, "without_context_records"
    )
    if not with_records:
        raise ValueError("with_context_records must contain at least one observation")

    all_records = with_records + without_records
    definition = METRIC_REGISTRY.get(all_records[0].metric_id)
    if definition.scope is not MetricScope.GENERATION:
        raise ValueError("only generation-scope metrics can be context-compared")

    pair_key = _pair_group_key(all_records[0])
    for record in all_records:
        if _pair_group_key(record) != pair_key:
            raise ValueError("records do not belong to one paired logical group")
    with_condition_key = (
        with_records[0].retriever,
        with_records[0].diversification_condition,
    )
    if any(
        (record.retriever, record.diversification_condition) != with_condition_key
        for record in with_records
    ):
        raise ValueError("with_context_records must belong to one retrieval condition")
    if any(record.context_mode is not ContextMode.WITH_CONTEXT for record in with_records):
        raise ValueError("with_context_records must use ContextMode.WITH_CONTEXT")
    if any(
        record.context_mode is not ContextMode.WITHOUT_CONTEXT
        for record in without_records
    ):
        raise ValueError(
            "without_context_records must use ContextMode.WITHOUT_CONTEXT"
        )

    with_by_sample = {record.sample_id: record for record in with_records}
    without_by_sample = {record.sample_id: record for record in without_records}
    both_present = tuple(
        sample_id
        for sample_id in expected
        if sample_id in with_by_sample and sample_id in without_by_sample
    )
    paired_ids = tuple(
        sample_id
        for sample_id in both_present
        if with_by_sample[sample_id].metric_result.status is MetricStatus.MEASURED
        and without_by_sample[sample_id].metric_result.status
        is MetricStatus.MEASURED
    )
    incomplete_ids = tuple(sample_id for sample_id in expected if sample_id not in paired_ids)
    missing_with = tuple(
        sample_id for sample_id in expected if sample_id not in with_by_sample
    )
    missing_without = tuple(
        sample_id for sample_id in expected if sample_id not in without_by_sample
    )

    if paired_ids:
        with_values = [
            float(with_by_sample[sample_id].metric_result.value)
            for sample_id in paired_ids
        ]
        without_values = [
            float(without_by_sample[sample_id].metric_result.value)
            for sample_id in paired_ids
        ]
        with_mean = statistics.fmean(with_values)
        without_mean = statistics.fmean(without_values)
        mean_difference = statistics.fmean(
            with_value - without_value
            for with_value, without_value in zip(with_values, without_values)
        )
    else:
        with_mean = without_mean = mean_difference = None

    return PairedMetricSummary(
        metric_id=all_records[0].metric_id,
        metric_scope=definition.scope,
        dataset_id=with_records[0].dataset_id,
        dataset_split=with_records[0].dataset_split,
        retriever=with_records[0].retriever,
        diversification_condition=with_records[0].diversification_condition,
        model_provider=with_records[0].model_provider,
        model_id=with_records[0].model_id,
        model_revision=with_records[0].model_revision,
        generation_replica=int(with_records[0].generation_replica),
        n_pair_expected=len(expected),
        n_with_context_present=len(with_records),
        n_without_context_present=len(without_records),
        n_pair_rows_present=len(both_present),
        n_pair_complete=len(paired_ids),
        n_pair_incomplete=len(expected) - len(paired_ids),
        missing_with_context_sample_ids=missing_with,
        missing_without_context_sample_ids=missing_without,
        incomplete_pair_sample_ids=incomplete_ids,
        paired_sample_ids=paired_ids,
        with_context_mean_on_pairs=with_mean,
        without_context_mean_on_pairs=without_mean,
        mean_paired_difference=mean_difference,
    )


def _validate_expected_sample_ids(expected_sample_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(expected_sample_ids, (str, bytes)) or not isinstance(
        expected_sample_ids, Sequence
    ):
        raise TypeError("expected_sample_ids must be an ordered sequence")
    expected = tuple(expected_sample_ids)
    if not expected:
        raise ValueError("expected_sample_ids must not be empty")
    if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in expected):
        raise ValueError("expected sample IDs must be non-empty strings")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_sample_ids must not contain duplicates")
    return expected


def _validate_observations(
    records: Sequence[MetricObservation],
    expected: tuple[str, ...],
    field_name: str,
) -> tuple[MetricObservation, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(f"{field_name} must be an ordered sequence")
    observations = tuple(records)
    if not all(isinstance(record, MetricObservation) for record in observations):
        raise TypeError(f"{field_name} must contain MetricObservation objects")
    sample_ids = [record.sample_id for record in observations]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{field_name} contains duplicate sample_id observations")
    unknown = set(sample_ids) - set(expected)
    if unknown:
        raise ValueError(f"observed sample IDs are absent from expected manifest: {sorted(unknown)!r}")
    return observations


def _group_key(record: MetricObservation, scope: MetricScope) -> tuple[object, ...]:
    base = (
        record.dataset_id,
        record.dataset_split,
        record.retriever,
        record.diversification_condition,
        record.metric_id,
    )
    if scope is MetricScope.RETRIEVAL_ARTIFACT:
        return base
    generation_key = (
        record.model_provider,
        record.model_id,
        record.model_revision,
        record.context_mode,
        int(record.generation_replica),
    )
    if record.context_mode is ContextMode.WITHOUT_CONTEXT:
        return (
            record.dataset_id,
            record.dataset_split,
            record.metric_id,
        ) + generation_key
    return base + generation_key


def _pair_group_key(record: MetricObservation) -> tuple[object, ...]:
    return (
        record.dataset_id,
        record.dataset_split,
        record.model_provider,
        record.model_id,
        record.model_revision,
        int(record.generation_replica),
        record.metric_id,
    )
