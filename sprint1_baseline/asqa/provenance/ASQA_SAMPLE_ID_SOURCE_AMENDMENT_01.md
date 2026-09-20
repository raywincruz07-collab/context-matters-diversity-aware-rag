# ASQA Sample-ID Source Amendment 01

## Status

Prospective correction created before ASQA partition materialization,
selection, retrieval, generation, or protected-final evaluation.

## Evidence

Authoritative dataset: `din0s/asqa`.

Current revision:

`084060f16b46f3165318f760b2339208b19a0bde`

Current train:
- 4,353 rows
- `sample_id` stored as exact decimal string

Older revision:

`087739578577ce56966efd6e4dc7a63b40eecaf0`

Older train:
- 4,353 rows
- `sample_id` stored as int64

The two revisions have identical scientific train content and row order:
4,353 / 4,353 rows match when `sample_id` is excluded.

Only 24 / 4,353 IDs are numerically identical.

For all 4,353 rows:

`int(float(current_string_sample_id)) == older_int64_sample_id`

Therefore the older int64 IDs contain float64 precision loss.
4,329 IDs differ from the exact current IDs.

## Decision

Sprint 3 ASQA will use revision:

`084060f16b46f3165318f760b2339208b19a0bde`

The exact source `sample_id` decimal string is authoritative.

No floating-point conversion of ASQA IDs is permitted.

The canonical ID must satisfy:

`sample_id == str(int(sample_id))`

and its numeric value must fit signed int64.

## Partition Rule

The frozen 3,482 DEVELOPMENT / 871 SELECTION split is unchanged.

The existing SHA-256 partition rule is unchanged except that
`canonical_decimal_sample_id` means the exact validated source string.

## Supersession

This amendment supersedes only the statement in
`ASQA_INTERNAL_PARTITION_PROTOCOL.md` that the physical upstream ID is int64.

All other partition, evidence-role, leakage, and protected-final rules remain
unchanged.

## Leakage Status

At this amendment:
- the 3,482/871 partition has not been materialized;
- SELECTION outcomes have not been inspected;
- protected ASQA dev948 outcomes have not been inspected;
- no ASQA retrieval or generation results have been produced.

## Physical Source Identity

Pinned train file:

`data/train-00000-of-00001-87b7d64f7913b544.parquet`

SHA-256:

`ce8b9c0563bfcc746afaea504701aa90da9f997ab1f45fe5dc7ba64fd6d7f619`

Pinned dev file:

`data/dev-00000-of-00001-58a9a40c6e69f07b.parquet`

SHA-256:

`9e017bc3a4409902ec994e87c1d0407c06d4b09342bae982031b6c3ea4daaf5b`

Physical verification confirmed:
- train = 4,353 unique IDs;
- dev = 948 unique IDs;
- train/dev ID overlap = 0;
- all IDs use canonical decimal-string representation;
- no protected dev question, answer, annotation, or result was inspected.
