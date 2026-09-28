# Sprint 3 Generation Protocol

## 1. Authority and Purpose

This document records the already-adjudicated canonical Sprint-3 generation
methodology in tracked repository authority.

It does not redesign the generator experiment.

It supersedes older Sprint-1 and Sprint-2 prompt, model, and default-generation
behavior for canonical Sprint-3 runs.

## 2. Generator LLMs

The canonical primary generator set is:

- `llama-3.3-70b`;
- `gemma4-26b`;
- `ministral-3-14b`.

The reserve generator is:

- `qwen3.5-122b`.

Reserve substitution is permitted only prospectively before Stage 2 closes.
It may be invoked only for:

- primary-model unavailability;
- operability failure;
- repeatability-gate failure.

It requires the prospective amendment and provenance procedure in
`docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`.

Once Stage 3 `SELECTION` opens, canonical generator substitution is
prohibited.

Never substitute based on:

- correctness;
- faithfulness;
- a retrieval result;
- answer quality;
- comparative performance.

There is no mid-experiment quality-based model swap. Provider and runtime
physical IDs are recorded later as provenance.

## 3. Fixed System Instruction

The canonical system instruction is frozen as the following exact text:

```text
Answer the question accurately using the requested output format. When context is provided, base your answer on that context. If you cannot answer reliably, state that briefly rather than inventing facts. Give only the requested output; do not provide step-by-step reasoning.
```

The system instruction:

- is byte-identical between `WITH_CONTEXT` and `WITHOUT_CONTEXT`;
- must preserve the exact UTF-8 content above after the protocol text is
  materialized into the prompt asset;
- has no dataset-specific, method-specific, or retriever-specific variants.

## 4. PubMedQA Output Contract

The canonical requested format is:

```text
Decision: <yes|no|maybe>
Explanation: <1-3 concise factual sentences>
```

The explanation exists because faithfulness requires self-contained factual
claims. Correctness remains based only on the parsed `Decision` field.

Do not request chain-of-thought.

## 5. HotpotQA Output Contract

The canonical requested format is:

```text
Answer: <short answer>
Explanation: <1-3 concise factual sentences>
```

The short `Answer` field is used for official answer F1 and EM. The
`Explanation` is excluded from short-answer correctness parsing but may supply
factual claims for faithfulness and ACC.

Do not request chain-of-thought.

## 6. ASQA Output Contract

The canonical requested format is:

```text
Answer: <clear long-form answer resolving ambiguity and covering relevant distinct interpretations when applicable>
```

Do not impose a fixed number of interpretations. Do not force arbitrary bullet
counts.

The answer should resolve relevant ambiguity while remaining concise enough
for the frozen evaluator and window contract.

## Exact User-Message Templates

The canonical user-message template is part of scientific prompt identity.

No implementation may add, remove, reorder, paraphrase, or otherwise alter the
user-facing instructions without a prospective protocol amendment.

The only allowed dynamic substitutions are:

- `{question}`;
- `{context_block}` for `WITH_CONTEXT`.

The exact templates are frozen below.

### PubMedQA WITH_CONTEXT

```text
Question:
{question}

Context:
{context_block}

Output format:
Decision: <yes|no|maybe>
Explanation: <1-3 concise factual sentences>
```

### PubMedQA WITHOUT_CONTEXT

```text
Question:
{question}

Output format:
Decision: <yes|no|maybe>
Explanation: <1-3 concise factual sentences>
```

### HotpotQA WITH_CONTEXT

```text
Question:
{question}

Context:
{context_block}

Output format:
Answer: <short answer>
Explanation: <1-3 concise factual sentences>
```

### HotpotQA WITHOUT_CONTEXT

```text
Question:
{question}

Output format:
Answer: <short answer>
Explanation: <1-3 concise factual sentences>
```

### ASQA WITH_CONTEXT

```text
Question:
{question}

Context:
{context_block}

Output format:
Answer: <clear long-form answer resolving ambiguity and covering relevant distinct interpretations when applicable>
```

### ASQA WITHOUT_CONTEXT

```text
Question:
{question}

Output format:
Answer: <clear long-form answer resolving ambiguity and covering relevant distinct interpretations when applicable>
```

### Context-Mode Identity

`WITH_CONTEXT` and `WITHOUT_CONTEXT` use byte-identical non-context text.

For a given dataset, the only user-message difference between the two modes is
the presence or absence of:

```text
Context:
{context_block}
```

Do not add wording such as:

- `answer from your own knowledge` to `WITHOUT_CONTEXT`;
- `use only the supplied context` as an additional user-message instruction;
- a context-specific refusal instruction;
- retriever names;
- diversification-method names;
- LLM-specific wording;
- method-specific wording.

`{question}` is replaced by the exact canonical dataset question text.
Canonical question text must not be:

- paraphrased;
- normalized;
- rewritten;
- shortened;
- expanded;
- semantically altered.

The separate paraphrase-robustness pilot is governed by
`docs/sprint3/PARAPHRASE_ROBUSTNESS_PROTOCOL.md` and does not modify canonical
experiment query text.

## 7. Context Treatment

### WITH_CONTEXT

`WITH_CONTEXT` requires exactly five successfully selected passages.

`{context_block}` is exactly:

```text
[Document 1]
<body_1>

[Document 2]
<body_2>

[Document 3]
<body_3>

[Document 4]
<body_4>

[Document 5]
<body_5>
```

The rendering rules are frozen as:

- exactly one newline between `[Document N]` and its passage body;
- exactly one empty line between consecutive documents;
- no colon after `[Document N]`;
- no title;
- no retrieval score;
- no relevance metadata;
- no retriever identity;
- no diversification identity;
- no document ID;
- no qrel;
- no gold label;
- no method identity;
- passages remain in selected rank order;
- use the canonical passage-body text defined by the corresponding corpus
  contract;
- do not silently rewrite or normalize passage text beyond already-frozen
  corpus normalization.

The `[Document N]` scaffold is prompt structure only. Faithfulness evidence
excludes this scaffold and uses the actual passage body.

## 8. WITHOUT_CONTEXT Treatment

`WITHOUT_CONTEXT` uses the same:

- system instruction;
- dataset-specific question and output instruction;
- question text;
- decoding configuration.

The only treatment difference is that the context block is absent.

Do not insert:

- `no context available`;
- empty fake document tags;
- method names;
- retriever names;
- any placeholder that changes the linguistic treatment.

## 9. WITHOUT_CONTEXT Identity and Reuse

A canonical `WITHOUT_CONTEXT` generation is keyed by:

```text
(
  dataset,
  sample_id,
  LLM,
  prompt_version,
  decoding_version
)
```

It is not keyed by:

- retriever;
- diversifier;
- diversification hyperparameter;
- candidate artifact.

Generate and reuse exactly once per:

```text
dataset x sample x LLM x canonical generation replica
```

Retrieval-independent `WITHOUT_CONTEXT` must not be multiplied across
retrieval conditions.

## 10. Decoding

The decoding controls are frozen as:

```text
temperature = 0
canonical_generation_replicas = 1
```

Maximum output tokens are:

```text
PubMedQA = 256
HotpotQA = 256
ASQA     = 512
```

Use the provider's direct or non-thinking mode where supported by the frozen
Maki runtime contract. Do not request hidden reasoning or step-by-step
reasoning.

Temperature zero and direct mode are treated as low-stochasticity controls.
They are not assumed to guarantee bitwise deterministic service output.

## 11. Generation Status

The canonical statuses are:

```text
OK
REFUSAL
PARSE_FAILURE
TRUNCATED
ERROR
```

Their definitions must remain compatible with the answer-correctness and
experiment stage-gate protocols.

### OK

A successful response satisfying upstream transport and runtime conditions.

### REFUSAL

A successful model response explicitly declining or not answering the task.

### PARSE_FAILURE

A successful model response whose required output grammar cannot be parsed.

### TRUNCATED

A response terminated by the configured output-length or token limit.

### ERROR

A transport, infrastructure, provider, or runtime failure after governed
retries.

## 12. Retry Policy

Content outcomes are not retried merely to obtain a preferable response.

Do not retry because of:

- `REFUSAL`;
- malformed successful model content;
- `PARSE_FAILURE`;
- poor correctness;
- low faithfulness;
- an undesirable answer;
- inconvenient wording.

Only transport or infrastructure errors may be retried.

The maximum is:

```text
3 total attempts for one canonical generation request
```

Preserve attempt metadata. Do not silently replace a failed or content-invalid
response with a later convenient answer.

## 13. Parse Failure

A successful provider response that violates the dataset output grammar is:

```text
PARSE_FAILURE
```

Do not:

- infer the field from explanation prose;
- ask the model again until it follows the format;
- semantically repair it using another model.

Dataset-specific parsers are governed by
`docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`.

## 14. Truncation

If the provider indicates length or output-token truncation:

```text
status = TRUNCATED
```

Do not score a partial answer as complete under canonical correctness. Do not
automatically regenerate with a larger token limit.

Any change to token limits requires a prospective protocol amendment.

## 15. Refusal

`REFUSAL` is retained as the model's semantic task outcome.

Under the frozen correctness protocol:

```text
REFUSAL -> correctness zero
```

Faithfulness and ACC treatment follows their own frozen construct protocols.
Do not convert refusal to infrastructure missingness.

## 16. Context Success Requirement

Canonical `WITH_CONTEXT` generation requires exactly five valid selected
passages.

A retrieval or context-selection failure such as:

```text
SHORT_CANDIDATE_LIST
```

does not produce a repaired successful `WITH_CONTEXT` row.

Do not fill missing passages using:

- lower-ranked external documents;
- gold documents;
- duplicated documents;
- alternate-retriever output.

The generation row remains unavailable or failure-linked according to the run
and artifact contract.

## 17. Controlled Comparison Principle

Within a comparison, hold fixed:

- generator LLM;
- model and runtime identity;
- system prompt;
- dataset output instruction;
- question text;
- decoding;
- maximum output tokens;
- context rendering;
- generation replica count.

Only the intended treatment changes.

For baseline-versus-diversified `WITH_CONTEXT`, only the selected passage set
or order may differ.

For `WITH_CONTEXT` versus `WITHOUT_CONTEXT`, only context presence differs.

## 18. Three-LLM Policy

The same generation protocol applies to all three primary LLMs.

There is no LLM-specific:

- prompt;
- formatting exception;
- decoding temperature;
- token budget;
- parser leniency;
- retry rule.

Provider-specific transport fields may differ only where scientifically
neutral and must be provenance-recorded.

## 19. Repeatability Gate

Before canonical Stage-3 generation, perform the already-frozen model
repeatability and operability gate.

For each primary LLM, use:

- 20 frozen `DEVELOPMENT` prompts;
- three identical calls per prompt.

The pass condition is:

```text
at least 19 of 20 prompts produce all three stripped raw responses identically
```

If a primary LLM fails, investigate provider-supported deterministic and
direct-mode controls on `DEVELOPMENT`.

If the failure remains unresolved before Stage 2 closes, the frozen reserve
policy may be invoked prospectively under the amendment and provenance
procedure. Never substitute based on answer quality.

Record the repeatability artifact and exact model and runtime identity.

## 20. Generation Artifact Identity

Every generation artifact must bind or reference:

- dataset;
- evidence role;
- sample ID;
- LLM logical ID;
- physical provider and model ID;
- model or runtime revision, if available;
- prompt version and hash;
- exact system-message UTF-8 bytes and hash;
- exact user-template UTF-8 bytes and hash;
- exact fully rendered user-message hash;
- exact fully rendered prompt hash;
- decoding version and hash;
- context mode;
- context-artifact hash for `WITH_CONTEXT`;
- exact context-block hash for `WITH_CONTEXT`;
- `NOT_APPLICABLE` retrieval fields for `WITHOUT_CONTEXT`;
- maximum output tokens;
- temperature;
- replica;
- attempt count;
- raw response;
- status;
- provider finish reason;
- timestamps;
- Git commit and run identity.

The fully rendered prompt hash must detect changes in:

- instruction wording;
- whitespace and newline structure;
- canonical question text;
- context body text;
- context order.

Do not duplicate a `WITHOUT_CONTEXT` row across retriever or diversifier IDs.

## 21. Legacy Implementation Warning

Existing Sprint-1 and Sprint-2 code or configuration may contain:

- `qwen3.5-122b` as a default generator;
- an older medical prompt;
- temperature values other than zero;
- historical generation contracts.

These historical defaults are not canonical Sprint-3 authority.

Canonical Sprint-3 implementation must fail fast rather than silently inherit
legacy defaults that differ from this protocol.

Actual code quarantine or update is an implementation task and is not
performed by this document.

## 22. Protected-Final Rule

This generation protocol must be implemented and validated on `DEVELOPMENT`
before `SELECTION` opens.

After `SELECTION` or protected-final outcomes are observed, do not change:

- system prompt;
- output grammar;
- context renderer;
- temperature;
- token limits;
- retry semantics;
- replica count;
- no-context identity.

Any scientifically meaningful change requires a prospective amendment and
exposure accounting.

## 23. Relation to Other Protocols

This protocol must be read with:

- `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`;
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`;
- `docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md`.

Forthcoming dedicated ACC and faithfulness protocols will carry their complete
evaluator semantics. This generation document does not define ACC or
faithfulness.

## 24. Frozen Before Observation

This tracked document records generation methodology already adjudicated
before Stage-3 `SELECTION` or protected-final outcomes are observed.

No selection or protected-final results were used to choose these generation
rules.
