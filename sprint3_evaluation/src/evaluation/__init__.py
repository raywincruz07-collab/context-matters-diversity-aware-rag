"""
Evaluation metrics for RAG pipeline.

Phase 1 baseline metrics:
- Exact Match (EM)
- Token-level F1
- ROUGE-L
- Retrieval metrics: Recall@K, MRR

Sprint 2 additions:
- retrieval_diversity: mean pairwise cosine distance among retrieved doc embeddings
- faithfulness_nli: NLI entailment score of generated answer against context docs
"""

import re
import string
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, articles, extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 score using longest common subsequence."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    # LCS length
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == truth_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def retrieval_recall_at_k(
    retrieved_doc_ids: List[int], gold_doc_ids: List[int]
) -> float:
    """What fraction of gold documents are in the retrieved set?"""
    if not gold_doc_ids:
        return 0.0
    retrieved_set = set(retrieved_doc_ids)
    hits = sum(1 for gid in gold_doc_ids if gid in retrieved_set)
    return hits / len(gold_doc_ids)


def mrr(retrieved_doc_ids: List[int], gold_doc_ids: List[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document."""
    gold_set = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def evaluate_single(
    prediction: str,
    ground_truth: str,
    retrieved_doc_ids: List[int],
    gold_doc_ids: List[int],
) -> Dict[str, float]:
    """Evaluate a single QA example."""
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "f1": token_f1(prediction, ground_truth),
        "rouge_l": rouge_l(prediction, ground_truth),
        "recall_at_k": retrieval_recall_at_k(retrieved_doc_ids, gold_doc_ids),
        "mrr": mrr(retrieved_doc_ids, gold_doc_ids),
    }


def evaluate_batch(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate evaluation over a batch.

    Args:
        results: List of dicts, each with keys from evaluate_single()

    Returns:
        Dict of averaged metric values
    """
    if not results:
        return {}

    metrics = {}
    for key in results[0]:
        values = [r[key] for r in results]
        metrics[f"avg_{key}"] = sum(values) / len(values)

    return metrics


# ---------------------------------------------------------------------------
# Sprint 2 metrics
# ---------------------------------------------------------------------------


def retrieval_diversity(doc_embeddings: List[np.ndarray]) -> float:
    """
    Mean pairwise cosine distance among the final retrieved doc embeddings.

    Cosine distance = 1 - cosine_similarity.  Range: [0.0, 2.0] (L2-normalised
    vectors bound similarity to [-1, 1], so distance to [0, 2]).

    Args:
        doc_embeddings: List of L2-normalised embedding vectors for each
                        retrieved document.  Computed on the FINAL top-k set
                        (after any MMR reranking) — never on gold documents.

    Returns:
        Mean pairwise cosine distance as a float.  Returns 0.0 for fewer
        than 2 documents (no pairs to compare).
    """
    n = len(doc_embeddings)
    if n < 2:
        return 0.0

    embs = np.array(doc_embeddings, dtype=np.float32)
    # L2-normalise (guard in case caller didn't)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embs = embs / norms

    # Cosine similarity matrix via dot product (O(n^2) but n <= top_k ~ 5)
    sim_matrix = embs @ embs.T  # [n, n]
    # Extract upper-triangle (pairwise, no self-similarity)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[triu_indices]

    mean_distance = float(np.mean(1.0 - pairwise_sims))
    return mean_distance


def faithfulness_nli(
    generated_answer: str,
    context_docs: List[str],
    nli_model,
) -> float:
    """
    Estimate faithfulness via NLI: max entailment probability of the generated
    answer (hypothesis) against each context document (premise).

    Uses facebook/bart-large-mnli (zero-shot NLI).  The generated answer and
    context come from the RAG output — gold references are NEVER passed here.

    Args:
        generated_answer: The LLM-generated answer string.
        context_docs: List of context document texts used for generation.
        nli_model: A loaded pipeline("zero-shot-classification", ...) instance
                   or any callable with the same interface.

    Returns:
        Max entailment probability across all context docs.  Range [0.0, 1.0].
        Returns 0.0 if context_docs is empty or generated_answer is empty.
    """
    if not generated_answer.strip() or not context_docs:
        return 0.0

    max_entailment = 0.0
    for premise in context_docs:
        if not premise.strip():
            continue
        result = nli_model(
            premise,
            candidate_labels=[generated_answer],
            hypothesis_template="{}",
        )
        # zero-shot-classification returns scores for each label
        # The single label score is the entailment probability
        score = float(result["scores"][0])
        if score > max_entailment:
            max_entailment = score

    return max_entailment
