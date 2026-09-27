"""
Shared helpers for DPP diversification.

Candidate preparation follows the same score and embedding conventions used by
the other diversification methods.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def prepare_candidates(
    candidates: List[Tuple[Dict, float]],
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Unpack candidates, min-max-normalise scores, and return L2-normalised embeddings.

    Returns:
        docs         — list of doc dicts, same order as candidates
        norm_scores  — np.ndarray float64, shape [n], values in [0, 1]
        embeddings   — np.ndarray float32, shape [n, dim], L2-normalised
    """
    docs = [doc for doc, _ in candidates]
    if any(isinstance(score, (bool, np.bool_)) for _, score in candidates):
        raise ValueError("candidate relevance scores must be finite real numeric values")
    try:
        raw_scores = np.asarray([score for _, score in candidates], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate relevance scores must be finite numeric values") from exc
    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("candidate relevance scores must be finite numeric values")

    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max > score_min:
        norm_scores = (raw_scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones_like(raw_scores)

    if precomputed_embs is not None:
        try:
            raw_embeddings = [precomputed_embs[doc["doc_id"]] for doc in docs]
        except KeyError as exc:
            raise KeyError(
                f"doc_id {exc} missing from precomputed_embs; "
                "ensure corpus embeddings cover all candidates."
            ) from exc
    else:
        texts = [doc.get("text", "") for doc in docs]
        raw_embeddings = embed_fn(texts)

    try:
        embeddings = np.asarray(raw_embeddings)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate embeddings must be a 2-D numeric array") from exc
    if not np.issubdtype(embeddings.dtype, np.number):
        raise ValueError("candidate embeddings must be a 2-D numeric array")
    if np.issubdtype(embeddings.dtype, np.complexfloating):
        raise ValueError("candidate embeddings must be real-valued")
    embeddings = embeddings.astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError("candidate embeddings must be a 2-D numeric array")
    if embeddings.shape[0] != len(candidates):
        raise ValueError("candidate embeddings must have one row per candidate")
    if embeddings.shape[1] == 0:
        raise ValueError("candidate embeddings must have a positive feature dimension")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("candidate embeddings must contain only finite values")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("candidate embeddings must have nonzero row norms")
    embeddings = embeddings / norms

    return docs, norm_scores, embeddings


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return [n, n] cosine similarity matrix for L2-normalised embeddings."""
    return (embeddings @ embeddings.T).astype(np.float64)
