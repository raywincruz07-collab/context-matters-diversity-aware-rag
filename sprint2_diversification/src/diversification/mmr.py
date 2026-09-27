"""
Maximal Marginal Relevance (MMR) diversification reranking — Sprint 2.

Standard MMR (Carbonell & Goldstein, 1998):
  score(c) = lambda * Rel(c) - (1 - lambda) * max_{s in S} sim(c, s)

  where S is the already-selected set.

Boundary conditions (verified by tests/test_mmr.py):
  lambda=1.0 → pure relevance ranking (diversity term is zeroed out)
  lambda=0.0 → pure diversity maximisation (relevance term is zeroed out)

Relevance scores are min-max normalised per query to [0, 1] so that
raw scores from BM25/DPR/Contriever/ColBERTv2 are comparable in the
combined formula.

Embeddings come from facebook/contriever (mean-pooling + L2-normalised),
reusing ContrieverRetriever._encode from retrievers/dense_retriever.py.
"""

import os
from numbers import Real
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import EMBEDDINGS_DIR


def mmr_rerank(
    query: str,
    candidates: List[Tuple[Dict, float]],
    top_k: int,
    lambda_param: float,
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
) -> List[Tuple[Dict, float]]:
    """
    Rerank candidates using Maximal Marginal Relevance.

    Args:
        query: The query string (kept for interface symmetry; not used directly
               because relevance comes from the retriever scores, not re-encoded).
        candidates: List of (doc, relevance_score) from a retriever, ordered by
                    relevance descending.  Must not be empty.
        top_k: Number of documents to return.
        lambda_param: MMR trade-off in [0.0, 1.0].
                      1.0 → pure relevance, 0.0 → pure diversity.
        embed_fn: Callable(texts: List[str]) -> np.ndarray of shape [n, dim],
                  expected to return L2-normalised vectors (cosine sim = dot product).
                  Called on candidate texts when precomputed_embs is None.
        precomputed_embs: Optional dict mapping doc_id -> np.ndarray (pre-computed
                          corpus embeddings).  When provided, embed_fn is not called.

    Returns:
        List of (doc, normalised_relevance_score) of length min(top_k, len(candidates)),
        ordered by MMR selection.
    """
    if (
        isinstance(lambda_param, (bool, np.bool_))
        or not isinstance(lambda_param, Real)
        or not np.isfinite(lambda_param)
        or not 0.0 <= lambda_param <= 1.0
    ):
        raise ValueError("lambda_param must be a finite real number in [0, 1]")
    if (
        isinstance(top_k, (bool, np.bool_))
        or not isinstance(top_k, (int, np.integer))
        or top_k < 0
    ):
        raise ValueError("top_k must be a nonnegative integer")

    if not candidates:
        return []

    docs = [doc for doc, _ in candidates]
    try:
        raw_scores = np.array([score for _, score in candidates], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate relevance scores must be finite real numbers") from exc
    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("candidate relevance scores must be finite real numbers")

    # Normalise relevance scores to [0, 1] per query
    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max > score_min:
        norm_scores = (raw_scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones_like(raw_scores)

    top_k = min(top_k, len(candidates))
    if top_k == 0:
        return []

    # Obtain embeddings for diversity computation
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
    embeddings = embeddings.astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError("candidate embeddings must be a 2-D numeric array")
    if embeddings.shape[0] != len(candidates):
        raise ValueError("candidate embeddings must have one row per candidate")
    if embeddings.shape[1] == 0:
        raise ValueError("candidate embeddings must have a positive feature dimension")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("candidate embeddings must contain only finite values")

    # Ensure L2-normalised (guard against embed_fn not normalising)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("candidate embeddings must have nonzero row norms")
    embeddings = embeddings / norms

    selected_indices: List[int] = []
    remaining_indices: List[int] = list(range(len(docs)))

    while len(selected_indices) < top_k and remaining_indices:
        if not selected_indices:
            # Seed every lambda endpoint with the highest-relevance candidate.
            # np.argmax gives deterministic first-occurrence tie-breaking.
            best_local = int(np.argmax(norm_scores[remaining_indices]))
            best_idx = remaining_indices[best_local]
        else:
            selected_embs = embeddings[selected_indices]  # [n_sel, dim]
            best_score = -float("inf")
            best_idx = remaining_indices[0]

            for idx in remaining_indices:
                rel = lambda_param * float(norm_scores[idx])
                # cosine sim to each already-selected doc (dot product, vectors are normalised)
                sims = embeddings[idx] @ selected_embs.T  # [n_sel]
                div = (1.0 - lambda_param) * float(sims.max())
                score = rel - div

                if score > best_score:
                    best_score = score
                    best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [(docs[i], float(norm_scores[i])) for i in selected_indices]


def get_hotpot_corpus_embeddings(
    corpus: List[Dict],
    embed_fn: Callable[[List[str]], np.ndarray],
) -> np.ndarray:
    """
    Load or compute Contriever embeddings for the full HotpotQA corpus.

    Uses the same stale-cache shape guard as OriginalDPRRetriever.index():
    if the cached file exists but its row count doesn't match len(corpus),
    recompute and overwrite.

    Returns:
        np.ndarray of shape [len(corpus), 768], L2-normalised.
    """
    emb_path = os.path.join(EMBEDDINGS_DIR, "contriever_embeddings_hotpot.npy")

    if os.path.exists(emb_path):
        cached = np.load(emb_path)
        if cached.shape[0] == len(corpus):
            print(f"Loaded cached HotpotQA Contriever embeddings: {cached.shape}")
            return cached
        print(
            f"Stale embedding cache (shape {cached.shape[0]} rows "
            f"!= corpus size {len(corpus)}); recomputing ..."
        )

    print(f"Computing Contriever embeddings for {len(corpus)} HotpotQA corpus docs ...")
    texts = [doc.get("text", "") for doc in corpus]
    embeddings = embed_fn(texts).astype(np.float32)

    # Ensure L2-normalised
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    np.save(emb_path, embeddings)
    print(f"Saved HotpotQA Contriever embeddings to {emb_path}  shape={embeddings.shape}")
    return embeddings
