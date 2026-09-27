"""
Determinantal Point Process (DPP) diversification reranking — Sprint 2.

Kernel:  L = Diag(q) · S · Diag(q)
         q_i = exp(theta * (rel_i - max(rel)) / 2)   (quality scores in (0, 1])
         S   = cosine Gram matrix of L2-normalised embeddings

Two modes
---------
map    — deterministic forward-greedy approximation to fixed-cardinality
         determinant maximisation.  No RNG required.
sample — exact k-DPP sample via eigendecomposition.  Requires seed != None.

Reference: Kulesza & Taskar (2012), "Determinantal Point Processes for Machine Learning".
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from diversification._common import cosine_similarity_matrix, prepare_candidates


def build_dpp_kernel(
    norm_scores: np.ndarray,
    embeddings: np.ndarray,
    theta: float = 1.0,
) -> np.ndarray:
    """
    Build the DPP L-kernel.

    Args:
        norm_scores: float64 array [n], min-max normalised relevance in [0, 1].
        embeddings:  float32 array [n, dim], L2-normalised.
        theta:       quality-score temperature (higher → relevance dominates).

    Returns:
        L: float64 [n, n] symmetric PSD matrix.
    """
    q = np.exp(theta * (norm_scores - norm_scores.max()) / 2.0)  # (0, 1]
    S = cosine_similarity_matrix(embeddings)  # [n, n]
    L = np.outer(q, q) * S

    # Scale-relative jitter for numerical stability
    jitter = 1e-6 * max(float(np.abs(L).max()), 1.0)
    L += jitter * np.eye(len(q))

    return L


def _greedy_map(L: np.ndarray, k: int) -> List[int]:
    """
    Forward-greedy approximation to fixed-cardinality determinant maximisation.

    Uses the Cholesky update trick for O(nk^2) complexity.
    """
    n = L.shape[0]
    k = min(k, n)
    selected: List[int] = []
    remaining = list(range(n))

    # Work in the Schur complement space
    # D[i] = L[i,i] — diagonal of the remaining submatrix (updated each step)
    D = np.diag(L).copy()
    # V[i] = projection vector of item i onto the selected subspace
    V = np.zeros((n, k), dtype=np.float64)

    for step in range(k):
        # D[i] is the Schur-complement determinant ratio (conditional
        # variance); log(D[i]) is the log-det increment when D[i] > 0.
        # Maximising either quantity selects the same candidate.
        scores = D[remaining]
        best_local = int(np.argmax(scores))
        best_i = remaining[best_local]
        e_i = D[best_i]
        if not np.isfinite(e_i) or e_i <= 0:
            raise RuntimeError("greedy DPP encountered a nonpositive or nonfinite pivot")

        selected.append(best_i)
        remaining.remove(best_i)

        if not remaining:
            break

        # Update Schur complement diagonal for remaining items
        v = (L[remaining, best_i] - V[remaining, :step] @ V[best_i, :step]) / np.sqrt(e_i)
        V[remaining, step] = v
        D[remaining] -= v ** 2

    return selected


def _sample_kdpp(L: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """
    Exact k-DPP sample via eigendecomposition (Kulesza & Taskar Algorithm 1).
    """
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix")
    if not np.all(np.isfinite(L)):
        raise ValueError("L must contain only finite values")
    if not isinstance(k, (int, np.integer)) or k < 0 or k > L.shape[0]:
        raise ValueError(f"k must be an integer in [0, {L.shape[0]}]")

    n = L.shape[0]
    scale = max(float(np.max(np.abs(L))) if L.size else 0.0, np.finfo(np.float64).tiny)
    if not np.allclose(L, L.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("L must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh((L + L.T) / 2.0)
    eigenvalue_scale = max(
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0,
        np.finfo(np.float64).tiny,
    )
    negative_tolerance = 1e-10 * eigenvalue_scale
    if eigenvalues.size and float(eigenvalues.min()) < -negative_tolerance:
        raise ValueError("L must be positive semidefinite")
    eigenvalues[eigenvalues < 0.0] = 0.0

    if k == 0:
        return []

    # E[l, i] is log e_l(lambda_1, ..., lambda_i).  A common rescaling of
    # all eigenvalues does not change a fixed-cardinality k-DPP and prevents
    # overflow in the elementary-symmetric-polynomial recurrence.
    max_eigenvalue = float(eigenvalues.max())
    if max_eigenvalue == 0.0:
        raise ValueError("k-DPP has zero mass for the requested cardinality")
    scaled_eigenvalues = eigenvalues / max_eigenvalue
    log_eigenvalues = np.full(n, -np.inf, dtype=np.float64)
    positive = scaled_eigenvalues > 0.0
    log_eigenvalues[positive] = np.log(scaled_eigenvalues[positive])

    log_E = np.full((k + 1, n + 1), -np.inf, dtype=np.float64)
    log_E[0, :] = 0.0
    for i in range(1, n + 1):
        upper = min(i, k)
        for degree in range(1, upper + 1):
            log_E[degree, i] = np.logaddexp(
                log_E[degree, i - 1],
                log_eigenvalues[i - 1] + log_E[degree - 1, i - 1],
            )

    if not np.isfinite(log_E[k, n]):
        raise ValueError("k-DPP has zero mass for the requested cardinality")

    # Select exactly k eigenvectors using the conditional k-DPP probabilities.
    chosen_vecs: List[int] = []
    degree = k
    for i in range(n, 0, -1):
        if degree == 0:
            break
        log_probability = (
            log_eigenvalues[i - 1]
            + log_E[degree - 1, i - 1]
            - log_E[degree, i]
        )
        if np.isneginf(log_probability):
            probability = 0.0
        else:
            probability = float(np.exp(log_probability))
            probability_tolerance = 1e-12
            if (
                not np.isfinite(probability)
                or probability < -probability_tolerance
                or probability > 1.0 + probability_tolerance
            ):
                raise RuntimeError("invalid k-DPP eigenvector-selection probability")
            probability = min(max(probability, 0.0), 1.0)
        if rng.random() < probability:
            chosen_vecs.append(i - 1)
            degree -= 1

    if degree != 0 or len(chosen_vecs) != k:
        raise RuntimeError("failed to select exactly k eigenvectors")

    # Sample from the resulting projection DPP, maintaining an orthonormal
    # basis and reducing its rank by one after every item selection.
    V = eigenvectors[:, chosen_vecs]  # [n, k]
    selected: List[int] = []
    for rank in range(k, 0, -1):
        probabilities = np.sum(V * V, axis=1) / rank
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities[selected] = 0.0
        total = float(probabilities.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("invalid projection-DPP item probabilities")
        probabilities /= total
        chosen_item = int(rng.choice(n, p=probabilities))
        selected.append(chosen_item)

        if rank == 1:
            continue

        pivot = int(np.argmax(np.abs(V[chosen_item, :])))
        pivot_value = float(V[chosen_item, pivot])
        if abs(pivot_value) <= np.finfo(np.float64).eps:
            raise RuntimeError("projection-DPP update encountered a zero pivot")
        keep = np.arange(rank) != pivot
        V = V[:, keep] - np.outer(
            V[:, pivot], V[chosen_item, keep] / pivot_value
        )
        V, R = np.linalg.qr(V, mode="reduced")
        rank_tolerance = np.finfo(np.float64).eps * max(n, rank)
        if np.any(np.abs(np.diag(R)) <= rank_tolerance):
            raise RuntimeError("projection-DPP basis lost numerical rank")

    return selected


def dpp_rerank(
    query: str,
    candidates: List[Tuple[Dict, float]],
    top_k: int,
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
    theta: float = 1.0,
    mode: str = "map",
    seed: Optional[int] = None,
) -> List[Tuple[Dict, float]]:
    """
    Rerank candidates using a DPP L-kernel.

    Args:
        query:           Query string (unused directly; kept for interface symmetry).
        candidates:      List of (doc, score) from retriever, relevance-descending.
        top_k:           Number of documents to return.
        embed_fn:        Callable(texts) -> np.ndarray [n, dim].
        precomputed_embs: Optional {doc_id -> embedding}.
        theta:           Quality temperature for the DPP kernel.
        mode:            "map" (deterministic forward-greedy approximation to
                         fixed-cardinality determinant maximisation) or
                         "sample" (exact k-DPP).
        seed:            Required when mode="sample"; ignored for mode="map".

    Returns:
        List of (doc, norm_score) of length min(top_k, len(candidates)).
    """
    if not candidates:
        return []

    if mode not in ("map", "sample"):
        raise ValueError(f"mode must be 'map' or 'sample', got {mode!r}")
    if mode == "sample" and seed is None:
        raise ValueError("mode='sample' requires a seed for reproducibility")

    docs, norm_scores, embeddings = prepare_candidates(candidates, embed_fn, precomputed_embs)
    top_k = min(top_k, len(docs))

    L = build_dpp_kernel(norm_scores, embeddings, theta=theta)

    if mode == "map":
        selected = _greedy_map(L, top_k)
    else:
        rng = np.random.default_rng(seed)
        selected = _sample_kdpp(L, top_k, rng)

    if mode == "map" and len(selected) != top_k:
        raise RuntimeError("greedy DPP returned the wrong cardinality")
    elif mode == "sample" and len(selected) != top_k:
        raise RuntimeError("exact k-DPP sampler returned the wrong cardinality")

    return [(docs[i], float(norm_scores[i])) for i in selected[:top_k]]
