"""
Condition dispatch for Sprint 2 diversification — routes a condition string
to the appropriate reranker (MMR, DPP, clustering, or none).

Condition grammar
-----------------
  none          — no reranking; return retriever output as-is
  mmr_<lambda>  — MMR with lambda in [0.0, 1.0]  e.g. mmr_0.5
  kmeans_k<k>   — k-means clustering with k clusters  e.g. kmeans_k5
  agglo_k<k>    — agglomerative clustering  e.g. agglo_k5
  dpp_map       — DPP greedy MAP (deterministic)
  dpp_seed<n>   — DPP exact k-DPP sample with seed n  e.g. dpp_seed42
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_CONDITION_PATTERNS = [
    (re.compile(r"^none$"), "none"),
    (re.compile(r"^mmr_(\d+(?:\.\d+)?)$"), "mmr"),
    (re.compile(r"^kmeans_k(\d+)$"), "kmeans"),
    (re.compile(r"^agglo_k(\d+)$"), "agglo"),
    (re.compile(r"^dpp_map$"), "dpp_map"),
    (re.compile(r"^dpp_seed(\d+)$"), "dpp_sample"),
]


def parse_condition(condition: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a condition string into (family, kwargs).

    Returns:
        family: one of "none", "mmr", "kmeans", "agglo", "dpp_map", "dpp_sample"
        kwargs: dict of extra arguments passed to the reranker

    Raises:
        ValueError: if condition does not match any known pattern.
    """
    for pattern, family in _CONDITION_PATTERNS:
        m = pattern.match(condition)
        if m is None:
            continue
        if family == "none":
            return "none", {}
        if family == "mmr":
            lambda_param = float(m.group(1))
            if not 0.0 <= lambda_param <= 1.0:
                raise ValueError("MMR lambda must be in [0, 1]")
            return "mmr", {"lambda_param": lambda_param}
        if family == "kmeans":
            n_clusters = int(m.group(1))
            if n_clusters < 1:
                raise ValueError("KMeans cluster count must be at least 1")
            return "kmeans", {"n_clusters": n_clusters, "method": "kmeans"}
        if family == "agglo":
            n_clusters = int(m.group(1))
            if n_clusters < 1:
                raise ValueError("Agglomerative cluster count must be at least 1")
            return "agglo", {"n_clusters": n_clusters, "method": "agglomerative"}
        if family == "dpp_map":
            return "dpp_map", {"mode": "map"}
        if family == "dpp_sample":
            return "dpp_sample", {"mode": "sample", "seed": int(m.group(1))}
    raise ValueError(
        f"Unknown condition {condition!r}. "
        "Expected: none | mmr_<lambda> | kmeans_k<k> | agglo_k<k> | dpp_map | dpp_seed<n>"
    )


def is_diversified(condition: str) -> bool:
    """Return True for any condition that applies reranking (i.e. not 'none')."""
    family, _ = parse_condition(condition)
    return family != "none"


def rerank(
    condition: str,
    query: str,
    candidates: List[Tuple[Dict, float]],
    top_k: int,
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
) -> List[Tuple[Dict, float]]:
    """
    Dispatch reranking based on condition string.

    For condition="none", returns the top_k candidates by retriever score
    without any reranking.

    Args:
        condition:        Condition string (see module docstring for grammar).
        query:            Query string forwarded to reranker.
        candidates:       List of (doc, score) from retriever.
        top_k:            Number of results to return.
        embed_fn:         Embedding callable required by all diversifiers.
        precomputed_embs: Optional pre-computed corpus embeddings.

    Returns:
        List of (doc, score) of length min(top_k, len(candidates)).
    """
    family, kwargs = parse_condition(condition)

    if family == "none":
        return candidates[:top_k]

    if family == "mmr":
        from diversification.mmr import mmr_rerank
        return mmr_rerank(
            query, candidates, top_k,
            lambda_param=kwargs["lambda_param"],
            embed_fn=embed_fn,
            precomputed_embs=precomputed_embs,
        )

    if family in ("kmeans", "agglo"):
        from diversification.clustering import cluster_rerank
        return cluster_rerank(
            query, candidates, top_k,
            embed_fn=embed_fn,
            precomputed_embs=precomputed_embs,
            n_clusters=kwargs["n_clusters"],
            method=kwargs["method"],
        )

    if family in ("dpp_map", "dpp_sample"):
        from diversification.dpp import dpp_rerank
        return dpp_rerank(
            query, candidates, top_k,
            embed_fn=embed_fn,
            precomputed_embs=precomputed_embs,
            mode=kwargs["mode"],
            seed=kwargs.get("seed"),
        )

    raise ValueError(f"Unhandled family {family!r}")  # unreachable
