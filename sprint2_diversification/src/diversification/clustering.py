"""
Clustering-based diversification reranking — Sprint 2.

Cluster-balanced round-robin reranking over embedding clusters, with
relevance-ranked members within clusters. KMeans and Ward-linkage Euclidean
agglomerative clustering operate on L2-normalised Contriever embeddings.

Supported methods: "kmeans", "agglomerative".
When n_clusters == 1 the method degenerates to a plain relevance top-k.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

def cluster_rerank(
    query: str,
    candidates: List[Tuple[Dict, float]],
    top_k: int,
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
    n_clusters: int = 5,
    method: str = "kmeans",
    seed: int = 42,
) -> List[Tuple[Dict, float]]:
    """
    Cluster-balanced round-robin reranking with relevance-ranked cluster members.

    Args:
        query:           Query string (unused; kept for interface symmetry).
        candidates:      List of (doc, score).
        top_k:           Number of documents to return.
        embed_fn:        Callable(texts) -> np.ndarray [n, dim].
        precomputed_embs: Optional {doc_id -> embedding}.
        n_clusters:      Number of clusters; must be in [1, len(candidates)].
                         n_clusters=1 degenerates to relevance top-k.
        method:          "kmeans" or "agglomerative".
        seed:            Random seed for KMeans initialisation.

    Returns:
        List of (doc, norm_score) of length min(top_k, len(candidates)).

    Raises:
        ValueError: if parameters, scores, or embeddings are invalid.
    """
    if method not in ("kmeans", "agglomerative"):
        raise ValueError(f"method must be 'kmeans' or 'agglomerative', got {method!r}")
    if (
        isinstance(n_clusters, (bool, np.bool_))
        or not isinstance(n_clusters, (int, np.integer))
        or n_clusters < 1
    ):
        raise ValueError("n_clusters must be a positive integer")
    if (
        isinstance(top_k, (bool, np.bool_))
        or not isinstance(top_k, (int, np.integer))
        or top_k < 0
    ):
        raise ValueError("top_k must be a nonnegative integer")
    if n_clusters > len(candidates):
        raise ValueError("n_clusters cannot exceed the number of candidates")

    docs = [doc for doc, _ in candidates]
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

    top_k = min(top_k, len(docs))
    if top_k == 0:
        return []

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
    if embeddings.shape[0] != len(docs):
        raise ValueError("candidate embeddings must have one row per candidate")
    if embeddings.shape[1] == 0:
        raise ValueError("candidate embeddings must have a positive feature dimension")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("candidate embeddings must contain only finite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("candidate embeddings must have nonzero row norms")
    embeddings = embeddings / norms

    if n_clusters == 1:
        # Degenerate case: pure relevance ranking
        order = sorted(range(len(docs)), key=lambda i: (-norm_scores[i], i))
        return [(docs[i], float(norm_scores[i])) for i in order[:top_k]]

    # Cluster the candidate embeddings
    labels = _cluster(embeddings, n_clusters=n_clusters, method=method, seed=seed)
    unique_labels = np.unique(labels)
    if len(unique_labels) != n_clusters:
        raise RuntimeError(
            f"clustering produced {len(unique_labels)} effective clusters; "
            f"requested {n_clusters}"
        )

    # Build per-cluster queues sorted by relevance (best first)
    cluster_queues: Dict[int, List[int]] = {int(c): [] for c in unique_labels}
    for idx, label in enumerate(labels):
        cluster_queues[int(label)].append(idx)
    for c in cluster_queues:
        cluster_queues[c].sort(key=lambda i: (-norm_scores[i], i))

    # Label-invariant fixed priority: best-member relevance, then its original
    # candidate position. Numeric sklearn labels never determine output order.
    selected: List[int] = []
    cluster_order = sorted(
        cluster_queues,
        key=lambda c: (-norm_scores[cluster_queues[c][0]], cluster_queues[c][0]),
    )
    pos = {c: 0 for c in cluster_order}

    while len(selected) < top_k:
        made_progress = False
        for c in cluster_order:
            if len(selected) >= top_k:
                break
            queue = cluster_queues[c]
            while pos[c] < len(queue) and queue[pos[c]] in selected:
                pos[c] += 1
            if pos[c] < len(queue):
                selected.append(queue[pos[c]])
                pos[c] += 1
                made_progress = True
        if not made_progress:
            break

    return [(docs[i], float(norm_scores[i])) for i in selected[:top_k]]


def _cluster(
    embeddings: np.ndarray,
    n_clusters: int,
    method: str,
    seed: int,
) -> np.ndarray:
    """Return integer label array [n] using the requested clustering method."""
    if method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            random_state=seed,
            algorithm="lloyd",
            max_iter=300,
            tol=1e-4,
        )
        return km.fit_predict(embeddings)

    # method == "agglomerative"
    from sklearn.cluster import AgglomerativeClustering

    ag = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward",
        connectivity=None,
        compute_full_tree="auto",
        distance_threshold=None,
        compute_distances=False,
    )
    return ag.fit_predict(embeddings)
