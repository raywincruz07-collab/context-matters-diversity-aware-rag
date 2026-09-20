"""
Abstract base class for all retrievers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseRetriever(ABC):
    """Common interface for all retrievers."""

    def __init__(self, name: str):
        self.name = name
        self.corpus = None
        self.is_indexed = False

    @abstractmethod
    def index(self, corpus: List[Dict]):
        """Build index from corpus documents."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k documents for a query.

        Returns:
            List of (document_dict, relevance_score) tuples, sorted by score descending.
        """
        pass

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Tuple[Dict, float]]]:
        """Retrieve for multiple queries."""
        return [self.retrieve(q, top_k) for q in queries]
