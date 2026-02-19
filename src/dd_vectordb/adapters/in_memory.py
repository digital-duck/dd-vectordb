"""
In-memory vector store — zero extra dependencies (numpy only).

Uses brute-force cosine similarity. Suitable for development, testing,
and small collections (< ~10 000 documents).

Example::

    from dd_vectordb.adapters.in_memory import InMemoryVectorDB

    db = InMemoryVectorDB()
    db.add_texts(
        texts=["hello world", "foo bar"],
        embeddings=[[0.1, 0.9], [0.8, 0.2]],
    )
    results = db.search([0.1, 0.9], k=1)
    print(results[0].document.text)  # "hello world"
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from dd_vectordb.base import BaseVectorDB
from dd_vectordb.models import CollectionInfo, Document, SearchResult


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector *a* and each row of *b*."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    return (b / b_norms) @ a_norm


class InMemoryVectorDB(BaseVectorDB):
    """Pure-Python / NumPy in-memory vector store.

    Args:
        name:    Logical name for the collection (informational only).
        metric:  Distance metric — only ``"cosine"`` is supported.
    """

    def __init__(self, name: str = "default", metric: str = "cosine") -> None:
        self._name = name
        self._metric = metric
        self._docs: Dict[str, Document] = {}     # id → Document
        self._matrix: Optional[np.ndarray] = None  # shape (N, D)
        self._ids: List[str] = []                # parallel to matrix rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_matrix(self) -> None:
        """Rebuild the numpy embedding matrix from the document store."""
        if not self._docs:
            self._matrix = None
            self._ids = []
            return
        self._ids = list(self._docs.keys())
        self._matrix = np.array(
            [self._docs[i].embedding for i in self._ids], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # BaseVectorDB interface
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(
                    f"Document id={doc.id!r} has no embedding. "
                    "Set doc.embedding before calling add_documents()."
                )
            self._docs[doc.id] = doc
        self._rebuild_matrix()

    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self._matrix is None or len(self._docs) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)

        if filter:
            # Build a sub-matrix of docs that pass the filter
            filtered_ids = [
                doc_id
                for doc_id, doc in self._docs.items()
                if all(doc.metadata.get(k) == v for k, v in filter.items())
            ]
            if not filtered_ids:
                return []
            sub_matrix = np.array(
                [self._docs[i].embedding for i in filtered_ids], dtype=np.float32
            )
            scores = _cosine_similarity(q, sub_matrix)
            top_k = min(k, len(filtered_ids))
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [
                SearchResult(
                    document=self._docs[filtered_ids[i]],
                    score=float(scores[i]),
                    rank=rank + 1,
                )
                for rank, i in enumerate(top_idx)
            ]

        scores = _cosine_similarity(q, self._matrix)
        top_k = min(k, len(self._ids))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                document=self._docs[self._ids[i]],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_idx)
        ]

    def delete(self, ids: List[str]) -> int:
        removed = 0
        for doc_id in ids:
            if doc_id in self._docs:
                del self._docs[doc_id]
                removed += 1
        self._rebuild_matrix()
        return removed

    def clear(self) -> None:
        self._docs.clear()
        self._matrix = None
        self._ids = []

    def count(self) -> int:
        return len(self._docs)

    def get_by_ids(self, ids: List[str]) -> List[Optional[Document]]:
        return [self._docs.get(doc_id) for doc_id in ids]

    def collection_info(self) -> CollectionInfo:
        dim = self._matrix.shape[1] if self._matrix is not None else None
        return CollectionInfo(
            name=self._name,
            adapter="InMemoryVectorDB",
            count=len(self._docs),
            dimension=dim,
            metric=self._metric,
        )
