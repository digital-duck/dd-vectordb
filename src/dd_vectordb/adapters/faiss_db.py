"""
FAISS vector store adapter.

Requires: ``pip install "dd-vectordb[faiss]"``

FAISS (Facebook AI Similarity Search) provides highly optimised nearest-
neighbour search with flat (exact) and approximate index types.

Example::

    from dd_vectordb.adapters.faiss_db import FAISSVectorDB

    db = FAISSVectorDB(dimension=768, metric="cosine")
    db.add_texts(texts=["hello"], embeddings=[[...768 floats...]])
    results = db.search([...768 floats...], k=5)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from dd_vectordb.base import BaseVectorDB
from dd_vectordb.models import CollectionInfo, Document, SearchResult


class FAISSVectorDB(BaseVectorDB):
    """FAISS-backed vector store.

    Args:
        dimension: Embedding dimensionality. Must be set before the first add.
        metric:    ``"cosine"`` (inner-product on normalised vectors) or
                   ``"l2"`` (Euclidean distance, lower = more similar).
        name:      Logical collection name.
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        name: str = "default",
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSVectorDB. "
                "Install with: pip install \"dd-vectordb[faiss]\""
            )
        self._dimension = dimension
        self._metric = metric
        self._name = name
        self._docs: Dict[str, Document] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx: int = 0
        self._index = self._build_index()

    def _build_index(self):
        import faiss
        if self._metric == "cosine":
            return faiss.IndexFlatIP(self._dimension)  # inner product on L2-normalised
        return faiss.IndexFlatL2(self._dimension)

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        """Normalise for cosine; cast to float32."""
        vectors = vectors.astype(np.float32)
        if self._metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
            vectors = vectors / norms
        return vectors

    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        vecs = np.array([doc.embedding for doc in documents], dtype=np.float32)
        vecs = self._prepare(vecs)
        self._index.add(vecs)
        for doc in documents:
            self._docs[doc.id] = doc
            self._id_to_idx[doc.id] = self._next_idx
            self._idx_to_id[self._next_idx] = doc.id
            self._next_idx += 1

    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self._index.ntotal == 0:
            return []
        q = np.array([query_vector], dtype=np.float32)
        q = self._prepare(q)
        k_search = min(k, self._index.ntotal)
        scores, indices = self._index.search(q, k_search)
        results = []
        rank = 1
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self._idx_to_id.get(idx)
            if doc_id is None:
                continue
            doc = self._docs[doc_id]
            if filter and not all(doc.metadata.get(k) == v for k, v in filter.items()):
                continue
            # Normalise: cosine/IP → higher is better (already so);
            # L2 distance → negate so higher score still means more similar.
            final_score = float(score) if self._metric == "cosine" else -float(score)
            results.append(SearchResult(document=doc, score=final_score, rank=rank))
            rank += 1
            if rank > k:
                break
        return results

    def delete(self, ids: List[str]) -> int:
        # FAISS flat index doesn't support in-place delete; rebuild without deleted docs.
        to_keep = [doc for doc_id, doc in self._docs.items() if doc_id not in set(ids)]
        removed = len(self._docs) - len(to_keep)
        self.clear()
        if to_keep:
            self.add_documents(to_keep)
        return removed

    def clear(self) -> None:
        self._docs.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._next_idx = 0
        self._index = self._build_index()

    def count(self) -> int:
        return self._index.ntotal

    def get_by_ids(self, ids: List[str]) -> List[Optional[Document]]:
        return [self._docs.get(doc_id) for doc_id in ids]

    def collection_info(self) -> CollectionInfo:
        return CollectionInfo(
            name=self._name,
            adapter="FAISSVectorDB",
            count=self._index.ntotal,
            dimension=self._dimension,
            metric=self._metric,
        )

    def save(self, path: str) -> None:
        """Persist the FAISS index to *path* (e.g. ``"index.faiss"``)."""
        import faiss, pickle, os
        faiss.write_index(self._index, path)
        meta_path = path + ".meta"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "docs": self._docs,
                "id_to_idx": self._id_to_idx,
                "idx_to_id": self._idx_to_id,
                "next_idx": self._next_idx,
                "dimension": self._dimension,
                "metric": self._metric,
                "name": self._name,
            }, f)

    @classmethod
    def load(cls, path: str) -> "FAISSVectorDB":
        """Load a previously saved index from *path*."""
        import faiss, pickle
        index = faiss.read_index(path)
        with open(path + ".meta", "rb") as f:
            meta = pickle.load(f)
        db = cls.__new__(cls)
        db._dimension = meta["dimension"]
        db._metric = meta["metric"]
        db._name = meta["name"]
        db._docs = meta["docs"]
        db._id_to_idx = meta["id_to_idx"]
        db._idx_to_id = meta["idx_to_id"]
        db._next_idx = meta["next_idx"]
        db._index = index
        return db
