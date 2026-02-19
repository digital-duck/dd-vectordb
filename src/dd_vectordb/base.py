"""
Abstract base class for all dd-vectordb adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from dd_vectordb.models import CollectionInfo, Document, SearchResult


class BaseVectorDB(ABC):
    """Abstract base for vector database adapters.

    Subclass this and implement the abstract methods to add a new backend.
    The concrete helpers (``__enter__``, ``__exit__``, ``add_texts``) are
    provided and do not need to be overridden.

    Typical lifecycle::

        with MyVectorDB(...) as db:
            db.add_documents([Document(id="1", text="...", embedding=[...])])
            results = db.search(query_vector, k=5)
    """

    # ------------------------------------------------------------------
    # Abstract — must be implemented by every adapter
    # ------------------------------------------------------------------

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add (or upsert) documents.

        Each document must have a pre-computed ``embedding``.
        If a document with the same ``id`` already exists it is overwritten.
        """

    @abstractmethod
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for the *k* most similar documents.

        Args:
            query_vector: Dense query embedding.
            k:            Number of results to return.
            filter:       Optional metadata equality filter, e.g.
                          ``{"source": "wiki"}``. Adapters that do not
                          support filtering should ignore this argument.

        Returns:
            List of :class:`SearchResult` ordered by descending similarity.
        """

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """Delete documents by id.

        Returns:
            Number of documents actually deleted.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove **all** documents from the store."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents currently stored."""

    @abstractmethod
    def collection_info(self) -> CollectionInfo:
        """Return a summary of the collection / index."""

    # ------------------------------------------------------------------
    # Concrete helpers — free to all adapters
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[Union[List[float], np.ndarray]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Convenience wrapper: build Documents and call add_documents.

        Args:
            texts:      Raw text strings.
            embeddings: Pre-computed dense vectors, one per text.
            ids:        Optional explicit IDs; auto-generated if omitted.
            metadatas:  Optional metadata dicts, one per text.

        Returns:
            List of document IDs that were added.
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length.")
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        docs = [
            Document(
                id=doc_id,
                text=text,
                embedding=list(map(float, emb)),
                metadata=meta,
            )
            for doc_id, text, emb, meta in zip(ids, texts, embeddings, metadatas)
        ]
        self.add_documents(docs)
        return ids

    def get_by_ids(self, ids: List[str]) -> List[Optional[Document]]:
        """Retrieve documents by ID.

        Default implementation performs a linear scan. Adapters with native
        get-by-ID support should override this for efficiency.

        Returns:
            List aligned with input *ids*. Missing documents are ``None``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_by_ids()."
        )

    def __enter__(self) -> "BaseVectorDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release any resources held by the adapter.

        The default implementation is a no-op. Adapters with persistent
        connections should override this.
        """

    def __repr__(self) -> str:
        info = self.collection_info()
        return (
            f"{info.adapter}(name={info.name!r}, count={info.count}, "
            f"dim={info.dimension}, metric={info.metric!r})"
        )
