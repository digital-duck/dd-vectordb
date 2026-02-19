"""
ChromaDB adapter for dd-vectordb.

Requires: ``pip install "dd-vectordb[chroma]"``

ChromaDB is a developer-friendly embedded vector database that can run
in-process (ephemeral or persistent) or as a client/server.

Example::

    from dd_vectordb.adapters.chroma_db import ChromaVectorDB

    # Ephemeral (in-memory)
    db = ChromaVectorDB(collection_name="my_docs")

    # Persistent
    db = ChromaVectorDB(collection_name="my_docs", persist_directory="./chroma_data")

    db.add_texts(texts=["hello"], embeddings=[[0.1, 0.9]])
    results = db.search([0.1, 0.9], k=1)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from dd_vectordb.base import BaseVectorDB
from dd_vectordb.models import CollectionInfo, Document, SearchResult


class ChromaVectorDB(BaseVectorDB):
    """ChromaDB-backed vector store.

    Args:
        collection_name:   Name of the Chroma collection.
        persist_directory: If set, data is persisted to this path.
                           If ``None``, data lives in memory only.
        metric:            ``"cosine"`` or ``"l2"`` (``"ip"`` also supported).
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        metric: str = "cosine",
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorDB. "
                "Install with: pip install \"dd-vectordb[chroma]\""
            )
        import chromadb

        self._collection_name = collection_name
        self._metric = metric

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": metric},
        )

    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[doc.id for doc in documents],
            documents=[doc.text for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        n_results = min(k, self._collection.count())
        if n_results == 0:
            return []
        kwargs: Dict[str, Any] = {
            "query_embeddings": [list(map(float, query_vector))],
            "n_results": n_results,
            "include": ["documents", "embeddings", "metadatas", "distances"],
        }
        if filter:
            kwargs["where"] = filter
        result = self._collection.query(**kwargs)

        results = []
        for rank, (doc_id, text, emb, meta, dist) in enumerate(zip(
            result["ids"][0],
            result["documents"][0],
            result["embeddings"][0],
            result["metadatas"][0],
            result["distances"][0],
        ), start=1):
            # Chroma returns distance (lower = more similar); convert to score
            score = 1.0 - dist if self._metric == "cosine" else -dist
            results.append(SearchResult(
                document=Document(id=doc_id, text=text, embedding=list(emb), metadata=meta),
                score=score,
                rank=rank,
            ))
        return results

    def delete(self, ids: List[str]) -> int:
        before = self._collection.count()
        self._collection.delete(ids=ids)
        after = self._collection.count()
        return before - after

    def clear(self) -> None:
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def count(self) -> int:
        return self._collection.count()

    def get_by_ids(self, ids: List[str]) -> List[Optional[Document]]:
        result = self._collection.get(
            ids=ids, include=["documents", "embeddings", "metadatas"]
        )
        id_set = dict(zip(result["ids"], range(len(result["ids"]))))
        out = []
        for doc_id in ids:
            if doc_id in id_set:
                i = id_set[doc_id]
                out.append(Document(
                    id=doc_id,
                    text=result["documents"][i],
                    embedding=list(result["embeddings"][i]),
                    metadata=result["metadatas"][i],
                ))
            else:
                out.append(None)
        return out

    def close(self) -> None:
        pass  # EphemeralClient / PersistentClient has no explicit close

    def collection_info(self) -> CollectionInfo:
        count = self._collection.count()
        return CollectionInfo(
            name=self._collection_name,
            adapter="ChromaVectorDB",
            count=count,
            metric=self._metric,
        )
