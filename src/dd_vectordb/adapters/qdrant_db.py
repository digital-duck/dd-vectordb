"""
Qdrant adapter for dd-vectordb.

Requires: ``pip install "dd-vectordb[qdrant]"``

Qdrant is a production-grade vector database with advanced filtering,
payload indexing, and both in-memory and persistent modes.

Example::

    from dd_vectordb.adapters.qdrant_db import QdrantVectorDB

    # In-memory (great for dev/test)
    db = QdrantVectorDB(dimension=768, collection_name="docs")

    # On-disk
    db = QdrantVectorDB(dimension=768, collection_name="docs", path="./qdrant_data")

    # Remote server
    db = QdrantVectorDB(dimension=768, collection_name="docs",
                        url="http://localhost:6333")
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from dd_vectordb.base import BaseVectorDB
from dd_vectordb.models import CollectionInfo, Document, SearchResult


class QdrantVectorDB(BaseVectorDB):
    """Qdrant-backed vector store.

    Args:
        dimension:       Embedding dimensionality.
        collection_name: Qdrant collection name (created if absent).
        metric:          ``"cosine"``, ``"l2"`` (Euclidean), or ``"dot"``.
        url:             Qdrant server URL. If ``None``, uses in-memory client.
        path:            Path for on-disk storage. Ignored if *url* is set.
        api_key:         API key for Qdrant Cloud.
    """

    _METRIC_MAP = {
        "cosine": "Cosine",
        "l2": "Euclid",
        "dot": "Dot",
    }

    def __init__(
        self,
        dimension: int,
        collection_name: str = "default",
        metric: str = "cosine",
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantVectorDB. "
                "Install with: pip install \"dd-vectordb[qdrant]\""
            )
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._dimension = dimension
        self._metric = metric
        self._collection_name = collection_name

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(":memory:")

        distance = getattr(Distance, self._METRIC_MAP.get(metric, "Cosine").upper())
        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=distance),
            )

    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=_str_to_uint64(doc.id),
                vector=list(map(float, doc.embedding)),
                payload={"__text__": doc.text, "__id__": doc.id, **doc.metadata},
            )
            for doc in documents
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        qdrant_filter = None
        if filter:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        hits = self._client.search(
            collection_name=self._collection_name,
            query_vector=list(map(float, query_vector)),
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True,
        )
        results = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            text = payload.pop("__text__", "")
            doc_id = payload.pop("__id__", str(hit.id))
            results.append(SearchResult(
                document=Document(
                    id=doc_id,
                    text=text,
                    embedding=list(hit.vector) if hit.vector else None,
                    metadata=payload,
                ),
                score=float(hit.score),
                rank=rank,
            ))
        return results

    def delete(self, ids: List[str]) -> int:
        from qdrant_client.models import PointIdsList
        uint_ids = [_str_to_uint64(i) for i in ids]
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=uint_ids),
        )
        return len(ids)  # Qdrant doesn't return a deletion count

    def clear(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        self._client.delete_collection(self._collection_name)
        distance_name = self._METRIC_MAP.get(self._metric, "Cosine").upper()
        from qdrant_client.models import Distance
        distance = getattr(Distance, distance_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._dimension, distance=distance),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    def close(self) -> None:
        self._client.close()

    def collection_info(self) -> CollectionInfo:
        count = self.count()
        return CollectionInfo(
            name=self._collection_name,
            adapter="QdrantVectorDB",
            count=count,
            dimension=self._dimension,
            metric=self._metric,
        )


def _str_to_uint64(s: str) -> int:
    """Convert a string ID to a stable uint64 for Qdrant.

    Qdrant requires integer or UUID point IDs. We hash the string ID.
    Collision risk is negligible for practical collection sizes.
    """
    import hashlib
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
