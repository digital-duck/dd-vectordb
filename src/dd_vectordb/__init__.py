"""
dd-vectordb — Unified Vector DB abstraction layer.

Quick start::

    from dd_vectordb import InMemoryVectorDB

    db = InMemoryVectorDB()
    db.add_texts(
        texts=["The quick brown fox", "Hello world"],
        embeddings=[[0.1, 0.9, 0.3], [0.8, 0.2, 0.5]],
    )
    results = db.search([0.1, 0.9, 0.3], k=1)
    print(results[0].document.text)  # "The quick brown fox"

Public API
----------
- :class:`BaseVectorDB`    — abstract base, subclass to add new adapters
- :class:`InMemoryVectorDB`— pure NumPy, zero extra deps
- :class:`FAISSVectorDB`   — FAISS (requires ``dd-vectordb[faiss]``)
- :class:`ChromaVectorDB`  — ChromaDB (requires ``dd-vectordb[chroma]``)
- :class:`QdrantVectorDB`  — Qdrant (requires ``dd-vectordb[qdrant]``)

Models
------
- :class:`Document`        — text + embedding + metadata
- :class:`SearchResult`    — matched document + similarity score + rank
- :class:`CollectionInfo`  — collection metadata summary
"""

__version__ = "0.1.2"

from dd_vectordb.base import BaseVectorDB
from dd_vectordb.models import CollectionInfo, Document, SearchResult
from dd_vectordb.adapters.in_memory import InMemoryVectorDB
from dd_vectordb.adapters.faiss_db import FAISSVectorDB
from dd_vectordb.adapters.chroma_db import ChromaVectorDB
from dd_vectordb.adapters.qdrant_db import QdrantVectorDB

__all__ = [
    "__version__",
    # Base
    "BaseVectorDB",
    # Models
    "Document",
    "SearchResult",
    "CollectionInfo",
    # Adapters
    "InMemoryVectorDB",
    "FAISSVectorDB",
    "ChromaVectorDB",
    "QdrantVectorDB",
]
