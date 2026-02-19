"""
dd-vectordb adapter registry.

Adapters with optional dependencies raise ImportError at instantiation time
(not at import time) so that the base package works without any extras.
"""

from dd_vectordb.adapters.in_memory import InMemoryVectorDB

# Optional adapters — imported here for discoverability
from dd_vectordb.adapters.faiss_db import FAISSVectorDB
from dd_vectordb.adapters.chroma_db import ChromaVectorDB
from dd_vectordb.adapters.qdrant_db import QdrantVectorDB

__all__ = [
    "InMemoryVectorDB",
    "FAISSVectorDB",
    "ChromaVectorDB",
    "QdrantVectorDB",
]
