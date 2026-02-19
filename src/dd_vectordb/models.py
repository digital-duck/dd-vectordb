"""
Data models for dd-vectordb.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document stored in the vector database.

    Attributes:
        id:        Unique identifier (auto-assigned if not provided).
        text:      The raw text content.
        embedding: The dense vector representation (set after encoding).
        metadata:  Arbitrary key-value metadata for filtering.
    """

    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        dim = len(self.embedding) if self.embedding else "none"
        meta = f", metadata={self.metadata}" if self.metadata else ""
        return (
            f"Document(id={self.id!r}, text={self.text[:40]!r}..., "
            f"dim={dim}{meta})"
        )


class SearchResult(BaseModel):
    """A single result from a vector similarity search.

    Attributes:
        document: The matched document.
        score:    Similarity score (higher is more similar for cosine/dot;
                  lower is more similar for L2 distance — adapter-dependent).
        rank:     1-based rank in the result list.
    """

    document: Document
    score: float
    rank: int

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"id={self.document.id!r}, text={self.document.text[:40]!r}...)"
        )


class CollectionInfo(BaseModel):
    """Summary of a collection / index.

    Attributes:
        name:          Collection name.
        adapter:       Adapter class name (e.g. ``"InMemoryVectorDB"``).
        count:         Number of documents stored.
        dimension:     Vector dimension (None if collection is empty).
        metric:        Distance metric (``"cosine"``, ``"l2"``, ``"dot"``).
        extra:         Adapter-specific metadata.
    """

    name: str
    adapter: str
    count: int
    dimension: Optional[int] = None
    metric: str = "cosine"
    extra: Dict[str, Any] = Field(default_factory=dict)
