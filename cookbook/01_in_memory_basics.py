"""
Cookbook 01 — InMemoryVectorDB Basics
======================================
Demonstrates core dd-vectordb operations using the built-in in-memory store.
Zero extra dependencies — only numpy is required.

Run:
    python cookbook/01_in_memory_basics.py
"""

import numpy as np
from dd_vectordb import InMemoryVectorDB, Document

# ------------------------------------------------------------------
# Helper: tiny fake embeddings (real usage would use a sentence encoder)
# ------------------------------------------------------------------

def fake_embed(text: str, dim: int = 8) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ------------------------------------------------------------------
# 1. Create a store and add documents
# ------------------------------------------------------------------
print("=== 1. Create store + add documents ===")

db = InMemoryVectorDB(name="cookbook-demo", metric="cosine")

texts = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a great programming language",
    "Machine learning models need lots of data",
    "Natural language processing with transformers",
    "DuckDB is fast for analytical SQL queries",
    "Vector databases power semantic search",
    "Cats and dogs are popular household pets",
    "The fox ran across the field at dawn",
]

categories = ["animals", "tech", "ml", "nlp", "db", "db", "animals", "animals"]

ids = db.add_texts(
    texts=texts,
    embeddings=[fake_embed(t) for t in texts],
    metadatas=[{"category": c} for c in categories],
)
print(f"Added {db.count()} documents")
print(f"IDs: {ids[:3]}... (auto-generated UUIDs)")
print(repr(db))

# ------------------------------------------------------------------
# 2. Semantic search
# ------------------------------------------------------------------
print("\n=== 2. Semantic search (top-3) ===")
query = "fox running in nature"
results = db.search(fake_embed(query), k=3)
for r in results:
    print(f"  #{r.rank}  score={r.score:.4f}  [{r.document.metadata['category']}]  {r.document.text}")

# ------------------------------------------------------------------
# 3. Search with metadata filter
# ------------------------------------------------------------------
print("\n=== 3. Filtered search (category=db) ===")
results_db = db.search(fake_embed("fast data query"), k=5, filter={"category": "db"})
for r in results_db:
    print(f"  #{r.rank}  score={r.score:.4f}  {r.document.text}")

# ------------------------------------------------------------------
# 4. Explicit document IDs + metadata
# ------------------------------------------------------------------
print("\n=== 4. Add documents with explicit IDs ===")
db.add_texts(
    texts=["New article about penguins", "New article about SQL"],
    embeddings=[fake_embed("New article about penguins"), fake_embed("New article about SQL")],
    ids=["article-001", "article-002"],
    metadatas=[{"category": "animals", "source": "manual"},
               {"category": "db", "source": "manual"}],
)
print(f"Store now has {db.count()} documents")

# ------------------------------------------------------------------
# 5. Retrieve by ID
# ------------------------------------------------------------------
print("\n=== 5. Get by IDs ===")
docs = db.get_by_ids(["article-001", "nonexistent-id"])
print(f"article-001: {docs[0].text}")
print(f"nonexistent: {docs[1]}")  # None

# ------------------------------------------------------------------
# 6. Upsert (overwrite existing)
# ------------------------------------------------------------------
print("\n=== 6. Upsert ===")
db.add_texts(
    texts=["Updated: penguins live in Antarctica"],
    embeddings=[fake_embed("Updated: penguins live in Antarctica")],
    ids=["article-001"],
)
updated = db.get_by_ids(["article-001"])[0]
print(f"Updated text: {updated.text}")
print(f"Count still: {db.count()}")

# ------------------------------------------------------------------
# 7. Delete documents
# ------------------------------------------------------------------
print("\n=== 7. Delete ===")
removed = db.delete(["article-001", "article-002"])
print(f"Removed {removed} documents. Count now: {db.count()}")

# ------------------------------------------------------------------
# 8. Collection info
# ------------------------------------------------------------------
print("\n=== 8. Collection info ===")
info = db.collection_info()
print(f"Name: {info.name}")
print(f"Adapter: {info.adapter}")
print(f"Count: {info.count}")
print(f"Dimension: {info.dimension}")
print(f"Metric: {info.metric}")

# ------------------------------------------------------------------
# 9. Context manager
# ------------------------------------------------------------------
print("\n=== 9. Context manager ===")
with InMemoryVectorDB(name="temp") as temp_db:
    temp_db.add_texts(
        texts=["temporary document"],
        embeddings=[fake_embed("temporary document")],
    )
    print(f"Inside context: count={temp_db.count()}")
print("Outside context: connection closed automatically")

# ------------------------------------------------------------------
# 10. Clear
# ------------------------------------------------------------------
print("\n=== 10. Clear ===")
db.clear()
print(f"After clear: {db.count()} documents")
