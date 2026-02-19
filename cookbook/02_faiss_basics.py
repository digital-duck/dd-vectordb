"""
Cookbook 02 — FAISSVectorDB Basics
=====================================
Demonstrates dd-vectordb with FAISS — highly optimised, production-grade
nearest-neighbour search from Meta Research.

Install:
    pip install "dd-vectordb[faiss]"

Run:
    python cookbook/02_faiss_basics.py
"""

import numpy as np
from dd_vectordb import FAISSVectorDB, Document

DIM = 64  # realistic-ish dimension for demo


def fake_embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ------------------------------------------------------------------
# 1. Create a FAISS store (cosine similarity)
# ------------------------------------------------------------------
print("=== 1. Create FAISSVectorDB (cosine) ===")

db = FAISSVectorDB(dimension=DIM, metric="cosine", name="demo")

corpus = [
    ("doc-001", "The stock market reached an all-time high today", "finance"),
    ("doc-002", "Interest rates were raised by the central bank", "finance"),
    ("doc-003", "The soccer team won the championship", "sports"),
    ("doc-004", "Machine learning beats humans at chess", "tech"),
    ("doc-005", "Neural networks power modern AI systems", "tech"),
    ("doc-006", "The athlete broke the 100-metre world record", "sports"),
    ("doc-007", "Inflation is at a 40-year high", "finance"),
    ("doc-008", "A new protein-folding model was released", "science"),
]

docs = [
    Document(id=d_id, text=text, embedding=fake_embed(text), metadata={"topic": topic})
    for d_id, text, topic in corpus
]
db.add_documents(docs)
print(repr(db))

# ------------------------------------------------------------------
# 2. Search
# ------------------------------------------------------------------
print("\n=== 2. Search: 'economic recession' ===")
results = db.search(fake_embed("economic recession"), k=3)
for r in results:
    print(f"  #{r.rank}  score={r.score:.4f}  [{r.document.metadata['topic']}]  {r.document.text}")

# ------------------------------------------------------------------
# 3. Metadata filter
# ------------------------------------------------------------------
print("\n=== 3. Filtered search (topic=tech) ===")
results = db.search(fake_embed("artificial intelligence"), k=5, filter={"topic": "tech"})
for r in results:
    print(f"  #{r.rank}  score={r.score:.4f}  {r.document.text}")

# ------------------------------------------------------------------
# 4. Save and load index
# ------------------------------------------------------------------
import tempfile, os

print("\n=== 4. Save / load ===")
with tempfile.TemporaryDirectory() as tmpdir:
    index_path = os.path.join(tmpdir, "demo.faiss")
    db.save(index_path)
    print(f"Saved index to {index_path}")

    loaded = FAISSVectorDB.load(index_path)
    print(f"Loaded index: count={loaded.count()}")

    # Results should be identical
    r_orig = db.search(fake_embed("AI"), k=2)
    r_load = loaded.search(fake_embed("AI"), k=2)
    assert r_orig[0].document.id == r_load[0].document.id
    print("Results consistent after load ✓")

# ------------------------------------------------------------------
# 5. Delete + rebuild
# ------------------------------------------------------------------
print("\n=== 5. Delete ===")
removed = db.delete(["doc-001", "doc-002"])
print(f"Removed {removed} docs. Count now: {db.count()}")

print("\n=== 6. Clear ===")
db.clear()
print(f"After clear: {db.count()}")
