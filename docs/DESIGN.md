# dd-vectordb Design Notes

## Motivation

`dd-vectordb` is a thin, opinionated abstraction layer over vector databases. The goals mirror those of its sibling `dd-db` (relational):

1. **Single consistent API** — swap backends without changing application code.
2. **Easy onboarding** — `InMemoryVectorDB` works with zero extra dependencies, enabling notebooks and tests to run anywhere.
3. **Production-ready adapters** — FAISS, ChromaDB, Qdrant cover the most common production use cases.

---

## Core Design Principles

### 1. Pre-computed embeddings

`dd-vectordb` does **not** embed text. Embeddings must be provided by the caller.

**Why?** Embedding models are a separate concern with their own latency, cost, and model-selection trade-offs. Keeping them separate makes `dd-vectordb` model-agnostic and avoids implicit coupling to any specific encoder (OpenAI, sentence-transformers, Ollama, etc.).

The `add_texts()` convenience method accepts both `texts` and `embeddings` lists — the caller is responsible for computing the embeddings before calling `add_texts`.

### 2. `Document` as the atomic unit

Every stored item is a `Document(id, text, embedding, metadata)`. Search returns `SearchResult(document, score, rank)`.

**Why?** A rich object model is more useful than raw tuples and avoids the need for callers to remember column ordering. Pydantic models provide free validation, serialisation, and IDE autocompletion.

### 3. Consistent parameter interface

Search always accepts:
- `query_vector: list[float] | np.ndarray`
- `k: int = 5`
- `filter: dict[str, Any] | None = None`

DML always returns a count:
- `delete(ids) -> int`

**Why?** Predictable signatures mean callers don't have to read adapter-specific docs.

### 4. Scores are "higher is better"

Regardless of the adapter's native metric:
- Cosine / inner product → score ∈ [-1, 1], higher is more similar.
- L2 / Euclidean → score is negated distance, higher is more similar.
- ChromaDB distances are converted to `1 - distance` for cosine.

**Why?** A uniform convention lets callers rank results without knowing which metric is in use.

### 5. Optional dependencies per adapter

```bash
pip install dd-vectordb                  # InMemoryVectorDB only (numpy)
pip install "dd-vectordb[faiss]"         # + FAISS
pip install "dd-vectordb[chroma]"        # + ChromaDB
pip install "dd-vectordb[qdrant]"        # + Qdrant
pip install "dd-vectordb[all]"           # everything
```

Adapters import their driver inside `__init__` and raise `ImportError` with a helpful install command if the driver is missing.

### 6. Context manager lifecycle

```python
with FAISSVectorDB(dimension=768) as db:
    db.add_texts(texts, embeddings)
    results = db.search(query, k=5)
# close() called automatically
```

### 7. No embedding dimension in base class

`BaseVectorDB` does not require a `dimension` parameter. Adapters that require a fixed dimension (FAISS, Qdrant) accept it in `__init__`. Adapters that infer dimension from the first vector (InMemoryVectorDB, ChromaDB) do not.

---

## Adapter Comparison

| Feature | InMemory | FAISS | ChromaDB | Qdrant |
|---------|----------|-------|----------|--------|
| Extra deps | none | faiss-cpu | chromadb | qdrant-client |
| Persistence | ❌ | ✅ (save/load) | ✅ | ✅ |
| Metadata filter | ✅ | ✅ (post-filter) | ✅ (native) | ✅ (native) |
| Scalability | ~10 k docs | millions | millions | billions |
| Approximate search | ❌ | optional (IVF) | HNSW | HNSW |
| Server required | ❌ | ❌ | ❌ | ❌ (in-memory) |

**InMemoryVectorDB** is brute-force cosine similarity over a numpy matrix. Linear scan, O(N·D) per query. Use for dev, testing, and small corpora.

**FAISSVectorDB** uses `IndexFlatIP` (exact inner product on L2-normalised vectors for cosine) or `IndexFlatL2`. Supports save/load to disk. For large corpora, swap in an IVF or HNSW index by overriding `_build_index()`.

**ChromaVectorDB** wraps ChromaDB's Python-embedded HNSW index. Ephemeral or persistent. Native metadata filtering. Good for rapid prototyping.

**QdrantVectorDB** wraps Qdrant — in-memory, on-disk, or remote server. Production-grade, supports rich payload filtering, scalar quantisation. Uses a hash-based ID mapping (string → uint64) because Qdrant requires integer or UUID point IDs.

---

## Adding a New Adapter

```python
from dd_vectordb.base import BaseVectorDB
from dd_vectordb.models import CollectionInfo, Document, SearchResult

class MyVectorDB(BaseVectorDB):
    def add_documents(self, documents): ...
    def search(self, query_vector, k=5, filter=None): ...
    def delete(self, ids): ...
    def clear(self): ...
    def count(self): ...
    def collection_info(self): ...
```

Implement all six abstract methods. The concrete helpers (`add_texts`, `get_by_ids`, `__enter__`, `__exit__`, `close`, `__repr__`) are provided by `BaseVectorDB`.

---

## Relationship to vanna.ai

[vanna.ai](https://github.com/vanna-ai/vanna) has an `AgentMemory` ABC that stores and retrieves documentation, DDL, and SQL pairs for Text2SQL prompting. Its interface is tightly coupled to vanna's internal `ToolContext`.

`dd-vectordb` has a broader scope (general-purpose vector search) and is standalone (no vanna dependency). It can be used as the underlying storage layer in a vanna-like system.

---

## File Layout

```
dd-vectordb/
├── pyproject.toml
├── README.md
├── docs/
│   └── DESIGN.md               # this file
├── src/
│   └── dd_vectordb/
│       ├── __init__.py         # public API
│       ├── base.py             # BaseVectorDB ABC
│       ├── models.py           # Document, SearchResult, CollectionInfo
│       └── adapters/
│           ├── __init__.py
│           ├── in_memory.py    # pure numpy, zero deps
│           ├── faiss_db.py     # FAISS flat index
│           ├── chroma_db.py    # ChromaDB embedded
│           └── qdrant_db.py    # Qdrant (local or remote)
├── tests/
│   ├── __init__.py
│   └── test_in_memory.py       # 30+ tests, no external deps
└── cookbook/
    ├── 01_in_memory_basics.py  # zero-dep walkthrough
    └── 02_faiss_basics.py      # FAISS with save/load
```

---

## Testing Strategy

- **Unit tests** (`tests/test_in_memory.py`) use `InMemoryVectorDB` — zero extra deps, run anywhere.
- **Integration tests** (not yet added) would require FAISS / ChromaDB / Qdrant installs and go in `tests/integration/`.
- **Cookbooks** serve as manual integration tests and learning materials.

---

## Known Limitations

1. **No built-in encoder** — by design. Plug in sentence-transformers, OpenAI embeddings, or any model.
2. **FAISS delete is expensive** — `IndexFlat` does not support in-place deletion; the entire index is rebuilt after each `delete()` call. Use a FAISS IVF index or soft-delete if this is a hot path.
3. **Qdrant ID mapping** — string IDs are hashed to uint64. Astronomically unlikely but theoretically possible hash collision would silently overwrite a document.
4. **ChromaDB filter syntax** — the `filter` dict is passed directly as Chroma's `where` clause, which supports only equality matches via this adapter. Complex boolean filters require using the Chroma client directly.
