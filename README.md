# dd-vectordb

**Unified Vector DB abstraction layer for Python.**

Add semantic search to any project in minutes. Swap backends (in-memory, FAISS, ChromaDB, Qdrant) without changing your application code.

## Supported Backends

| Adapter | Class | Extra | Notes |
|---------|-------|-------|-------|
| In-memory (NumPy) | `InMemoryVectorDB` | *(none)* | Brute-force cosine; dev/testing |
| FAISS | `FAISSVectorDB` | `faiss` | Facebook AI; exact + ANN |
| ChromaDB | `ChromaVectorDB` | `chroma` | Embedded HNSW; persistent |
| Qdrant | `QdrantVectorDB` | `qdrant` | Production-grade; local/remote |

## Install

```bash
pip install dd-vectordb                  # InMemoryVectorDB only (numpy)
pip install "dd-vectordb[faiss]"         # + FAISS
pip install "dd-vectordb[chroma]"        # + ChromaDB
pip install "dd-vectordb[qdrant]"        # + Qdrant
pip install "dd-vectordb[all]"           # all adapters
pip install "dd-vectordb[dev]"           # dev tools
```

## Quick Start

```python
import numpy as np
from dd_vectordb import InMemoryVectorDB

# 1. Embed your texts (any encoder — OpenAI, sentence-transformers, Ollama, etc.)
texts = ["The quick brown fox", "Python programming", "Vector search rocks"]
embeddings = [np.random.rand(768).tolist() for _ in texts]  # replace with real embeddings

# 2. Add to the store
db = InMemoryVectorDB()
db.add_texts(texts=texts, embeddings=embeddings)

# 3. Search
query_vec = np.random.rand(768).tolist()  # replace with real query embedding
results = db.search(query_vec, k=2)
for r in results:
    print(f"#{r.rank}  score={r.score:.4f}  {r.document.text}")
```

## API Reference

### Core methods (all adapters)

| Method | Returns | Description |
|--------|---------|-------------|
| `add_documents(docs)` | `None` | Add/upsert `Document` objects |
| `add_texts(texts, embeddings, ids?, metadatas?)` | `list[str]` | Convenience: build Documents and add |
| `search(query_vector, k=5, filter?)` | `list[SearchResult]` | Top-k similarity search |
| `delete(ids)` | `int` | Delete by ID; returns count removed |
| `clear()` | `None` | Remove all documents |
| `count()` | `int` | Number of documents stored |
| `get_by_ids(ids)` | `list[Document | None]` | Retrieve by ID |
| `collection_info()` | `CollectionInfo` | Name, count, dimension, metric |
| `close()` | `None` | Release resources |

### Context manager

```python
with FAISSVectorDB(dimension=768) as db:
    db.add_texts(texts, embeddings)
    results = db.search(query, k=5)
# close() called automatically
```

### Pydantic models

```python
from dd_vectordb import Document, SearchResult, CollectionInfo

doc = Document(id="1", text="hello", embedding=[0.1, 0.9], metadata={"src": "wiki"})
result: SearchResult  # .document, .score, .rank
info: CollectionInfo  # .name, .adapter, .count, .dimension, .metric
```

## Examples

### With FAISS

```python
from dd_vectordb import FAISSVectorDB

db = FAISSVectorDB(dimension=768, metric="cosine")
db.add_texts(texts=["hello world"], embeddings=[[...768 floats...]])
results = db.search([...768 floats...], k=5)

# Persist to disk
db.save("my_index.faiss")
db2 = FAISSVectorDB.load("my_index.faiss")
```

### With ChromaDB (persistent)

```python
from dd_vectordb import ChromaVectorDB

db = ChromaVectorDB(collection_name="my_docs", persist_directory="./chroma_data")
db.add_texts(texts=["hello"], embeddings=[[0.1, 0.9]])
results = db.search([0.1, 0.9], k=1)
```

### With Qdrant (in-memory)

```python
from dd_vectordb import QdrantVectorDB

db = QdrantVectorDB(dimension=768, collection_name="docs")
db.add_texts(texts=["hello"], embeddings=[[...768 floats...]])
results = db.search([...768 floats...], k=5)
```

### Metadata filtering

```python
db.add_texts(
    texts=["wiki article", "blog post"],
    embeddings=[emb1, emb2],
    metadatas=[{"source": "wiki"}, {"source": "blog"}],
)
# Only search within wiki documents
results = db.search(query_vec, k=5, filter={"source": "wiki"})
```

## Cookbooks

See `cookbook/` for runnable examples:

- `01_in_memory_basics.py` — full walkthrough with zero extra deps
- `02_faiss_basics.py` — FAISS with save/load, metadata filtering

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

Tests use `InMemoryVectorDB` — no external server or extra install required.

## Design

See `docs/DESIGN.md` for:
- Why pre-computed embeddings?
- Adapter comparison table
- Score normalisation convention
- How to add a new adapter

## License

MIT
