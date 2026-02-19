"""
Tests for InMemoryVectorDB — no external dependencies required.
Uses only numpy (already a dependency).
"""

import pytest
import numpy as np

from dd_vectordb import InMemoryVectorDB, Document, SearchResult, CollectionInfo


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DIM = 4  # small dimension for fast tests


def make_embedding(seed: int) -> list[float]:
    """Reproducible unit-ish vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def db():
    """Fresh InMemoryVectorDB with 4 documents."""
    store = InMemoryVectorDB(name="test", metric="cosine")
    docs = [
        Document(id="doc1", text="apple orange fruit", embedding=make_embedding(1),
                 metadata={"category": "food", "lang": "en"}),
        Document(id="doc2", text="car engine motor", embedding=make_embedding(2),
                 metadata={"category": "auto", "lang": "en"}),
        Document(id="doc3", text="python programming code", embedding=make_embedding(3),
                 metadata={"category": "tech", "lang": "en"}),
        Document(id="doc4", text="pomme orange fruit", embedding=make_embedding(1),  # same as doc1
                 metadata={"category": "food", "lang": "fr"}),
    ]
    store.add_documents(docs)
    return store


# ------------------------------------------------------------------
# Basic operations
# ------------------------------------------------------------------

def test_count(db):
    assert db.count() == 4


def test_repr(db):
    r = repr(db)
    assert "InMemoryVectorDB" in r
    assert "count=4" in r


def test_collection_info(db):
    info = db.collection_info()
    assert isinstance(info, CollectionInfo)
    assert info.adapter == "InMemoryVectorDB"
    assert info.count == 4
    assert info.dimension == DIM
    assert info.metric == "cosine"
    assert info.name == "test"


# ------------------------------------------------------------------
# add_documents / upsert
# ------------------------------------------------------------------

def test_add_document(db):
    new_doc = Document(id="doc5", text="new document", embedding=make_embedding(5))
    db.add_documents([new_doc])
    assert db.count() == 5


def test_upsert_updates_existing(db):
    updated = Document(id="doc1", text="UPDATED", embedding=make_embedding(99))
    db.add_documents([updated])
    assert db.count() == 4  # no new row
    retrieved = db.get_by_ids(["doc1"])[0]
    assert retrieved.text == "UPDATED"


def test_add_without_embedding_raises(db):
    bad = Document(id="x", text="no embedding")
    with pytest.raises(ValueError, match="no embedding"):
        db.add_documents([bad])


# ------------------------------------------------------------------
# add_texts helper
# ------------------------------------------------------------------

def test_add_texts(db):
    ids = db.add_texts(
        texts=["foo", "bar"],
        embeddings=[make_embedding(10), make_embedding(11)],
    )
    assert len(ids) == 2
    assert db.count() == 6


def test_add_texts_with_explicit_ids(db):
    ids = db.add_texts(
        texts=["hello"],
        embeddings=[make_embedding(20)],
        ids=["custom-id"],
    )
    assert ids == ["custom-id"]
    doc = db.get_by_ids(["custom-id"])[0]
    assert doc.text == "hello"


def test_add_texts_length_mismatch():
    store = InMemoryVectorDB()
    with pytest.raises(ValueError, match="same length"):
        store.add_texts(texts=["a", "b"], embeddings=[make_embedding(1)])


# ------------------------------------------------------------------
# search
# ------------------------------------------------------------------

def test_search_returns_list(db):
    results = db.search(make_embedding(1), k=2)
    assert isinstance(results, list)
    assert len(results) == 2


def test_search_results_are_search_result(db):
    results = db.search(make_embedding(1), k=1)
    assert isinstance(results[0], SearchResult)


def test_search_rank_order(db):
    results = db.search(make_embedding(1), k=4)
    ranks = [r.rank for r in results]
    assert ranks == [1, 2, 3, 4]


def test_search_most_similar_first(db):
    # doc1 and doc4 share the same seed (identical embeddings)
    results = db.search(make_embedding(1), k=2)
    ids = {r.document.id for r in results}
    assert "doc1" in ids
    assert "doc4" in ids


def test_search_score_descending(db):
    results = db.search(make_embedding(1), k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_store():
    store = InMemoryVectorDB()
    results = store.search(make_embedding(1), k=5)
    assert results == []


def test_search_k_larger_than_count(db):
    results = db.search(make_embedding(1), k=100)
    assert len(results) == 4  # only 4 docs exist


# ------------------------------------------------------------------
# search with filter
# ------------------------------------------------------------------

def test_search_with_filter(db):
    results = db.search(make_embedding(1), k=5, filter={"lang": "fr"})
    assert len(results) == 1
    assert results[0].document.id == "doc4"


def test_search_with_filter_no_match(db):
    results = db.search(make_embedding(1), k=5, filter={"lang": "de"})
    assert results == []


# ------------------------------------------------------------------
# get_by_ids
# ------------------------------------------------------------------

def test_get_by_ids_found(db):
    docs = db.get_by_ids(["doc1", "doc3"])
    assert len(docs) == 2
    assert docs[0].id == "doc1"
    assert docs[1].id == "doc3"


def test_get_by_ids_missing_returns_none(db):
    docs = db.get_by_ids(["doc1", "nonexistent"])
    assert docs[0] is not None
    assert docs[1] is None


# ------------------------------------------------------------------
# delete
# ------------------------------------------------------------------

def test_delete(db):
    removed = db.delete(["doc1", "doc2"])
    assert removed == 2
    assert db.count() == 2


def test_delete_nonexistent_id(db):
    removed = db.delete(["ghost"])
    assert removed == 0
    assert db.count() == 4


def test_search_after_delete(db):
    db.delete(["doc1", "doc4"])
    results = db.search(make_embedding(1), k=10)
    ids = {r.document.id for r in results}
    assert "doc1" not in ids
    assert "doc4" not in ids


# ------------------------------------------------------------------
# clear
# ------------------------------------------------------------------

def test_clear(db):
    db.clear()
    assert db.count() == 0


def test_search_after_clear(db):
    db.clear()
    results = db.search(make_embedding(1), k=5)
    assert results == []


def test_add_after_clear(db):
    db.clear()
    doc = Document(id="new", text="after clear", embedding=make_embedding(7))
    db.add_documents([doc])
    assert db.count() == 1


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------

def test_context_manager():
    with InMemoryVectorDB() as store:
        store.add_texts(texts=["hello"], embeddings=[make_embedding(1)])
        assert store.count() == 1
    # No error means __exit__ worked fine
