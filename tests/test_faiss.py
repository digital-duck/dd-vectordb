"""
Tests for FAISSVectorDB — requires faiss-cpu (pip install dd-vectordb[faiss]).
Uses small in-memory indexes so tests are fast and self-contained.

FAISS-specific behaviours tested beyond the InMemory baseline:
- cosine and L2 metric initialisation
- save() / load() round-trip
- delete() rebuilds index correctly
- get_by_ids()
- collection_info() reports correct dimension
"""

import os
import tempfile

import numpy as np
import pytest

from dd_vectordb import FAISSVectorDB, Document, SearchResult, CollectionInfo


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

DIM = 8


def make_embedding(seed: int, dim: int = DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def db():
    """Fresh FAISSVectorDB (cosine) with 4 documents."""
    store = FAISSVectorDB(dimension=DIM, metric="cosine", name="test")
    docs = [
        Document(id="doc1", text="apple orange fruit",    embedding=make_embedding(1),
                 metadata={"category": "food",  "lang": "en"}),
        Document(id="doc2", text="car engine motor",      embedding=make_embedding(2),
                 metadata={"category": "auto",  "lang": "en"}),
        Document(id="doc3", text="python programming",    embedding=make_embedding(3),
                 metadata={"category": "tech",  "lang": "en"}),
        Document(id="doc4", text="pomme orange fruit",    embedding=make_embedding(1),
                 metadata={"category": "food",  "lang": "fr"}),
    ]
    store.add_documents(docs)
    return store


# ------------------------------------------------------------------ #
# Basic operations                                                     #
# ------------------------------------------------------------------ #

def test_count(db):
    assert db.count() == 4


def test_repr(db):
    r = repr(db)
    assert "FAISSVectorDB" in r
    assert "count=4" in r


def test_collection_info(db):
    info = db.collection_info()
    assert isinstance(info, CollectionInfo)
    assert info.adapter == "FAISSVectorDB"
    assert info.count == 4
    assert info.dimension == DIM
    assert info.metric == "cosine"
    assert info.name == "test"


# ------------------------------------------------------------------ #
# add_documents / upsert                                               #
# ------------------------------------------------------------------ #

def test_add_document(db):
    new_doc = Document(id="doc5", text="new", embedding=make_embedding(5))
    db.add_documents([new_doc])
    assert db.count() == 5


def test_add_without_embedding_raises():
    store = FAISSVectorDB(dimension=DIM)
    bad = Document(id="x", text="no embedding")
    with pytest.raises((ValueError, TypeError)):
        store.add_documents([bad])


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


# ------------------------------------------------------------------ #
# search                                                               #
# ------------------------------------------------------------------ #

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
    # doc1 and doc4 have the same embedding (seed=1) — both should top the list
    results = db.search(make_embedding(1), k=2)
    ids = {r.document.id for r in results}
    assert "doc1" in ids
    assert "doc4" in ids


def test_search_scores_descending(db):
    results = db.search(make_embedding(1), k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_store():
    store = FAISSVectorDB(dimension=DIM)
    assert store.search(make_embedding(1), k=5) == []


def test_search_k_larger_than_count(db):
    results = db.search(make_embedding(1), k=100)
    assert len(results) == 4


def test_search_numpy_query(db):
    q = np.array(make_embedding(1), dtype=np.float32)
    results = db.search(q, k=2)
    assert len(results) == 2


# ------------------------------------------------------------------ #
# search with metadata filter                                          #
# ------------------------------------------------------------------ #

def test_search_with_filter(db):
    results = db.search(make_embedding(1), k=5, filter={"lang": "fr"})
    assert len(results) == 1
    assert results[0].document.id == "doc4"


def test_search_with_filter_no_match(db):
    results = db.search(make_embedding(1), k=5, filter={"lang": "de"})
    assert results == []


def test_search_filter_category(db):
    results = db.search(make_embedding(1), k=5, filter={"category": "food"})
    ids = {r.document.id for r in results}
    assert ids == {"doc1", "doc4"}


# ------------------------------------------------------------------ #
# get_by_ids                                                           #
# ------------------------------------------------------------------ #

def test_get_by_ids_found(db):
    docs = db.get_by_ids(["doc1", "doc3"])
    assert docs[0].id == "doc1"
    assert docs[1].id == "doc3"


def test_get_by_ids_missing_returns_none(db):
    docs = db.get_by_ids(["doc1", "ghost"])
    assert docs[0] is not None
    assert docs[1] is None


# ------------------------------------------------------------------ #
# delete                                                               #
# ------------------------------------------------------------------ #

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


def test_index_consistent_after_delete(db):
    """After delete+rebuild, remaining docs are still searchable."""
    db.delete(["doc2", "doc3"])
    results = db.search(make_embedding(1), k=5)
    ids = {r.document.id for r in results}
    assert ids == {"doc1", "doc4"}


# ------------------------------------------------------------------ #
# clear                                                                #
# ------------------------------------------------------------------ #

def test_clear(db):
    db.clear()
    assert db.count() == 0


def test_search_after_clear(db):
    db.clear()
    assert db.search(make_embedding(1)) == []


def test_add_after_clear(db):
    db.clear()
    doc = Document(id="new", text="after clear", embedding=make_embedding(7))
    db.add_documents([doc])
    assert db.count() == 1


# ------------------------------------------------------------------ #
# L2 metric                                                            #
# ------------------------------------------------------------------ #

def test_l2_metric_initialises():
    store = FAISSVectorDB(dimension=DIM, metric="l2")
    store.add_texts(
        texts=["hello", "world"],
        embeddings=[make_embedding(1), make_embedding(2)],
    )
    results = store.search(make_embedding(1), k=2)
    assert len(results) == 2


def test_l2_nearest_is_highest_score():
    """For L2 we negate distance, so higher score = closer."""
    store = FAISSVectorDB(dimension=DIM, metric="l2")
    store.add_texts(
        texts=["a", "b"],
        embeddings=[make_embedding(1), make_embedding(99)],
    )
    results = store.search(make_embedding(1), k=2)
    assert results[0].score > results[1].score
    # The closest doc should be the one with the same seed
    assert results[0].document.text == "a"


# ------------------------------------------------------------------ #
# save / load                                                          #
# ------------------------------------------------------------------ #

def test_save_and_load(db):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "index.faiss")
        db.save(path)
        assert os.path.exists(path)
        assert os.path.exists(path + ".meta")

        loaded = FAISSVectorDB.load(path)
        assert loaded.count() == db.count()
        assert loaded.collection_info().dimension == DIM


def test_search_results_consistent_after_load(db):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "index.faiss")
        db.save(path)
        loaded = FAISSVectorDB.load(path)

        q = make_embedding(1)
        orig = db.search(q, k=2)
        reloaded = loaded.search(q, k=2)
        assert orig[0].document.id == reloaded[0].document.id
        assert abs(orig[0].score - reloaded[0].score) < 1e-5


def test_metadata_preserved_after_load(db):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "index.faiss")
        db.save(path)
        loaded = FAISSVectorDB.load(path)

        docs = loaded.get_by_ids(["doc1"])
        assert docs[0].metadata["category"] == "food"
        assert docs[0].text == "apple orange fruit"


# ------------------------------------------------------------------ #
# Context manager                                                      #
# ------------------------------------------------------------------ #

def test_context_manager():
    with FAISSVectorDB(dimension=DIM) as store:
        store.add_texts(texts=["hello"], embeddings=[make_embedding(1)])
        assert store.count() == 1
