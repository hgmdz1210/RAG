import pytest

from conftest import CORPUS, FakeDense, FakeSparse
from rag.documents import Document
from rag.ingestion.indexer import Indexer


def _search(store, mode, query, **kwargs):
    return store.search(
        mode,
        dense=FakeDense().embed_query(query),
        sparse=FakeSparse().embed_query(query),
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_each_mode_finds_the_matching_document(store, indexer, mode):
    indexer.index(CORPUS)

    hits = _search(store, mode, "term frequency and inverse document frequency", limit=3)

    assert hits[0].chunk.doc_id == "bm25.md"
    assert hits[0].chunk.title == "BM25"
    assert hits[0].chunk.source == "bm25.md"


def test_limit_caps_the_number_of_hits(store, indexer):
    indexer.index(CORPUS)

    assert len(_search(store, "hybrid", "vectors rankings passages", limit=2)) == 2


def test_sparse_search_weights_rare_terms_above_common_ones(store):
    """Without IDF the document repeating the common term would win on raw counts."""
    documents = [
        Document(doc_id="common", text="shared shared shared shared filler"),
        Document(doc_id="rare", text="unique shared"),
        *[Document(doc_id=f"pad{i}", text=f"shared padding{i}") for i in range(6)],
    ]
    Indexer(store, FakeDense(), FakeSparse(), chunk_size=400, chunk_overlap=0).index(documents)

    hits = _search(store, "sparse", "shared unique", limit=2)

    assert hits[0].chunk.doc_id == "rare"


def test_hybrid_ranks_a_document_found_by_both_signals_first(store, indexer):
    indexer.index(CORPUS)
    query = "reciprocal rank fusion merges rankings"

    dense_top = _search(store, "dense", query, limit=1)[0].chunk.doc_id
    sparse_top = _search(store, "sparse", query, limit=1)[0].chunk.doc_id
    hybrid_top = _search(store, "hybrid", query, limit=1)[0].chunk.doc_id

    assert dense_top == sparse_top == hybrid_top == "fusion.md"


def test_hybrid_falls_back_to_dense_when_the_query_has_no_sparse_terms(store, indexer):
    indexer.index(CORPUS)

    hits = store.search(
        "hybrid",
        dense=FakeDense().embed_query("cross encoder rescores candidates"),
        sparse=FakeSparse().embed_query(""),
        limit=1,
    )

    assert hits[0].chunk.doc_id == "rerank.md"


def test_sparse_search_with_no_terms_returns_nothing(store, indexer):
    indexer.index(CORPUS)

    assert store.search("sparse", sparse=FakeSparse().embed_query(""), limit=3) == []


def test_missing_query_vector_is_rejected(store, indexer):
    indexer.index(CORPUS)

    with pytest.raises(ValueError):
        store.search("dense", limit=1)
    with pytest.raises(ValueError):
        store.search("hybrid", dense=FakeDense().embed_query("x"), limit=1)


def test_count_is_zero_before_the_collection_exists(store):
    assert store.count() == 0


def test_dimension_mismatch_is_reported(store):
    store.ensure_collection(256)

    with pytest.raises(ValueError, match="256-dimensional"):
        store.ensure_collection(384)


def test_hybrid_returns_the_same_order_on_repeated_calls(store, indexer):
    indexer.index(CORPUS)
    query = "vectors passages rankings chunks"

    first = [hit.chunk.chunk_id for hit in _search(store, "hybrid", query, limit=5)]

    for _ in range(5):
        assert [hit.chunk.chunk_id for hit in _search(store, "hybrid", query, limit=5)] == first


def test_owner_filter_isolates_users_in_every_mode(store):
    indexer = Indexer(store, FakeDense(), FakeSparse(), chunk_size=400, chunk_overlap=0)
    text = "Quarterly revenue grew because subscriptions renewed."
    indexer.index(
        [
            Document(doc_id="alice-report", text=text, owner="alice"),
            Document(doc_id="bob-report", text=text, owner="bob"),
            Document(doc_id="shared-notes", text=text),
        ]
    )

    for mode in ("dense", "sparse", "hybrid"):
        mine = _search(store, mode, "quarterly revenue subscriptions", limit=10, owner="alice")
        assert [hit.chunk.doc_id for hit in mine] == ["alice-report"], mode
        assert mine[0].chunk.owner == "alice"
        everyone = _search(store, mode, "quarterly revenue subscriptions", limit=10)
        assert len(everyone) == 3, mode


def test_owner_with_no_documents_gets_nothing(store, indexer):
    indexer.index(CORPUS)

    assert _search(store, "hybrid", "vectors payload", limit=5, owner="nobody") == []
