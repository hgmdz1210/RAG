import pytest

from conftest import CORPUS, FakeDense, FakeSparse
from rag.documents import Document
from rag.ingestion.indexer import Indexer, IndexStats
from rag.retrieval.retriever import Retriever


def test_index_reports_documents_and_chunks(store, indexer):
    stats = indexer.index(CORPUS)

    assert stats == IndexStats(documents=len(CORPUS), chunks=len(CORPUS))
    assert store.count() == len(CORPUS)


def test_indexing_twice_does_not_duplicate_chunks(store, indexer):
    indexer.index(CORPUS)
    indexer.index(CORPUS)

    assert store.count() == len(CORPUS)


def test_reindexing_a_shorter_document_removes_stale_chunks(store):
    indexer = Indexer(store, FakeDense(), FakeSparse(), chunk_size=40, chunk_overlap=0)
    long_text = " ".join(f"Sentence {i} has a few words." for i in range(10))
    indexer.index([Document(doc_id="doc", text=long_text)])
    assert store.count() > 1

    indexer.index([Document(doc_id="doc", text="Now it is short.")])

    assert store.count() == 1


def test_reindexing_one_document_leaves_the_others_alone(store, indexer):
    indexer.index(CORPUS)

    indexer.index([Document(doc_id="bm25.md", text="Replaced text about lexical scoring.")])

    assert store.count() == len(CORPUS)


def test_small_batches_keep_every_chunk(store):
    indexer = Indexer(
        store, FakeDense(), FakeSparse(), chunk_size=40, chunk_overlap=0, batch_size=2
    )
    long_text = " ".join(f"Sentence {i} has a few words." for i in range(10))
    documents = [Document(doc_id=f"doc{i}", text=long_text) for i in range(3)]

    stats = indexer.index(documents)

    assert stats.documents == 3
    assert store.count() == stats.chunks


def test_progress_callback_reports_running_totals(indexer):
    seen: list[IndexStats] = []

    stats = indexer.index(CORPUS, on_progress=seen.append)

    assert seen[-1] == stats


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_retrieve_returns_top_k_in_each_mode(retriever, mode):
    hits = retriever.retrieve("sparse vectors and dense vectors payload", mode=mode, top_k=2)

    assert len(hits) == 2
    assert hits[0].chunk.doc_id == "qdrant.md"


def test_rerank_reorders_candidates_by_reranker_score(retriever, reranker):
    query = "cross encoder question"

    hits = retriever.retrieve(query, mode="dense", top_k=2, candidate_k=5, rerank=True)

    assert hits[0].chunk.doc_id == "rerank.md"
    assert hits[0].score == 3.0
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    # The reranker saw the full candidate pool, not just the final top_k.
    (call_query, call_texts) = reranker.calls[0]
    assert call_query == query
    assert len(call_texts) == 5


def test_rerank_is_skipped_unless_requested(retriever, reranker):
    retriever.retrieve("anything about vectors", mode="dense", top_k=2)

    assert reranker.calls == []


def test_rerank_without_a_reranker_is_rejected(store, indexer):
    indexer.index(CORPUS)
    retriever = Retriever(store, FakeDense(), FakeSparse(), reranker=None)

    with pytest.raises(ValueError, match="no reranker"):
        retriever.retrieve("vectors", rerank=True)


def test_empty_query_is_rejected(retriever):
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("   ")


def test_retriever_passes_its_fusion_constant_to_the_store(store, indexer, monkeypatch):
    indexer.index(CORPUS)
    seen = {}
    original = store.search

    def spy(mode, **kwargs):
        seen.update(kwargs)
        return original(mode, **kwargs)

    monkeypatch.setattr(store, "search", spy)

    Retriever(store, FakeDense(), FakeSparse(), rrf_k=7).retrieve("vectors", mode="hybrid")

    assert seen["rrf_k"] == 7
