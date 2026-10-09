"""The two pipelines: indexing documents and retrieving chunks for a query."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from grounded_rag.chunking import chunk_document
from grounded_rag.config import SearchMode
from grounded_rag.documents import Chunk, Document, Hit
from grounded_rag.embeddings import DenseEmbedder, SparseEmbedder
from grounded_rag.fusion import RRF_K
from grounded_rag.rerank import Reranker
from grounded_rag.store import QdrantStore


@dataclass(frozen=True, slots=True)
class IndexStats:
    documents: int
    chunks: int


class Indexer:
    def __init__(
        self,
        store: QdrantStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        *,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int = 64,
    ) -> None:
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    def index(
        self,
        documents: Iterable[Document],
        on_progress: Callable[[IndexStats], None] | None = None,
    ) -> IndexStats:
        """Chunk, embed and store documents.

        Chunks already stored for a document are deleted before its new chunks are
        written, so indexing is idempotent and a document that shrank leaves no
        stale chunks behind.
        """
        self.store.ensure_collection(self.dense.dim)
        documents_done = 0
        chunks_done = 0
        doc_ids: list[str] = []
        chunks: list[Chunk] = []

        def flush() -> None:
            nonlocal documents_done, chunks_done
            self._write(doc_ids, chunks)
            documents_done += len(doc_ids)
            chunks_done += len(chunks)
            doc_ids.clear()
            chunks.clear()
            if on_progress:
                on_progress(IndexStats(documents_done, chunks_done))

        for document in documents:
            # Batches always hold whole documents: the delete in _write must not
            # remove chunks of a document that an earlier batch has just written.
            doc_ids.append(document.doc_id)
            chunks.extend(chunk_document(document, self.chunk_size, self.chunk_overlap))
            if len(chunks) >= self.batch_size:
                flush()
        if doc_ids:
            flush()
        return IndexStats(documents_done, chunks_done)

    def _write(self, doc_ids: list[str], chunks: list[Chunk]) -> None:
        texts = [chunk.embedding_text for chunk in chunks]
        dense = self.dense.embed_documents(texts) if texts else []
        sparse = self.sparse.embed_documents(texts) if texts else []
        self.store.delete_documents(doc_ids)
        self.store.upsert(chunks, dense, sparse)


class Retriever:
    def __init__(
        self,
        store: QdrantStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        reranker: Reranker | None = None,
        *,
        rrf_k: int = RRF_K,
    ) -> None:
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        top_k: int = 5,
        candidate_k: int = 30,
        rerank: bool = False,
    ) -> list[Hit]:
        """Return the ``top_k`` chunks for ``query``.

        With ``rerank`` the store returns ``candidate_k`` chunks, the cross-encoder
        rescores them against the query, and the best ``top_k`` are kept.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if rerank and self.reranker is None:
            raise ValueError("rerank was requested but no reranker is configured")

        # Only compute the query vectors the chosen mode will use.
        dense = self.dense.embed_query(query) if mode != "sparse" else None
        sparse = self.sparse.embed_query(query) if mode != "dense" else None
        limit = max(top_k, candidate_k) if rerank else top_k
        hits = self.store.search(
            mode,
            dense=dense,
            sparse=sparse,
            limit=limit,
            candidate_k=candidate_k,
            rrf_k=self.rrf_k,
        )

        if rerank and hits:
            scores = self.reranker.score(query, [hit.chunk.embedding_text for hit in hits])
            ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
            hits = [Hit(chunk=hit.chunk, score=score) for hit, score in ranked]
        return hits[:top_k]
