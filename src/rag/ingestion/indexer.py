"""Index documents: chunk them, embed the chunks and write them to the store."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rag.documents import Chunk, Document
from rag.ingestion.chunking import chunk_document
from rag.retrieval.embeddings import DenseEmbedder, SparseEmbedder
from rag.retrieval.store import QdrantStore


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
