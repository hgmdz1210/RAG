"""Retrieve the chunks for a query, optionally reranked."""

from rag.config import SearchMode
from rag.documents import Hit
from rag.retrieval.embeddings import DenseEmbedder, SparseEmbedder
from rag.retrieval.fusion import RRF_K
from rag.retrieval.rerank import Reranker
from rag.retrieval.store import QdrantStore


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
        owner: str | None = None,
    ) -> list[Hit]:
        """Return the ``top_k`` chunks for ``query``.

        With ``rerank`` the store returns ``candidate_k`` chunks, the cross-encoder
        rescores them against the query, and the best ``top_k`` are kept. With
        ``owner`` the search is restricted to that user's documents.
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
            owner=owner,
        )

        if rerank and hits:
            scores = self.reranker.score(query, [hit.chunk.embedding_text for hit in hits])
            ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
            hits = [Hit(chunk=hit.chunk, score=score) for hit, score in ranked]
        return hits[:top_k]
