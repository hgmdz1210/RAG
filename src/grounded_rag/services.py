"""Build the concrete pipeline objects from settings."""

from dataclasses import dataclass

from grounded_rag.config import Settings
from grounded_rag.embeddings import FastEmbedDense, FastEmbedSparse
from grounded_rag.generate import GeminiGenerator, Generator
from grounded_rag.pipeline import Indexer, Retriever
from grounded_rag.rerank import FastEmbedReranker
from grounded_rag.store import QdrantStore, make_client


@dataclass(slots=True)
class Services:
    settings: Settings
    store: QdrantStore
    indexer: Indexer
    retriever: Retriever
    generator: Generator


def build_services(settings: Settings, *, collection: str | None = None) -> Services:
    """Wire up the pipeline. Models are not loaded until something uses them."""
    cache_dir = settings.model_cache_dir
    dense = FastEmbedDense(
        settings.dense_model, query_prefix=settings.dense_query_prefix, cache_dir=cache_dir
    )
    sparse = FastEmbedSparse(settings.sparse_model, cache_dir=cache_dir)
    reranker = FastEmbedReranker(settings.rerank_model, cache_dir=cache_dir)

    client = make_client(settings.qdrant_url, settings.qdrant_path)
    store = QdrantStore(client, collection or settings.collection)

    api_key = settings.gemini_api_key
    return Services(
        settings=settings,
        store=store,
        indexer=Indexer(
            store,
            dense,
            sparse,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        retriever=Retriever(store, dense, sparse, reranker, rrf_k=settings.rrf_k),
        generator=GeminiGenerator(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            api_key=api_key.get_secret_value() if api_key else None,
        ),
    )
