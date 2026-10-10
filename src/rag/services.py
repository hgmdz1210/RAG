"""Build the concrete pipeline objects from settings."""

from dataclasses import dataclass
from pathlib import Path

from rag.config import Settings
from rag.generation.answer import GeminiGenerator, Generator
from rag.ingestion.indexer import Indexer
from rag.library.auth import Authenticator, SupabaseAuthenticator
from rag.library.db import create_tables, make_engine
from rag.library.service import Library
from rag.library.storage import FileStorage, LocalStorage, SupabaseStorage
from rag.retrieval.embeddings import FastEmbedDense, FastEmbedSparse
from rag.retrieval.rerank import FastEmbedReranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.store import QdrantStore, make_client


@dataclass(slots=True)
class Services:
    settings: Settings
    store: QdrantStore
    indexer: Indexer
    retriever: Retriever
    generator: Generator
    library: Library
    # None means sign-in is not configured and there is a single local user.
    auth: Authenticator | None = None


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

    indexer = Indexer(
        store,
        dense,
        sparse,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    engine = make_engine(settings.database_url)
    if engine.dialect.name == "sqlite":
        # A local SQLite file needs no migration step; PostgreSQL is migrated
        # explicitly with `rag migrate`.
        if engine.url.database and engine.url.database != ":memory:":
            Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
        create_tables(engine)

    api_key = settings.gemini_api_key
    return Services(
        settings=settings,
        store=store,
        indexer=indexer,
        library=Library(
            engine,
            _make_storage(settings),
            indexer,
            store,
            max_upload_bytes=settings.max_upload_mb * 1_000_000,
        ),
        auth=_make_auth(settings),
        retriever=Retriever(store, dense, sparse, reranker, rrf_k=settings.rrf_k),
        generator=GeminiGenerator(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            api_key=api_key.get_secret_value() if api_key else None,
        ),
    )


def _make_auth(settings: Settings) -> Authenticator | None:
    if settings.supabase_url and settings.supabase_anon_key:
        return SupabaseAuthenticator(
            settings.supabase_url, settings.supabase_anon_key.get_secret_value()
        )
    return None


def _make_storage(settings: Settings) -> FileStorage:
    if settings.supabase_url and settings.supabase_service_role_key:
        return SupabaseStorage(
            settings.supabase_url,
            settings.supabase_service_role_key.get_secret_value(),
            settings.storage_bucket,
        )
    return LocalStorage(settings.upload_dir)
