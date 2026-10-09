"""Shared fixtures.

The fakes below stand in for the ONNX models so the suite needs no network and
no model downloads. They are deterministic bag-of-words embedders: crude, but
they rank by term overlap, which is enough to exercise the retrieval mechanics.
"""

import math
import re
import zlib
from collections import Counter
from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from rag.auth import User
from rag.config import Settings
from rag.db import create_tables, make_engine
from rag.documents import Document, Hit
from rag.embeddings import SparseVector
from rag.generate import Answer
from rag.library import Library
from rag.pipeline import Indexer, Retriever
from rag.services import Services
from rag.store import QdrantStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _term_id(term: str) -> int:
    # zlib.crc32 is stable across processes, unlike the built-in hash().
    return zlib.crc32(term.encode())


class FakeDense:
    dim = 256

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for term in tokenize(text):
            vector[_term_id(term) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FakeSparse:
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._embed(text, binary=False) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text, binary=True)

    @staticmethod
    def _embed(text: str, *, binary: bool) -> SparseVector:
        counts = Counter(tokenize(text))
        return SparseVector(
            indices=[_term_id(term) for term in counts],
            values=[1.0 if binary else float(count) for count in counts.values()],
        )


class FakeReranker:
    """Scores a text by how many query terms it contains."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        terms = set(tokenize(query))
        return [float(len(terms & set(tokenize(text)))) for text in texts]


CORPUS = [
    Document(
        doc_id="qdrant.md",
        title="Qdrant",
        source="qdrant.md",
        text="Qdrant stores dense vectors and sparse vectors with a payload for filtering.",
    ),
    Document(
        doc_id="bm25.md",
        title="BM25",
        source="bm25.md",
        text="BM25 ranks passages by term frequency and inverse document frequency.",
    ),
    Document(
        doc_id="rerank.md",
        title="Reranking",
        source="rerank.md",
        text="A cross encoder rescores candidate passages against the question.",
    ),
    Document(
        doc_id="fusion.md",
        title="Fusion",
        source="fusion.md",
        text="Reciprocal rank fusion merges two rankings using only the ranks.",
    ),
    Document(
        doc_id="chunking.md",
        title="Chunking",
        source="chunking.md",
        text="Long files are split into overlapping chunks before they are embedded.",
    ),
]


class FakeGenerator:
    """Cites the first source, or raises ``error`` when one is set."""

    def __init__(self) -> None:
        self.error: Exception | None = None

    def answer(self, question: str, hits: Sequence[Hit]) -> Answer:
        if self.error is not None:
            raise self.error
        if not hits:
            return Answer(text="No sources.")
        return Answer(text="Based on the first passage [1].", cited=[1], model="fake-model")


class MemoryStorage:
    """File storage kept in a dict; set ``fail`` to make writes raise."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.fail: Exception | None = None

    def put(self, key: str, data: bytes, content_type: str) -> None:
        if self.fail is not None:
            raise self.fail
        self.files[key] = data

    def delete(self, key: str) -> None:
        self.files.pop(key, None)


class FakeAuth:
    """Accepts the tokens ``alice-token`` and ``bob-token``."""

    users = {
        "alice-token": User(id="alice", email="alice@example.com"),
        "bob-token": User(id="bob", email="bob@example.com"),
    }

    def verify(self, token: str) -> User | None:
        return self.users.get(token)


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"), "test")


@pytest.fixture
def indexer(store: QdrantStore) -> Indexer:
    return Indexer(store, FakeDense(), FakeSparse(), chunk_size=400, chunk_overlap=0)


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def retriever(store: QdrantStore, indexer: Indexer, reranker: FakeReranker) -> Retriever:
    indexer.index(CORPUS)
    return Retriever(store, FakeDense(), FakeSparse(), reranker)


@pytest.fixture
def services(store: QdrantStore, indexer: Indexer, reranker: FakeReranker) -> Services:
    """Fully wired services over an empty collection and an empty database."""
    engine = make_engine("sqlite://")
    create_tables(engine)
    library = Library(engine, MemoryStorage(), indexer, store, max_upload_bytes=1_000_000)
    return Services(
        settings=Settings(_env_file=None),
        store=store,
        indexer=indexer,
        retriever=Retriever(store, FakeDense(), FakeSparse(), reranker),
        generator=FakeGenerator(),
        library=library,
    )


@pytest.fixture
def indexed_services(services: Services) -> Services:
    services.indexer.index(CORPUS)
    return services
