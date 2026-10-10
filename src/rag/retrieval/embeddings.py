"""Dense and sparse text embedders.

The fastembed-backed classes run ONNX models on CPU and load them on first use,
so importing this module is cheap and the tests can substitute fakes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class DenseEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class FastEmbedDense:
    """Sentence-embedding model. ``query_prefix`` is prepended to queries only, for
    models trained with an asymmetric instruction."""

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        cache_dir: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    @cached_property
    def _model(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)

    @property
    def dim(self) -> int:
        return int(self._model.embedding_size)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.embed(list(texts), batch_size=self.batch_size)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        (vector,) = self._model.embed([self.query_prefix + text])
        return vector.tolist()


class FastEmbedSparse:
    """BM25 term weights. Documents carry saturated term frequencies and queries a
    weight of one per term; the inverse document frequency is applied by Qdrant at
    query time, so it always reflects the current collection."""

    def __init__(
        self, model_name: str, *, cache_dir: str | None = None, batch_size: int = 64
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    @cached_property
    def _model(self) -> Any:
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        embeddings = self._model.embed(list(texts), batch_size=self.batch_size)
        return [_to_sparse_vector(embedding) for embedding in embeddings]

    def embed_query(self, text: str) -> SparseVector:
        (embedding,) = self._model.query_embed(text)
        return _to_sparse_vector(embedding)


def _to_sparse_vector(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=[int(index) for index in embedding.indices],
        values=[float(value) for value in embedding.values],
    )
