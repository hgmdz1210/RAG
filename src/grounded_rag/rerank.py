"""Cross-encoder reranking.

A cross-encoder reads the query and a passage together, so it can judge relevance
more precisely than comparing two independently computed vectors. It is too slow
to run over a whole corpus, which is why it only rescores retrieved candidates.
"""

from collections.abc import Sequence
from functools import cached_property
from typing import Any, Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one relevance score per text; higher means more relevant."""
        ...


class FastEmbedReranker:
    def __init__(
        self, model_name: str, *, cache_dir: str | None = None, batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    @cached_property
    def _model(self) -> Any:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self.model_name, cache_dir=self.cache_dir)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self._model.rerank(query, list(texts), batch_size=self.batch_size)
        return [float(score) for score in scores]
