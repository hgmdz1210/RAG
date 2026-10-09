"""Qdrant collection holding one dense and one sparse vector per chunk."""

import warnings
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from grounded_rag.config import SearchMode
from grounded_rag.documents import Chunk, Hit
from grounded_rag.embeddings import SparseVector

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def make_client(url: str | None, path: str) -> QdrantClient:
    """Connect to a Qdrant server, or run Qdrant embedded on a local directory.

    Embedded mode needs no server but locks its directory, so only one process
    can use it at a time.
    """
    if url:
        return QdrantClient(url=url, timeout=30)
    return QdrantClient(path=path)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def ensure_collection(self, dense_dim: int) -> None:
        """Create the collection if missing, and reject an existing one whose dense
        vector size does not match the configured embedding model."""
        if self.client.collection_exists(self.collection):
            vectors = self.client.get_collection(self.collection).config.params.vectors
            existing = vectors[DENSE_VECTOR].size
            if existing != dense_dim:
                raise ValueError(
                    f"Collection '{self.collection}' stores {existing}-dimensional vectors but "
                    f"the embedding model produces {dense_dim}. Re-index into a new collection."
                )
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            # The IDF modifier makes Qdrant weight each query term by its inverse
            # document frequency, which turns stored term frequencies into BM25.
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        with warnings.catch_warnings():
            # Embedded Qdrant ignores payload indexes and warns about it.
            warnings.simplefilter("ignore", UserWarning)
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def drop(self) -> None:
        self.client.delete_collection(self.collection)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        """Remove every chunk of the given documents."""
        if not doc_ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchAny(any=list(doc_ids))
                        )
                    ]
                )
            ),
            wait=True,
        )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        if not (len(chunks) == len(dense) == len(sparse)):
            raise ValueError("chunks, dense and sparse must have the same length")
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE_VECTOR: list(dense_vector),
                    SPARSE_VECTOR: _to_qdrant(sparse_vector),
                },
                payload={
                    "doc_id": chunk.doc_id,
                    "index": chunk.index,
                    "text": chunk.text,
                    "title": chunk.title,
                    "source": chunk.source,
                },
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self,
        mode: SearchMode,
        *,
        dense: Sequence[float] | None = None,
        sparse: SparseVector | None = None,
        limit: int = 10,
        candidate_k: int = 30,
    ) -> list[Hit]:
        """Return the top ``limit`` chunks.

        In hybrid mode each signal contributes its top ``candidate_k`` chunks and the
        two rankings are merged with reciprocal rank fusion. RRF uses only ranks, so
        cosine similarities and BM25 scores never have to be put on a common scale.
        """
        has_sparse = sparse is not None and bool(sparse.indices)
        if mode == "dense":
            if dense is None:
                raise ValueError("dense search needs a dense query vector")
            response = self.client.query_points(
                self.collection, query=list(dense), using=DENSE_VECTOR, limit=limit
            )
        elif mode == "sparse":
            if sparse is None:
                raise ValueError("sparse search needs a sparse query vector")
            if not has_sparse:
                # The query had no indexable terms (for example only stop words).
                return []
            response = self.client.query_points(
                self.collection, query=_to_qdrant(sparse), using=SPARSE_VECTOR, limit=limit
            )
        elif mode == "hybrid":
            if dense is None or sparse is None:
                raise ValueError("hybrid search needs both a dense and a sparse query vector")
            depth = max(limit, candidate_k)
            prefetch = [models.Prefetch(query=list(dense), using=DENSE_VECTOR, limit=depth)]
            if has_sparse:
                prefetch.append(
                    models.Prefetch(query=_to_qdrant(sparse), using=SPARSE_VECTOR, limit=depth)
                )
            response = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )
        else:
            raise ValueError(f"unknown search mode: {mode!r}")
        return [_to_hit(point) for point in response.points]


def _to_qdrant(vector: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices, values=vector.values)


def _to_hit(point: models.ScoredPoint) -> Hit:
    payload = point.payload or {}
    chunk = Chunk(
        chunk_id=str(point.id),
        doc_id=payload["doc_id"],
        index=payload["index"],
        text=payload["text"],
        title=payload.get("title", ""),
        source=payload.get("source", ""),
    )
    return Hit(chunk=chunk, score=float(point.score))
