"""HTTP API.

Indexing is deliberately not exposed here: documents are ingested from the command
line, so the service never reads a file path supplied by a client.
"""

import logging
import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from grounded_rag import __version__
from grounded_rag.config import SearchMode, get_settings
from grounded_rag.documents import Hit
from grounded_rag.generate import GenerationError
from grounded_rag.services import Services, build_services

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: SearchMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None


class HitOut(BaseModel):
    number: int
    score: float
    doc_id: str
    chunk_index: int
    title: str
    source: str
    text: str
    cited: bool = False


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    reranked: bool
    hits: list[HitOut]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: SearchMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    refused: bool
    sources: list[HitOut]


class HealthResponse(BaseModel):
    status: str
    version: str
    collection: str
    chunks: int


def create_app(services: Services | None = None) -> FastAPI:
    """Create the app. Tests pass ``services``; otherwise they are built from settings
    when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not hasattr(app.state, "services"):
            app.state.services = build_services(get_settings())
        yield

    app = FastAPI(title="grounded-rag", version=__version__, lifespan=lifespan)
    if services is not None:
        app.state.services = services

    def get_services(request: Request) -> Services:
        return request.app.state.services

    ServicesDep = Annotated[Services, Depends(get_services)]

    def require_api_key(
        services: ServicesDep, x_api_key: Annotated[str | None, Header()] = None
    ) -> None:
        expected = services.settings.api_key
        if expected is None:
            return
        if x_api_key is None or not secrets.compare_digest(
            x_api_key.encode(), expected.get_secret_value().encode()
        ):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key.")

    def retrieve(
        services: Services,
        query: str,
        mode: SearchMode | None,
        top_k: int | None,
        rerank: bool | None,
    ) -> tuple[list[Hit], SearchMode, bool]:
        settings = services.settings
        mode = mode or settings.search_mode
        rerank = settings.rerank if rerank is None else rerank
        try:
            hits = services.retriever.retrieve(
                query,
                mode=mode,
                top_k=top_k or settings.top_k,
                candidate_k=settings.candidate_k,
                rerank=rerank,
            )
        except ValueError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
        return hits, mode, rerank

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return files("grounded_rag").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health(services: ServicesDep) -> HealthResponse:
        try:
            chunks = services.store.count()
        except Exception as error:
            logger.exception("Vector store is unreachable")
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Vector store is unreachable."
            ) from error
        return HealthResponse(
            status="ok", version=__version__, collection=services.store.collection, chunks=chunks
        )

    @app.post("/search", dependencies=[Depends(require_api_key)])
    def search(body: SearchRequest, services: ServicesDep) -> SearchResponse:
        hits, mode, reranked = retrieve(services, body.query, body.mode, body.top_k, body.rerank)
        return SearchResponse(query=body.query, mode=mode, reranked=reranked, hits=_hits_out(hits))

    @app.post("/ask", dependencies=[Depends(require_api_key)])
    def ask(body: AskRequest, services: ServicesDep) -> AskResponse:
        hits, _, _ = retrieve(services, body.question, body.mode, body.top_k, body.rerank)
        try:
            answer = services.generator.answer(body.question, hits)
        except GenerationError as error:
            logger.warning("Answer generation failed: %s", error)
            code = (
                status.HTTP_503_SERVICE_UNAVAILABLE
                if error.retryable
                else status.HTTP_502_BAD_GATEWAY
            )
            raise HTTPException(code, str(error)) from error
        return AskResponse(
            question=body.question,
            answer=answer.text,
            model=answer.model,
            refused=answer.refused,
            sources=_hits_out(hits, cited=answer.cited),
        )

    return app


def _hits_out(hits: Sequence[Hit], cited: Sequence[int] = ()) -> list[HitOut]:
    return [
        HitOut(
            number=number,
            score=hit.score,
            doc_id=hit.chunk.doc_id,
            chunk_index=hit.chunk.index,
            title=hit.chunk.title,
            source=hit.chunk.source,
            text=hit.chunk.text,
            cited=number in cited,
        )
        for number, hit in enumerate(hits, start=1)
    ]
