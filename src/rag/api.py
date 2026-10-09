"""HTTP API.

Documents reach the service as uploaded bytes. No route accepts a file path, so
the service never reads a location chosen by a client.
"""

import logging
import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.resources import files
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag import __version__
from rag.auth import LOCAL_USER, AuthUnavailable, User
from rag.config import SearchMode, get_settings
from rag.db import DocumentRecord, MessageRecord
from rag.documents import Hit
from rag.generate import GenerationError
from rag.library import DuplicateDocument, UploadRejected
from rag.services import Services, build_services

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
    answerable: bool
    sources: list[HitOut]


class DocumentOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    chunks: int
    status: str
    error: str | None
    created_at: datetime


class MessageOut(BaseModel):
    id: str
    question: str
    answer: str
    model: str
    answerable: bool
    sources: list[dict[str, Any]]
    created_at: datetime


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

    app = FastAPI(title="rag", version=__version__, lifespan=lifespan)
    if services is not None:
        app.state.services = services

    def get_services(request: Request) -> Services:
        return request.app.state.services

    ServicesDep = Annotated[Services, Depends(get_services)]

    def current_user(
        services: ServicesDep,
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> User:
        """Resolve who is calling. With sign-in configured this is the user behind
        the bearer token; otherwise a single local user, optionally behind a
        shared API key."""
        if services.auth is None:
            expected = services.settings.api_key
            if expected is not None and (
                x_api_key is None
                or not secrets.compare_digest(
                    x_api_key.encode(), expected.get_secret_value().encode()
                )
            ):
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key.")
            return LOCAL_USER

        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to use this service.")
        try:
            user = services.auth.verify(token.strip())
        except AuthUnavailable as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
        if user is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "The session is invalid or has expired."
            )
        return user

    UserDep = Annotated[User, Depends(current_user)]

    def retrieve(
        services: Services,
        user: User,
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
                # Signed-in users search only their own documents. Without
                # sign-in there is one user and the whole collection is theirs.
                owner=user.id if services.auth is not None else None,
            )
        except ValueError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
        return hits, mode, rerank

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return files("rag").joinpath("static/index.html").read_text(encoding="utf-8")

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

    @app.post("/search")
    def search(body: SearchRequest, services: ServicesDep, user: UserDep) -> SearchResponse:
        hits, mode, reranked = retrieve(
            services, user, body.query, body.mode, body.top_k, body.rerank
        )
        return SearchResponse(query=body.query, mode=mode, reranked=reranked, hits=_hits_out(hits))

    @app.post("/ask")
    def ask(body: AskRequest, services: ServicesDep, user: UserDep) -> AskResponse:
        hits, _, _ = retrieve(services, user, body.question, body.mode, body.top_k, body.rerank)
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
        try:
            services.library.record_answer(user, body.question, answer, hits)
        except Exception:
            # History is a convenience; losing one entry must not lose the answer.
            logger.exception("Could not save the answer to the history")
        return AskResponse(
            question=body.question,
            answer=answer.text,
            model=answer.model,
            refused=answer.refused,
            answerable=answer.answerable,
            sources=_hits_out(hits, cited=answer.cited),
        )

    @app.post("/documents", status_code=status.HTTP_201_CREATED)
    def upload_document(file: UploadFile, services: ServicesDep, user: UserDep) -> DocumentOut:
        limit = services.library.max_upload_bytes
        # Read one byte past the limit: enough to know the file is too large
        # without holding an arbitrarily large upload in memory.
        data = file.file.read(limit + 1)
        try:
            record = services.library.add(user, file.filename or "", data)
        except UploadRejected as error:
            code = (
                status.HTTP_413_CONTENT_TOO_LARGE
                if len(data) > limit
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            )
            raise HTTPException(code, str(error)) from error
        except DuplicateDocument as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        return _document_out(record)

    @app.get("/documents")
    def list_documents(services: ServicesDep, user: UserDep) -> list[DocumentOut]:
        return [_document_out(record) for record in services.library.list(user)]

    @app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(document_id: str, services: ServicesDep, user: UserDep) -> None:
        # A document of another user is reported as missing, not as forbidden,
        # so ids cannot be probed.
        if not services.library.delete(user, document_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")

    @app.get("/messages")
    def list_messages(services: ServicesDep, user: UserDep, limit: int = 50) -> list[MessageOut]:
        limit = max(1, min(limit, 200))
        return [_message_out(m) for m in services.library.history(user, limit)]

    return app


def _document_out(record: DocumentRecord) -> DocumentOut:
    return DocumentOut(
        id=record.id,
        filename=record.filename,
        size_bytes=record.size_bytes,
        chunks=record.chunks,
        status=record.status,
        error=record.error,
        created_at=record.created_at,
    )


def _message_out(record: MessageRecord) -> MessageOut:
    return MessageOut(
        id=record.id,
        question=record.question,
        answer=record.answer,
        model=record.model,
        answerable=record.answerable,
        sources=record.sources,
        created_at=record.created_at,
    )


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
